#include "nn.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Helper to free a Layer safely if allocation fails halfway
void layer_free_partial(Layer *layer) {
    if (!layer) return;
    matrix_free(layer->weights);
    matrix_free(layer->biases);
    matrix_free(layer->zs);
    matrix_free(layer->activations);
    matrix_free(layer->d_weights);
    matrix_free(layer->d_biases);
    matrix_free(layer->d_activations);
    matrix_free(layer->d_zs);
    free(layer);
}

/**
 * @brief Allocates and initializes a single Layer, including weights, biases, workspaces, and gradients.
 */
Layer* layer_alloc(int input_dim, int output_dim, ActivationType act_type) {
    Layer *layer = (Layer*)malloc(sizeof(Layer));
    if (!layer) {
        fprintf(stderr, "Error: Failed to allocate Layer struct.\n");
        return NULL;
    }

    layer->input_dim = input_dim;
    layer->output_dim = output_dim;
    layer->activation_type = act_type;

    // Set activation and derivative function pointers based on type
    if (act_type == ACTIVATION_RELU) {
        layer->activation_func = relu_activation;
        layer->activation_deriv_func = relu_derivative;
    } else if (act_type == ACTIVATION_SIGMOID) {
        layer->activation_func = sigmoid_activation;
        layer->activation_deriv_func = sigmoid_derivative;
    } else {
        fprintf(stderr, "Error: Unknown activation type %d.\n", act_type);
        free(layer);
        return NULL;
    }

    // Allocate all parameters, workspaces, and gradients
    // This pre-allocation allows zero mallocs during training and testing loops.
    layer->weights = matrix_alloc(output_dim, input_dim);
    layer->biases = matrix_alloc(output_dim, 1);
    layer->zs = matrix_alloc(output_dim, 1);
    layer->activations = matrix_alloc(output_dim, 1);
    
    layer->d_weights = matrix_alloc(output_dim, input_dim);
    layer->d_biases = matrix_alloc(output_dim, 1);
    layer->d_activations = matrix_alloc(output_dim, 1);
    layer->d_zs = matrix_alloc(output_dim, 1);

    // If any matrix allocation failed, perform clean rollback to prevent leaks
    if (!layer->weights || !layer->biases || !layer->zs || !layer->activations ||
        !layer->d_weights || !layer->d_biases || !layer->d_activations || !layer->d_zs) {
        fprintf(stderr, "Error: Failed to allocate matrix workspaces inside layer.\n");
        layer_free_partial(layer);
        return NULL;
    }

    return layer;
}

/**
 * @brief Deallocates a Layer and all its underlying matrix allocations.
 */
void layer_free(Layer *layer) {
    layer_free_partial(layer);
}

/**
 * @brief Initializes layer weights using Xavier/He uniform bounds and sets biases to zero.
 */
void layer_initialize_parameters(Layer *layer) {
    if (!layer) return;

    double limit;
    if (layer->activation_type == ACTIVATION_RELU) {
        // He Uniform initialization for layers with ReLU activation:
        // limit = sqrt(6 / fan_in)
        limit = sqrt(6.0 / (double)layer->input_dim);
    } else {
        // Xavier/Glorot Uniform initialization for layers with Sigmoid activation:
        // limit = sqrt(6 / (fan_in + fan_out))
        limit = sqrt(6.0 / (double)(layer->input_dim + layer->output_dim));
    }

    matrix_randomize(layer->weights, -limit, limit);
    matrix_fill(layer->biases, 0.0);
}

/**
 * @brief Allocates the global NeuralNetwork MLP structure.
 */
NeuralNetwork* network_alloc(int num_layers, const int *layer_sizes, const ActivationType *activation_types) {
    if (num_layers <= 0 || !layer_sizes || !activation_types) {
        fprintf(stderr, "Error: Invalid parameters for network allocation.\n");
        return NULL;
    }

    NeuralNetwork *nn = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    if (!nn) {
        fprintf(stderr, "Error: Failed to allocate NeuralNetwork struct.\n");
        return NULL;
    }

    nn->num_layers = num_layers;
    nn->layers = (Layer**)malloc(num_layers * sizeof(Layer*));
    if (!nn->layers) {
        fprintf(stderr, "Error: Failed to allocate layers pointer array.\n");
        free(nn);
        return NULL;
    }

    // Allocate layers sequentially
    for (int i = 0; i < num_layers; i++) {
        int input_dim = layer_sizes[i];
        int output_dim = layer_sizes[i + 1];
        nn->layers[i] = layer_alloc(input_dim, output_dim, activation_types[i]);
        if (!nn->layers[i]) {
            // Free previously allocated layers and clean up on failure
            for (int j = 0; j < i; j++) {
                layer_free(nn->layers[j]);
            }
            free(nn->layers);
            free(nn);
            return NULL;
        }
        // Immediately initialize weights and biases
        layer_initialize_parameters(nn->layers[i]);
    }

    return nn;
}

/**
 * @brief Deallocates the neural network and all constituent layers.
 */
void network_free(NeuralNetwork *nn) {
    if (nn) {
        if (nn->layers) {
            for (int i = 0; i < nn->num_layers; i++) {
                layer_free(nn->layers[i]);
            }
            free(nn->layers);
        }
        free(nn);
    }
}

/**
 * @brief Performs a complete forward pass through the Neural Network.
 *        z^[l] = W^[l] * a^[l-1] + b^[l]
 *        a^[l] = g^[l](z^[l])
 */
Matrix* network_forward(NeuralNetwork *nn, const Matrix *input) {
    if (!nn || !input) return NULL;

    const Matrix *prev_activation = input;
    for (int l = 0; l < nn->num_layers; l++) {
        Layer *layer = nn->layers[l];

        // 1. Compute pre-activation zs:
        //    zs = biases
        matrix_copy(layer->zs, layer->biases);
        //    zs = weights * prev_activation + zs (which is weights * prev_activation + biases)
        //    Here we use matrix_gemm with alpha=1.0 and beta=1.0
        matrix_gemm(layer->weights, false, prev_activation, false, layer->zs, 1.0, 1.0);

        // 2. Compute post-activation outputs:
        //    activations = activation_func(zs)
        matrix_apply_dest(layer->zs, layer->activations, layer->activation_func);

        // Current layer's activations become the input to the next layer
        prev_activation = layer->activations;
    }

    // Return the activations of the final output layer
    return nn->layers[nn->num_layers - 1]->activations;
}

/**
 * @brief Implements the backpropagation algorithm.
 *        Calculates gradients of the loss w.r.t weights, biases, and activations
 *        working backwards from the output layer to the first hidden layer.
 */
void network_backward(NeuralNetwork *nn, const Matrix *input, const Matrix *target) {
    if (!nn || !input || !target) return;
    int L = nn->num_layers;

    // 1. Seed the backpropagation at the output layer:
    //    For Mean Squared Error Loss L = 0.5 * (a^[L] - y)^2, the derivative w.r.t activations is:
    //    d_activations = a^[L] - y
    Layer *output_layer = nn->layers[L - 1];
    matrix_sub_dest(output_layer->activations, target, output_layer->d_activations);

    // 2. Loop backwards through layers
    for (int l = L - 1; l >= 0; l--) {
        Layer *layer = nn->layers[l];

        // 3. Compute delta (gradient w.r.t zs):
        //    delta = d_zs = d_activations * g'(z)
        //    where g'(z) is evaluated using the pre-activation zs and post-activation values.
        matrix_apply_derivative_dest(layer->zs, layer->activations, layer->d_zs, layer->activation_deriv_func);
        matrix_hadamard_inplace(layer->d_zs, layer->d_activations);

        // 4. Compute weight gradients:
        //    d_weights = delta * (a^[l-1])^T
        //    If it's the first layer (l=0), the previous activations are the input.
        const Matrix *a_prev = (l == 0) ? input : nn->layers[l - 1]->activations;
        //    d_weights = 1.0 * d_zs * a_prev^T + 0.0 * d_weights
        matrix_gemm(layer->d_zs, false, a_prev, true, layer->d_weights, 1.0, 0.0);

        // 5. Compute bias gradients:
        //    d_biases = delta (since z = W*a + b, dz/db = I)
        matrix_copy(layer->d_biases, layer->d_zs);

        // 6. Backpropagate error to previous layer:
        //    d_activations^[l-1] = (W^[l])^T * delta
        //    This calculates the loss gradient for the activations of layer l-1,
        //    which becomes the incoming d_activations for that layer's backward step.
        if (l > 0) {
            Layer *prev_layer = nn->layers[l - 1];
            //    prev_layer->d_activations = 1.0 * (weights^T * d_zs) + 0.0 * prev_layer->d_activations
            matrix_gemm(layer->weights, true, layer->d_zs, false, prev_layer->d_activations, 1.0, 0.0);
        }
    }
}

/**
 * @brief Updates parameters using Gradient Descent optimization.
 *        W = W - learning_rate * d_weights
 *        b = b - learning_rate * d_biases
 */
void network_update_weights(NeuralNetwork *nn, double learning_rate) {
    if (!nn) return;

    for (int l = 0; l < nn->num_layers; l++) {
        Layer *layer = nn->layers[l];

        // Update weights
        int w_size = layer->weights->rows * layer->weights->cols;
        for (int i = 0; i < w_size; i++) {
            layer->weights->data[i] -= learning_rate * layer->d_weights->data[i];
        }

        // Update biases
        int b_size = layer->biases->rows * layer->biases->cols;
        for (int i = 0; i < b_size; i++) {
            layer->biases->data[i] -= learning_rate * layer->d_biases->data[i];
        }
    }
}

/**
 * @brief Computes Mean Squared Error (MSE) loss:
 *        Loss = 0.5 * sum((output - target)^2)
 */
double network_calculate_mse(const Matrix *output, const Matrix *target) {
    if (!output || !target || output->rows != target->rows || output->cols != target->cols) {
        return -1.0;
    }
    double error = 0.0;
    int size = output->rows * output->cols;
    for (int i = 0; i < size; i++) {
        double diff = output->data[i] - target->data[i];
        error += diff * diff;
    }
    return 0.5 * error;
}

// ==========================================
// Activation Functions and Their Derivatives
// ==========================================

/**
 * @brief Rectified Linear Unit activation: f(z) = max(0, z)
 */
double relu_activation(double z) {
    return z > 0.0 ? z : 0.0;
}

/**
 * @brief Derivative of ReLU: f'(z) = 1 if z > 0, else 0
 */
double relu_derivative(double z, double a) {
    (void)a; // Unused for ReLU, but signature compatibility is kept
    return z > 0.0 ? 1.0 : 0.0;
}

/**
 * @brief Sigmoid activation: f(z) = 1 / (1 + e^-z)
 */
double sigmoid_activation(double z) {
    return 1.0 / (1.0 + exp(-z));
}

/**
 * @brief Derivative of Sigmoid: f'(z) = a * (1 - a) where a = sigmoid(z)
 */
double sigmoid_derivative(double z, double a) {
    (void)z; // Unused for Sigmoid when using activations directly
    return a * (1.0 - a);
}
