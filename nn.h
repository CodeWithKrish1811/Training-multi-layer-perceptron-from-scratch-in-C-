#ifndef NN_H
#define NN_H

#include "matrix.h"

// Activation function identifiers
typedef enum {
    ACTIVATION_RELU,
    ACTIVATION_SIGMOID
} ActivationType;

// Typedefs for cleaner function pointers
typedef double (*ActivationFunc)(double);
typedef double (*DerivativeFunc)(double, double);

/**
 * @brief Structure representing a single dense layer in the Neural Network.
 *        All variables (weights, biases, activations, workspaces, and gradients)
 *        are pre-allocated to ensure zero allocations during backprop loops.
 */
typedef struct {
    int input_dim;
    int output_dim;

    // Parameters
    Matrix *weights;            // Weights matrix (output_dim x input_dim)
    Matrix *biases;             // Biases matrix (output_dim x 1)

    // Pre-allocated Workspaces for Forward Pass (zs and activations)
    Matrix *zs;                 // Pre-activation weighted input (output_dim x 1)
    Matrix *activations;        // Post-activation output (output_dim x 1)

    // Pre-allocated Workspaces for Backward Pass (gradients and errors)
    Matrix *d_weights;          // Gradient of loss w.r.t weights (output_dim x input_dim)
    Matrix *d_biases;           // Gradient of loss w.r.t biases (output_dim x 1)
    Matrix *d_activations;      // Gradient of loss w.r.t activations (output_dim x 1)
    Matrix *d_zs;               // Gradient of loss w.r.t zs (delta) (output_dim x 1)

    // Activation logic pointers
    ActivationFunc activation_func;
    DerivativeFunc activation_deriv_func;
    ActivationType activation_type;
} Layer;

/**
 * @brief Structure representing the entire Multi-Layer Perceptron (MLP).
 */
typedef struct {
    int num_layers;             // Number of trainable layers (excluding input layer)
    Layer **layers;             // Array of Layer pointers
} NeuralNetwork;

// Layer Lifecycle and Initialization
Layer* layer_alloc(int input_dim, int output_dim, ActivationType act_type);
void layer_free(Layer *layer);
void layer_initialize_parameters(Layer *layer);

// Neural Network Lifecycle
NeuralNetwork* network_alloc(int num_layers, const int *layer_sizes, const ActivationType *activation_types);
void network_free(NeuralNetwork *nn);

// Core Backpropagation Routines
Matrix* network_forward(NeuralNetwork *nn, const Matrix *input);
void network_backward(NeuralNetwork *nn, const Matrix *input, const Matrix *target);
void network_update_weights(NeuralNetwork *nn, double learning_rate);

// Evaluation Helpers
double network_calculate_mse(const Matrix *output, const Matrix *target);

// Activation Functions and Derivatives
double relu_activation(double z);
double relu_derivative(double z, double a);
double sigmoid_activation(double z);
double sigmoid_derivative(double z, double a);

#endif // NN_H
