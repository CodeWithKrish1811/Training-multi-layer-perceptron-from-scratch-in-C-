#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "matrix.h"
#include "nn.h"

int main() {
    // 1. Seed the random number generator for reproducibility.
    //    Using a fixed seed ensures that training runs identically every time,
    //    making correctness checks deterministic.
    srand(42);

    printf("=== Training Multi-Layer Perceptron from Scratch in C ===\n\n");

    // 2. Define the XOR problem training dataset
    //    XOR is a non-linearly separable problem, which requires a hidden layer to solve.
    double xor_inputs[4][2] = {
        {0.0, 0.0},
        {0.0, 1.0},
        {1.0, 0.0},
        {1.0, 1.0}
    };
    
    double xor_targets[4][1] = {
        {0.0},
        {1.0},
        {1.0},
        {0.0}
    };

    // 3. Define the Network Architecture:
    //    - 3 Layers: Input (2 units) -> Hidden (4 units, ReLU) -> Output (1 unit, Sigmoid)
    int layer_sizes[] = {2, 8, 1};
    ActivationType activations[] = {ACTIVATION_RELU, ACTIVATION_SIGMOID};
    int num_layers = 2; // Number of weighted connections/layers

    printf("Configuring MLP architecture:\n");
    printf("  Input layer size: %d\n", layer_sizes[0]);
    printf("  Hidden layer size: %d (Activation: ReLU)\n", layer_sizes[1]);
    printf("  Output layer size: %d (Activation: Sigmoid)\n\n", layer_sizes[2]);

    NeuralNetwork *nn = network_alloc(num_layers, layer_sizes, activations);
    if (!nn) {
        fprintf(stderr, "Error: Neural Network allocation failed.\n");
        return EXIT_FAILURE;
    }

    // 4. Pre-allocate training buffers
    Matrix *input_buffer = matrix_alloc(2, 1);
    Matrix *target_buffer = matrix_alloc(1, 1);
    if (!input_buffer || !target_buffer) {
        fprintf(stderr, "Error: Training buffer allocation failed.\n");
        network_free(nn);
        matrix_free(input_buffer);
        matrix_free(target_buffer);
        return EXIT_FAILURE;
    }

    // 5. Training loop
    int epochs = 15000;
    double learning_rate = 0.15;
    printf("Starting training: epochs = %d, learning_rate = %.2f...\n", epochs, learning_rate);

    for (int epoch = 1; epoch <= epochs; epoch++) {
        double epoch_loss = 0.0;
        
        // Loop over all 4 XOR samples (Stochastic/Mini-batch Gradient Descent style)
        for (int i = 0; i < 4; i++) {
            // Load input data
            input_buffer->data[0] = xor_inputs[i][0];
            input_buffer->data[1] = xor_inputs[i][1];
            
            // Load target data
            target_buffer->data[0] = xor_targets[i][0];

            // Forward pass: compute predictions
            Matrix *output = network_forward(nn, input_buffer);
            
            // Accumulate MSE loss
            epoch_loss += network_calculate_mse(output, target_buffer);

            // Backward pass: compute gradients w.r.t weights/biases
            network_backward(nn, input_buffer, target_buffer);

            // Update parameters using gradient descent steps
            network_update_weights(nn, learning_rate);
        }

        epoch_loss /= 4.0; // Average loss across samples

        // Print training progress at regular intervals
        if (epoch % 1500 == 0 || epoch == 1) {
            printf("  Epoch %5d/%d - Average MSE Loss: %.6f\n", epoch, epochs, epoch_loss);
        }
    }

    printf("\nTraining complete!\n\n");

    // 6. Test and Verify the trained model's predictions
    printf("=== Final Model Predictions ===\n");
    for (int i = 0; i < 4; i++) {
        input_buffer->data[0] = xor_inputs[i][0];
        input_buffer->data[1] = xor_inputs[i][1];
        
        Matrix *output = network_forward(nn, input_buffer);
        double prediction = output->data[0];
        double target = xor_targets[i][0];
        
        // A prediction is correct if it matches the target binary value when rounded
        int rounded_pred = prediction >= 0.5 ? 1 : 0;
        const char *status = (rounded_pred == (int)target) ? "SUCCESS" : "FAIL";

        printf("  Input: [%.1f, %.1f] -> Target: %.1f -> Prediction: %8.6f (Class: %d) [%s]\n",
               input_buffer->data[0], input_buffer->data[1], target, prediction, rounded_pred, status);
    }

    // 7. Clean up all allocated memory resources.
    //    Proper teardown is critical in C development to prevent leaks.
    printf("\nDeallocating memory and freeing workspaces...\n");
    matrix_free(input_buffer);
    matrix_free(target_buffer);
    network_free(nn);

    printf("Cleanup complete. Program exiting successfully.\n");
    return EXIT_SUCCESS;
}
