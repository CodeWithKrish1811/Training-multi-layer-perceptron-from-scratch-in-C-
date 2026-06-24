#include "matrix.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/**
 * @brief Allocates memory for a Matrix struct and its underlying 1D flat array.
 * @param rows Number of rows in the matrix.
 * @param cols Number of columns in the matrix.
 * @return Pointer to the allocated Matrix, or NULL on failure.
 */
Matrix* matrix_alloc(int rows, int cols) {
    if (rows <= 0 || cols <= 0) {
        fprintf(stderr, "Error: Invalid matrix dimensions %dx%d for allocation.\n", rows, cols);
        return NULL;
    }

    Matrix *m = (Matrix*)malloc(sizeof(Matrix));
    if (!m) {
        fprintf(stderr, "Error: Failed to allocate memory for Matrix struct.\n");
        return NULL;
    }

    m->rows = rows;
    m->cols = cols;
    // Single contiguous malloc is extremely cache-friendly and minimizes heap fragmentation
    m->data = (double*)malloc(rows * cols * sizeof(double));
    if (!m->data) {
        fprintf(stderr, "Error: Failed to allocate memory for Matrix data array of size %d.\n", rows * cols);
        free(m);
        return NULL;
    }

    return m;
}

/**
 * @brief Safely deallocates a Matrix struct and its data.
 * @param m Pointer to the Matrix to free.
 */
void matrix_free(Matrix *m) {
    if (m) {
        if (m->data) {
            free(m->data);
        }
        free(m);
    }
}

/**
 * @brief Copies elements from one matrix to another. Both must have matching dimensions.
 * @param dest Destination matrix.
 * @param src Source matrix.
 */
void matrix_copy(Matrix *dest, const Matrix *src) {
    if (!dest || !src || !dest->data || !src->data) {
        fprintf(stderr, "Error: Cannot copy NULL matrix pointers.\n");
        return;
    }
    if (dest->rows != src->rows || dest->cols != src->cols) {
        fprintf(stderr, "Error: Dimension mismatch in matrix_copy: dest is %dx%d, src is %dx%d.\n",
                dest->rows, dest->cols, src->rows, src->cols);
        return;
    }
    memcpy(dest->data, src->data, dest->rows * dest->cols * sizeof(double));
}

/**
 * @brief Fills a matrix with a scalar value.
 * @param m Matrix to fill.
 * @param val Scalar value to assign to all elements.
 */
void matrix_fill(Matrix *m, double val) {
    if (!m || !m->data) return;
    int size = m->rows * m->cols;
    for (int i = 0; i < size; i++) {
        m->data[i] = val;
    }
}

/**
 * @brief Fills a matrix with random values drawn from a uniform distribution.
 * @param m Matrix to randomize.
 * @param min_val Minimum bound of the uniform distribution.
 * @param max_val Maximum bound of the uniform distribution.
 */
void matrix_randomize(Matrix *m, double min_val, double max_val) {
    if (!m || !m->data) return;
    int size = m->rows * m->cols;
    double range = max_val - min_val;
    for (int i = 0; i < size; i++) {
        // Standard random generation scaling to [min_val, max_val]
        double scale = (double)rand() / (double)RAND_MAX;
        m->data[i] = min_val + scale * range;
    }
}

/**
 * @brief Prints the matrix dimensions and contents formatted to standard output.
 * @param m Matrix to print.
 */
void matrix_print(const Matrix *m) {
    if (!m || !m->data) {
        printf("Matrix: NULL\n");
        return;
    }
    printf("Matrix (%dx%d):\n", m->rows, m->cols);
    for (int i = 0; i < m->rows; i++) {
        printf("  [");
        for (int j = 0; j < m->cols; j++) {
            printf(" %8.4f", m->data[i * m->cols + j]);
        }
        printf(" ]\n");
    }
}

/**
 * @brief Computes in-place addition: a = a + b.
 * @param a Target matrix that will store the result.
 * @param b Addend matrix.
 */
void matrix_add_inplace(Matrix *a, const Matrix *b) {
    if (!a || !b || !a->data || !b->data) return;
    if (a->rows != b->rows || a->cols != b->cols) {
        fprintf(stderr, "Error: Dimension mismatch in matrix_add_inplace.\n");
        return;
    }
    int size = a->rows * a->cols;
    for (int i = 0; i < size; i++) {
        a->data[i] += b->data[i];
    }
}

/**
 * @brief Computes subtraction: dest = a - b.
 * @param a Minuend matrix.
 * @param b Subtrahend matrix.
 * @param dest Destination matrix to store the difference.
 */
void matrix_sub_dest(const Matrix *a, const Matrix *b, Matrix *dest) {
    if (!a || !b || !dest || !a->data || !b->data || !dest->data) return;
    if (a->rows != b->rows || a->cols != b->cols || a->rows != dest->rows || a->cols != dest->cols) {
        fprintf(stderr, "Error: Dimension mismatch in matrix_sub_dest.\n");
        return;
    }
    int size = a->rows * a->cols;
    for (int i = 0; i < size; i++) {
        dest->data[i] = a->data[i] - b->data[i];
    }
}

/**
 * @brief Computes in-place Hadamard product (element-wise multiplication): a = a * b.
 * @param a Target matrix that will store the result.
 * @param b Factor matrix.
 */
void matrix_hadamard_inplace(Matrix *a, const Matrix *b) {
    if (!a || !b || !a->data || !b->data) return;
    if (a->rows != b->rows || a->cols != b->cols) {
        fprintf(stderr, "Error: Dimension mismatch in matrix_hadamard_inplace.\n");
        return;
    }
    int size = a->rows * a->cols;
    for (int i = 0; i < size; i++) {
        a->data[i] *= b->data[i];
    }
}

/**
 * @brief Applies a mathematical function element-wise on a source matrix and stores it in destination.
 * @param src Source matrix.
 * @param dest Destination matrix.
 * @param func Function pointer for the transformation.
 */
void matrix_apply_dest(const Matrix *src, Matrix *dest, double (*func)(double)) {
    if (!src || !dest || !src->data || !dest->data) return;
    if (src->rows != dest->rows || src->cols != dest->cols) {
        fprintf(stderr, "Error: Dimension mismatch in matrix_apply_dest.\n");
        return;
    }
    int size = src->rows * src->cols;
    for (int i = 0; i < size; i++) {
        dest->data[i] = func(src->data[i]);
    }
}

/**
 * @brief Applies an activation derivative function element-wise, considering both z and activation values.
 * @param zs Pre-activation values.
 * @param activations Post-activation values.
 * @param dest Destination matrix to store calculated derivatives.
 * @param deriv_func Function pointer to the derivative function.
 */
void matrix_apply_derivative_dest(const Matrix *zs, const Matrix *activations, Matrix *dest, double (*deriv_func)(double, double)) {
    if (!zs || !activations || !dest || !zs->data || !activations->data || !dest->data) return;
    if (zs->rows != activations->rows || zs->cols != activations->cols || zs->rows != dest->rows || zs->cols != dest->cols) {
        fprintf(stderr, "Error: Dimension mismatch in matrix_apply_derivative_dest.\n");
        return;
    }
    int size = zs->rows * zs->cols;
    for (int i = 0; i < size; i++) {
        dest->data[i] = deriv_func(zs->data[i], activations->data[i]);
    }
}

/**
 * @brief High-performance General Matrix Multiplication (GEMM) in C.
 *        C = alpha * op(A) * op(B) + beta * C
 *        This function supports transpose flags for both input matrices, preventing
 *        unnecessary memory allocation and transposition routines in the backprop loop.
 */
void matrix_gemm(const Matrix *A, bool transA, const Matrix *B, bool transB, Matrix *C, double alpha, double beta) {
    if (!A || !B || !C || !A->data || !B->data || !C->data) {
        fprintf(stderr, "Error: NULL pointers in matrix_gemm.\n");
        return;
    }

    int rows_A_op = transA ? A->cols : A->rows;
    int cols_A_op = transA ? A->rows : A->cols;
    int rows_B_op = transB ? B->cols : B->rows;
    int cols_B_op = transB ? B->rows : B->cols;

    if (cols_A_op != rows_B_op) {
        fprintf(stderr, "Error: Dimension mismatch in matrix_gemm multiplication: op(A) cols (%d) != op(B) rows (%d).\n",
                cols_A_op, rows_B_op);
        return;
    }
    if (C->rows != rows_A_op || C->cols != cols_B_op) {
        fprintf(stderr, "Error: Destination matrix C (%dx%d) dimensions mismatch with output size (%dx%d).\n",
                C->rows, C->cols, rows_A_op, cols_B_op);
        return;
    }

    // Standard matrix multiplication nested loops, optimized with transpositions inline
    for (int i = 0; i < C->rows; i++) {
        for (int j = 0; j < C->cols; j++) {
            double sum = 0.0;
            for (int k = 0; k < cols_A_op; k++) {
                // Inline transposition indexing to avoid allocating memory for transposition copies
                double valA = transA ? A->data[k * A->cols + i] : A->data[i * A->cols + k];
                double valB = transB ? B->data[j * B->cols + k] : B->data[k * B->cols + j];
                sum += valA * valB;
            }
            C->data[i * C->cols + j] = alpha * sum + beta * C->data[i * C->cols + j];
        }
    }
}
