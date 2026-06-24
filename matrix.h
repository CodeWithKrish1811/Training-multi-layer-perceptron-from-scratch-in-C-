#ifndef MATRIX_H
#define MATRIX_H

#include <stdbool.h>

/**
 * @brief Matrix structure representing a 2D matrix stored in a cache-friendly 1D flat array.
 */
typedef struct {
    int rows;
    int cols;
    double *data;
} Matrix;

// Memory Management
Matrix* matrix_alloc(int rows, int cols);
void matrix_free(Matrix *m);
void matrix_copy(Matrix *dest, const Matrix *src);

// Initialization & Visualizers
void matrix_fill(Matrix *m, double val);
void matrix_randomize(Matrix *m, double min_val, double max_val);
void matrix_print(const Matrix *m);

// Mathematical Operations
void matrix_add_inplace(Matrix *a, const Matrix *b);
void matrix_sub_dest(const Matrix *a, const Matrix *b, Matrix *dest);
void matrix_hadamard_inplace(Matrix *a, const Matrix *b);

// Element-wise Transformations
void matrix_apply_dest(const Matrix *src, Matrix *dest, double (*func)(double));
void matrix_apply_derivative_dest(const Matrix *zs, const Matrix *activations, Matrix *dest, double (*deriv_func)(double, double));

// General Matrix Multiply (GEMM)
// C = alpha * op(A) * op(B) + beta * C
// where op(X) is X if transX is false, and X^T if transX is true.
void matrix_gemm(const Matrix *A, bool transA, const Matrix *B, bool transB, Matrix *C, double alpha, double beta);

#endif // MATRIX_H
