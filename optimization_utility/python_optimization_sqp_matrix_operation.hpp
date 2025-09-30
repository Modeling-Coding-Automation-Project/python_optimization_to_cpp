/**
 * @file python_optimization_sqp_matrix_operation.hpp
 *
 * @brief Matrix operation utilities for SQP-based Python optimization, ported
 * to C++.
 *
 * This header provides a collection of template-based matrix operations for use
 * in sequential quadratic programming (SQP) optimization routines, supporting
 * fixed-size matrices and compile-time recursion for performance and type
 * safety.
 *
 * Main Features:
 * - Row and column extraction and assignment between matrices.
 * - Element-wise matrix multiplication.
 * - Quadratic form calculations (weighted and unweighted).
 * - Penalty calculations for output constraints (Y limits), including active
 * set detection.
 * - Contract operations for Hessian and gradient matrices with lambda weights.
 * - Masking and active set management for free variables.
 * - Saturation and inversion operations for control horizons and diagonal
 * matrices.
 */
#ifndef __PYTHON_OPTIMIZATION_SQP_MATRIX_OPERATION_HPP__
#define __PYTHON_OPTIMIZATION_SQP_MATRIX_OPERATION_HPP__

#include "python_control.hpp"
#include "python_numpy.hpp"

#include <array>
#include <functional>
#include <type_traits>

namespace PythonOptimization {

namespace MatrixOperation {

namespace SetRow {

// Row recursion when I_idx > 0
template <typename Matrix_Out_Type, typename Matrix_In_Type, std::size_t I_idx>
struct Row {
  /**
   * @brief Recursively assigns out_matrix(I_idx, row_index) = in_matrix(I_idx,
   * 0).
   *
   * @tparam Matrix_Out_Type Output matrix type.
   * @tparam Matrix_In_Type  Input matrix type (expected ROWS == 1).
   * @tparam I_idx           Current column index in the recursion (0-based).
   * @param out_matrix The destination matrix.
   * @param in_matrix  The source 1-row matrix.
   * @param row_index  The destination row index (runtime parameter).
   */
  static inline void compute(Matrix_Out_Type &out_matrix,
                             const Matrix_In_Type &in_matrix,
                             const std::size_t &row_index) {

    out_matrix.access(I_idx, row_index) = in_matrix.template get<I_idx, 0>();

    Row<Matrix_Out_Type, Matrix_In_Type, I_idx - 1>::compute(
        out_matrix, in_matrix, row_index);
  }
};

// Row recursion termination for I_idx == 0
/**
 * @brief Specialization of the Row struct for the base case where I_idx == 0.
 *
 * This struct provides a static inline compute function that assigns the value
 * from the first element (0, 0) of the input matrix to the corresponding
 * position (0, row_index) in the output matrix.
 *
 * @tparam Matrix_Out_Type Type of the output matrix.
 * @tparam Matrix_In_Type Type of the input matrix.
 * @param[out] out_matrix Reference to the output matrix where the value will be
 * assigned.
 * @param[in] in_matrix Const reference to the input matrix from which the value
 * is retrieved.
 * @param[in] row_index The row index in the output matrix where the value will
 * be assigned.
 */
template <typename Matrix_In_Type>
inline auto get_row(const Matrix_In_Type &in_matrix,
                    const std::size_t &row_index)
    -> PythonNumpy::DenseMatrix_Type<typename Matrix_In_Type::Value_Type,
                                     Matrix_In_Type::COLS, 1> {

  using Out_Type =
      PythonNumpy::DenseMatrix_Type<typename Matrix_In_Type::Value_Type,
                                    Matrix_In_Type::COLS, 1>;
  Out_Type out;

  GetRow::compute<Matrix_In_Type, Out_Type>(in_matrix, row_index, out);

  return out;
}

/**
 * @brief Calculates the quadratic form X * W * X^T for a given row vector X and
 * matrix W.
 *
 * This function computes the quadratic form by multiplying the transpose of X
 * with W, then multiplying the result by X. The result is extracted as a scalar
 * value.
 *
 * @tparam X_Type Type of the input vector X. Must have a static member COLS and
 * ROWS.
 * @tparam W_Type Type of the input matrix W. Must have a static member ROWS.
 * @param X Input row vector of type X_Type. Must have ROWS == 1.
 * @param W Input matrix of type W_Type. Must have ROWS == X_Type::COLS.
 * @return Scalar value representing the quadratic form.
 *
 * @note Compile-time assertions ensure that X is a row vector and that the
 * dimensions of W and X are compatible.
 */
template <typename X_Type, typename W_Type>
inline auto calculate_quadratic_form(const X_Type &X, const W_Type &W) ->
    typename X_Type::Value_Type {

  static_assert(W_Type::ROWS == X_Type::COLS, "W_Type::ROWS != X_Type::COLS");
  static_assert(X_Type::ROWS == 1, "X_Type::ROWS != 1");

  auto result = PythonNumpy::ATranspose_mul_B(X, W) * X;

  return result.template get<0, 0>();
}

/**
 * @brief Calculates the quadratic form of a row vector without weighting.
 *
 * This function computes the quadratic form X * X^T for a given row vector X,
 * where X is expected to have exactly one row (i.e., X_Type::ROWS == 1).
 * The result is the scalar value at position (0, 0) of the resulting matrix.
 *
 * @tparam X_Type Type of the input vector, must have a static member ROWS == 1
 * and a nested Value_Type.
 * @param X The input row vector.
 * @return The scalar result of the quadratic form (X * X^T).
 */
template <typename X_Type>
inline auto calculate_quadratic_no_weighted(const X_Type &X) ->
    typename X_Type::Value_Type {

  static_assert(X_Type::ROWS == 1, "X_Type::ROWS != 1");

  auto result = PythonNumpy::ATranspose_mul_B(X, X);

  return result.template get<0, 0>();
}

/* calculate Y_limit penalty */

namespace CalculateY_LimitPenalty {

/**
 * @brief Template struct for conditional minimum matrix operation in SQP
 * optimization.
 *
 * This struct serves as a template for performing conditional minimum
 * operations on matrices, typically used within Sequential Quadratic
 * Programming (SQP) optimization utilities. The actual implementation is
 * expected to be provided via template specialization.
 *
 * @tparam Y_Mat_Type         The type representing the input matrix.
 * @tparam Y_Min_Matrix_Type  The type representing the matrix to store minimum
 * values.
 * @tparam Out_Type           The output type for the operation.
 * @tparam I                  The number of rows (or a specific row index,
 * depending on usage).
 * @tparam J_idx              The number of columns (or a specific column index,
 * depending on usage).
 * @tparam limit_valid_flag   Boolean flag indicating if limit validation is
 * enabled.
 */
template <typename Y_Mat_Type, typename Y_Min_Matrix_Type, typename Out_Type,
          std::size_t I, std::size_t J_idx, bool limit_valid_flag>
struct MinConditional {};

/**
 * @brief Specialization of MinConditional for conditional minimum penalty
 * computation.
 *
 * This struct template computes a penalty when an element in the Y_horizon
 * matrix is less than the corresponding element in the Y_min_matrix. If the
 * condition is met, it sets the penalty in the Y_limit_penalty output matrix at
 * the same index.
 *
 * @tparam Y_Mat_Type         Type of the input matrix Y_horizon.
 * @tparam Y_Min_Matrix_Type  Type of the minimum constraint matrix
 * Y_min_matrix.
 * @tparam Out_Type           Type of the output penalty matrix Y_limit_penalty.
 * @tparam I                  Row index (compile-time constant).
 * @tparam J_idx              Column index (compile-time constant).
 *
 * @param Y_horizon           Input matrix containing values to be checked.
 * @param Y_min_matrix        Matrix containing minimum allowed values.
 * @param Y_limit_penalty     Output matrix where penalty is set if condition is
 * met.
 *
 * The penalty is computed as (y - y_min) and set only if y < y_min.
 */
template <typename Y_Mat_Type, typename Y_Min_Matrix_Type, typename Out_Type,
          std::size_t I, std::size_t J_idx>
struct MinConditional<Y_Mat_Type, Y_Min_Matrix_Type, Out_Type, I, J_idx, true> {
  static void compute(const Y_Mat_Type &Y_horizon,
                      const Y_Min_Matrix_Type &Y_min_matrix,
                      Out_Type &Y_limit_penalty) {

    const auto y = Y_horizon.template get<I, J_idx>();
    const auto y_min = Y_min_matrix.template get<I, J_idx>();

    if (y < y_min) {
      Y_limit_penalty.template set<I, J_idx>(y - y_min);
    }
  }
};

/**
 * @brief Specialization of MinConditional struct for the case when the
 * condition is false.
 *
 * This specialization provides a no-op implementation of the compute function.
 * When the boolean template parameter is false, this struct's compute method
 * does nothing and simply ignores its arguments.
 *
 * @tparam Y_Mat_Type         Type of the input matrix Y_horizon.
 * @tparam Y_Min_Matrix_Type  Type of the input matrix Y_min_matrix.
 * @tparam Out_Type           Type of the output Y_limit_penalty.
 * @tparam I                  Row index or size parameter.
 * @tparam J_idx              Column index or size parameter.
 *
 * @param Y_horizon           Input matrix (unused).
 * @param Y_min_matrix        Input matrix (unused).
 * @param Y_limit_penalty     Output parameter (unused).
 */
template <typename Y_Mat_Type, typename Y_Min_Matrix_Type, typename Out_Type,
          std::size_t I, std::size_t J_idx>
struct MinConditional<Y_Mat_Type, Y_Min_Matrix_Type, Out_Type, I, J_idx,
                      false> {
  static void compute(const Y_Mat_Type &Y_horizon,
                      const Y_Min_Matrix_Type &Y_min_matrix,
                      Out_Type &Y_limit_penalty) {

    static_cast<void>(Y_horizon);
    static_cast<void>(Y_min_matrix);
    static_cast<void>(Y_limit_penalty);
    /* Do Nothing */
  }
};

/**
 * @brief Template struct for conditional maximum matrix operations.
 *
 * This struct template is designed to perform conditional maximum operations
 * on matrices, with customizable types and compile-time parameters.
 *
 * @tparam Y_Mat_Type         The type representing the input matrix.
 * @tparam Y_Max_Matrix_Type  The type representing the matrix used for maximum
 * value comparison.
 * @tparam Out_Type           The type representing the output/result.
 * @tparam I                  Compile-time row index or dimension parameter.
 * @tparam J_idx              Compile-time column index or dimension parameter.
 * @tparam limit_valid_flag   Boolean flag indicating whether a limit condition
 * is valid.
 */
template <typename Y_Mat_Type, typename Y_Max_Matrix_Type, typename Out_Type,
          std::size_t I, std::size_t J_idx, bool limit_valid_flag>
struct MaxConditional {};

/**
 * @brief Specialization of MaxConditional for applying a conditional maximum
 * penalty.
 *
 * This struct template provides a static compute function that compares an
 * element at position (I, J_idx) in the input matrix Y_horizon with the
 * corresponding element in Y_max_matrix. If the value in Y_horizon exceeds the
 * maximum allowed value in Y_max_matrix, the difference (penalty) is set in the
 * output matrix Y_limit_penalty at the same position.
 *
 * @tparam Y_Mat_Type         Type of the input matrix Y_horizon.
 * @tparam Y_Max_Matrix_Type  Type of the matrix containing maximum allowed
 * values.
 * @tparam Out_Type           Type of the output matrix for storing penalties.
 * @tparam I                  Row index (compile-time constant).
 * @tparam J_idx              Column index (compile-time constant).
 *
 * @note This specialization is enabled when the last template parameter is
 * true.
 *
 * @param Y_horizon       Input matrix containing current values.
 * @param Y_max_matrix    Matrix containing maximum allowed values.
 * @param Y_limit_penalty Output matrix where the penalty is set if the limit is
 * exceeded.
 */
template <typename Y_Mat_Type, typename Y_Max_Matrix_Type, typename Out_Type,
          std::size_t I, std::size_t J_idx>
struct MaxConditional<Y_Mat_Type, Y_Max_Matrix_Type, Out_Type, I, J_idx, true> {
  static void compute(const Y_Mat_Type &Y_horizon,
                      const Y_Max_Matrix_Type &Y_max_matrix,
                      Out_Type &Y_limit_penalty) {

    const auto y = Y_horizon.template get<I, J_idx>();
    const auto y_max = Y_max_matrix.template get<I, J_idx>();

    if (y > y_max) {
      Y_limit_penalty.template set<I, J_idx>(y - y_max);
    }
  }
};

/**
 * @brief Specialization of MaxConditional struct for the case when the
 * condition is false.
 *
 * This specialization provides a static compute function that takes three
 * parameters:
 * - Y_horizon: A constant reference to a matrix or data structure representing
 * the horizon values.
 * - Y_max_matrix: A constant reference to a matrix or data structure
 * representing the maximum values.
 * - Y_limit_penalty: A reference to an output variable for storing the penalty
 * or result.
 *
 * When the condition is false, this function performs no operation (no-op) on
 * the inputs. The parameters are explicitly marked as unused to avoid compiler
 * warnings.
 *
 * @tparam Y_Mat_Type         Type of the horizon matrix.
 * @tparam Y_Max_Matrix_Type  Type of the maximum matrix.
 * @tparam Out_Type           Type of the output variable.
 * @tparam I                  Compile-time index or size parameter.
 * @tparam J_idx              Compile-time index or size parameter.
 */
template <typename Y_Mat_Type, typename Y_Max_Matrix_Type, typename Out_Type,
          std::size_t I, std::size_t J_idx>
struct MaxConditional<Y_Mat_Type, Y_Max_Matrix_Type, Out_Type, I, J_idx,
                      false> {
  static void compute(const Y_Mat_Type &Y_horizon,
                      const Y_Max_Matrix_Type &Y_max_matrix,
                      Out_Type &Y_limit_penalty) {

    static_cast<void>(Y_horizon);
    static_cast<void>(Y_max_matrix);
    static_cast<void>(Y_limit_penalty);
    /* Do Nothing */
  }
};

// Column recursion when J_idx > 0
/**
 * @brief Template struct to recursively process a column of matrix operations
 * for optimization.
 *
 * This struct template performs conditional minimum and maximum operations on a
 * specific column (indexed by J_idx) of the input matrices, and accumulates the
 * results into the output penalty variable. It recursively processes each
 * column by decrementing the column index (J_idx) until the base case is
 * reached (not shown here).
 *
 * @tparam Y_Mat_Type         Type of the main input matrix (Y_horizon).
 * @tparam Y_Min_Matrix_Type  Type of the minimum constraint matrix.
 * @tparam Y_Max_Matrix_Type  Type of the maximum constraint matrix.
 * @tparam Out_Type           Type of the output penalty accumulator.
 * @tparam M                  Number of rows in the matrices.
 * @tparam N                  Number of columns in the matrices.
 * @tparam I                  Current row index being processed.
 * @tparam J_idx              Current column index being processed (recursively
 * decremented).
 *
 * @note This struct assumes the existence of MinConditional and MaxConditional
 * templates, as well as a SparseAvailable_Type::lists static member in the
 * min/max matrix types to determine sparsity for conditional computation.
 *
 * @param Y_horizon           The main input matrix for the optimization
 * horizon.
 * @param Y_min_matrix        The matrix containing minimum constraints.
 * @param Y_max_matrix        The matrix containing maximum constraints.
 * @param Y_limit_penalty     The output variable accumulating penalty values
 * for constraint violations.
 */
template <typename Y_Mat_Type, typename Y_Min_Matrix_Type,
          typename Y_Max_Matrix_Type, typename Out_Type, std::size_t M,
          std::size_t N, std::size_t I, std::size_t J_idx>
struct Column {
  static void compute(const Y_Mat_Type &Y_horizon,
                      const Y_Min_Matrix_Type &Y_min_matrix,
                      const Y_Max_Matrix_Type &Y_max_matrix,
                      Out_Type &Y_limit_penalty) {

    MinConditional<Y_Mat_Type, Y_Min_Matrix_Type, Out_Type, I, J_idx,
                   Y_Min_Matrix_Type::SparseAvailable_Type::lists[I][J_idx]>::
        compute(Y_horizon, Y_min_matrix, Y_limit_penalty);

    MaxConditional<Y_Mat_Type, Y_Max_Matrix_Type, Out_Type, I, J_idx,
                   Y_Max_Matrix_Type::SparseAvailable_Type::lists[I][J_idx]>::
        compute(Y_horizon, Y_max_matrix, Y_limit_penalty);

    Column<Y_Mat_Type, Y_Min_Matrix_Type, Y_Max_Matrix_Type, Out_Type, M, N, I,
           (J_idx - 1)>::compute(Y_horizon, Y_min_matrix, Y_max_matrix,
                                 Y_limit_penalty);
  }
};

// Column recursion termination for J_idx == 0
/**
 * @brief Specialization of the Column struct for column index 0.
 *
 * This struct provides a static compute function that applies minimum and
 * maximum conditional operations to a specific column (column 0) of the input
 * matrices.
 *
 * @tparam Y_Mat_Type         Type of the input matrix representing the horizon
 * values.
 * @tparam Y_Min_Matrix_Type  Type of the matrix containing minimum constraints.
 * @tparam Y_Max_Matrix_Type  Type of the matrix containing maximum constraints.
 * @tparam Out_Type           Type of the output penalty accumulator.
 * @tparam M                  Number of rows in the matrices.
 * @tparam N                  Number of columns in the matrices.
 * @tparam I                  Row index for which the operation is performed.
 *
 * The compute function applies MinConditional and MaxConditional operations
 * for the specified row (I) and column (0), updating the Y_limit_penalty
 * accordingly.
 *
 * @param Y_horizon       Input matrix of horizon values.
 * @param Y_min_matrix    Matrix containing minimum constraint values.
 * @param Y_max_matrix    Matrix containing maximum constraint values.
 * @param Y_limit_penalty Output accumulator for the penalty values.
 */
template <typename Y_Mat_Type, typename Y_Min_Matrix_Type,
          typename Y_Max_Matrix_Type, typename Out_Type, std::size_t M,
          std::size_t N, std::size_t I>
struct Column<Y_Mat_Type, Y_Min_Matrix_Type, Y_Max_Matrix_Type, Out_Type, M, N,
              I, 0> {
  static void compute(const Y_Mat_Type &Y_horizon,
                      const Y_Min_Matrix_Type &Y_min_matrix,
                      const Y_Max_Matrix_Type &Y_max_matrix,
                      Out_Type &Y_limit_penalty) {

    MinConditional<Y_Mat_Type, Y_Min_Matrix_Type, Out_Type, I, 0,
                   Y_Min_Matrix_Type::SparseAvailable_Type::lists[I][0]>::
        compute(Y_horizon, Y_min_matrix, Y_limit_penalty);

    MaxConditional<Y_Mat_Type, Y_Max_Matrix_Type, Out_Type, I, 0,
                   Y_Max_Matrix_Type::SparseAvailable_Type::lists[I][0]>::
        compute(Y_horizon, Y_max_matrix, Y_limit_penalty);
  }
};

// Row recursion when I_idx > 0
/**
 * @brief Recursive template struct to compute operations over rows of matrices.
 *
 * This struct template recursively processes each row (indexed by I_idx) of the
 * input matrices, applying a computation defined in the Column struct for each
 * column in the row, and then recursing to the previous row until the base case
 * is reached.
 *
 * @tparam Y_Mat_Type         Type of the main input matrix (Y_horizon).
 * @tparam Y_Min_Matrix_Type  Type of the minimum constraint matrix
 * (Y_min_matrix).
 * @tparam Y_Max_Matrix_Type  Type of the maximum constraint matrix
 * (Y_max_matrix).
 * @tparam Out_Type           Type of the output accumulator (Y_limit_penalty).
 * @tparam M                  Number of rows in the matrices.
 * @tparam N                  Number of columns in the matrices.
 * @tparam I_idx              Current row index being processed (recursively
 * decremented).
 *
 * @param Y_horizon           The main input matrix.
 * @param Y_min_matrix        The matrix containing minimum constraints.
 * @param Y_max_matrix        The matrix containing maximum constraints.
 * @param Y_limit_penalty     Output accumulator for computed penalties or
 * results.
 */
template <typename Y_Mat_Type, typename Y_Min_Matrix_Type,
          typename Y_Max_Matrix_Type, typename Out_Type, std::size_t M,
          std::size_t N, std::size_t I_idx>
struct Row {
  static void compute(const Y_Mat_Type &Y_horizon,
                      const Y_Min_Matrix_Type &Y_min_matrix,
                      const Y_Max_Matrix_Type &Y_max_matrix,
                      Out_Type &Y_limit_penalty) {
    Column<Y_Mat_Type, Y_Min_Matrix_Type, Y_Max_Matrix_Type, Out_Type, M, N,
           I_idx, (N - 1)>::compute(Y_horizon, Y_min_matrix, Y_max_matrix,
                                    Y_limit_penalty);
    Row<Y_Mat_Type, Y_Min_Matrix_Type, Y_Max_Matrix_Type, Out_Type, M, N,
        (I_idx - 1)>::compute(Y_horizon, Y_min_matrix, Y_max_matrix,
                              Y_limit_penalty);
  }
};

// Row recursion termination for I_idx == 0
/**
 * @brief Specialization of the Row struct for the base case where the row index
 * is 0.
 *
 * This struct provides a static compute function that processes the first row
 * (row index 0) of the matrices involved in the SQP (Sequential Quadratic
 * Programming) matrix operation. It delegates the computation to the
 * corresponding Column specialization, starting from column index 0 up to (N -
 * 1).
 *
 * @tparam Y_Mat_Type         Type of the main matrix (Y_horizon).
 * @tparam Y_Min_Matrix_Type  Type of the minimum constraint matrix
 * (Y_min_matrix).
 * @tparam Y_Max_Matrix_Type  Type of the maximum constraint matrix
 * (Y_max_matrix).
 * @tparam Out_Type           Type of the output penalty accumulator
 * (Y_limit_penalty).
 * @tparam M                  Number of rows in the matrices.
 * @tparam N                  Number of columns in the matrices.
 *
 * @param Y_horizon           Reference to the main matrix containing horizon
 * values.
 * @param Y_min_matrix        Reference to the matrix containing minimum
 * constraints.
 * @param Y_max_matrix        Reference to the matrix containing maximum
 * constraints.
 * @param Y_limit_penalty     Reference to the output variable accumulating
 * penalty values.
 */
template <typename Y_Mat_Type, typename Y_Min_Matrix_Type,
          typename Y_Max_Matrix_Type, typename Out_Type, std::size_t M,
          std::size_t N>
struct Row<Y_Mat_Type, Y_Min_Matrix_Type, Y_Max_Matrix_Type, Out_Type, M, N,
           0> {
  static void compute(const Y_Mat_Type &Y_horizon,
                      const Y_Min_Matrix_Type &Y_min_matrix,
                      const Y_Max_Matrix_Type &Y_max_matrix,
                      Out_Type &Y_limit_penalty) {
    Column<Y_Mat_Type, Y_Min_Matrix_Type, Y_Max_Matrix_Type, Out_Type, M, N, 0,
           (N - 1)>::compute(Y_horizon, Y_min_matrix, Y_max_matrix,
                             Y_limit_penalty);
  }
};

} // namespace CalculateY_LimitPenalty

/**
 * @brief Calculates the penalty for violating Y matrix limits over a prediction
 * horizon.
 *
 * This function computes the penalty associated with the elements of the
 * Y_horizon matrix that exceed the specified minimum (Y_min_matrix) and maximum
 * (Y_max_matrix) bounds. The result is stored in the Y_limit_penalty output
 * matrix.
 *
 * @tparam Y_Mat_Type Type of the input Y_horizon matrix.
 * @tparam Y_Min_Matrix_Type Type of the minimum bounds matrix.
 * @tparam Y_Max_Matrix_Type Type of the maximum bounds matrix.
 * @tparam Out_Type Type of the output penalty matrix.
 *
 * @param Y_horizon The matrix representing the predicted values over the
 * horizon.
 * @param Y_min_matrix The matrix specifying the minimum allowed values for
 * Y_horizon.
 * @param Y_max_matrix The matrix specifying the maximum allowed values for
 * Y_horizon.
 * @param Y_limit_penalty Output matrix where the computed penalties are stored.
 *
 * @note The dimensions of Out_Type (COLS and ROWS) must be positive.
 * @note This function uses a recursive template implementation for computation.
 */
template <typename Y_Mat_Type, typename Y_Min_Matrix_Type,
          typename Y_Max_Matrix_Type, typename Out_Type>
inline void calculate_Y_limit_penalty(const Y_Mat_Type &Y_horizon,
                                      const Y_Min_Matrix_Type &Y_min_matrix,
                                      const Y_Max_Matrix_Type &Y_max_matrix,
                                      Out_Type &Y_limit_penalty) {

  constexpr std::size_t M = Out_Type::COLS;
  constexpr std::size_t N = Out_Type::ROWS;
  static_assert(M > 0 && N > 0, "Matrix dimensions must be positive");

  CalculateY_LimitPenalty::Row<Y_Mat_Type, Y_Min_Matrix_Type, Y_Max_Matrix_Type,
                               Out_Type, M, N,
                               (M - 1)>::compute(Y_horizon, Y_min_matrix,
                                                 Y_max_matrix, Y_limit_penalty);
}

/* calculate Y_limit penalty and active */

namespace CalculateY_LimitPenaltyAndActive {

// Per-element conditional for Y_min with penalty+active update
/**
 * @brief Template struct for conditional minimum matrix operation in SQP
 * optimization.
 *
 * This struct serves as a template for performing conditional minimum
 * operations on matrices within the context of Sequential Quadratic Programming
 * (SQP) optimization.
 *
 * @tparam Y_Mat_Type         The type representing the main matrix involved in
 * the operation.
 * @tparam Y_Min_Matrix_Type  The type representing the matrix used for minimum
 * comparison.
 * @tparam Out_Penalty_Type   The type used for output penalties or constraint
 * violations.
 * @tparam Active_Type        The type indicating active constraints or
 * elements.
 * @tparam I                  The number of rows or a specific dimension index.
 * @tparam J_idx              The number of columns or a specific dimension
 * index.
 * @tparam limit_valid_flag   Boolean flag indicating whether to enforce limit
 * validation.
 */
template <typename Y_Mat_Type, typename Y_Min_Matrix_Type,
          typename Out_Penalty_Type, typename Active_Type, std::size_t I,
          std::size_t J_idx, bool limit_valid_flag>
struct MinConditional {};

/**
 * @brief Specialization of MinConditional for conditional minimum operation.
 *
 * This struct provides a static compute function that checks if the value at
 * position (I, J_idx) in the input matrix Y_horizon is less than the
 * corresponding minimum value in Y_min_matrix. If the condition is met, it
 * updates the Y_limit_penalty and Y_limit_active matrices at the same position.
 *
 * @tparam Y_Mat_Type         Type of the input matrix Y_horizon.
 * @tparam Y_Min_Matrix_Type  Type of the minimum matrix Y_min_matrix.
 * @tparam Out_Penalty_Type   Type of the output penalty matrix Y_limit_penalty.
 * @tparam Active_Type        Type of the output active matrix Y_limit_active.
 * @tparam I                  Row index (compile-time constant).
 * @tparam J_idx              Column index (compile-time constant).
 *
 * @note This specialization is enabled when the last template parameter is
 * true.
 *
 * @param Y_horizon       Input matrix containing current values.
 * @param Y_min_matrix    Matrix containing minimum allowed values.
 * @param Y_limit_penalty Output matrix to store penalty values when the
 * condition is met.
 * @param Y_limit_active  Output matrix to indicate active constraints (set to 1
 * if active).
 */
template <typename Y_Mat_Type, typename Y_Min_Matrix_Type,
          typename Out_Penalty_Type, typename Active_Type, std::size_t I,
          std::size_t J_idx>
struct MinConditional<Y_Mat_Type, Y_Min_Matrix_Type, Out_Penalty_Type,
                      Active_Type, I, J_idx, true> {
  static inline void compute(const Y_Mat_Type &Y_horizon,
                             const Y_Min_Matrix_Type &Y_min_matrix,
                             Out_Penalty_Type &Y_limit_penalty,
                             Active_Type &Y_limit_active) {

    const auto y = Y_horizon.template get<I, J_idx>();
    const auto y_min = Y_min_matrix.template get<I, J_idx>();

    if (y < y_min) {
      Y_limit_penalty.template set<I, J_idx>(y - y_min);
      Y_limit_active.template set<I, J_idx>(
          static_cast<typename Active_Type::Value_Type>(1));
    }
  }
};

/**
 * @brief Specialization of MinConditional struct for the case when the
 * condition is false.
 *
 * This specialization provides a no-op implementation of the compute function.
 * All input parameters are marked as unused to avoid compiler warnings.
 *
 * @tparam Y_Mat_Type         Type of the input matrix Y_horizon.
 * @tparam Y_Min_Matrix_Type  Type of the minimum matrix Y_min_matrix.
 * @tparam Out_Penalty_Type   Type for the output penalty Y_limit_penalty.
 * @tparam Active_Type        Type for the active flag Y_limit_active.
 * @tparam I                  Compile-time index parameter.
 * @tparam J_idx              Compile-time index parameter.
 *
 * The compute function does nothing in this specialization.
 */
template <typename Y_Mat_Type, typename Y_Min_Matrix_Type,
          typename Out_Penalty_Type, typename Active_Type, std::size_t I,
          std::size_t J_idx>
struct MinConditional<Y_Mat_Type, Y_Min_Matrix_Type, Out_Penalty_Type,
                      Active_Type, I, J_idx, false> {
  static inline void compute(const Y_Mat_Type &Y_horizon,
                             const Y_Min_Matrix_Type &Y_min_matrix,
                             Out_Penalty_Type &Y_limit_penalty,
                             Active_Type &Y_limit_active) {
    static_cast<void>(Y_horizon);
    static_cast<void>(Y_min_matrix);
    static_cast<void>(Y_limit_penalty);
    static_cast<void>(Y_limit_active);
    /* Do Nothing */
  }
};

// Per-element conditional for Y_max with penalty+active update
/**
 * @brief Template struct for conditional maximum matrix operations with penalty
 * and activity tracking.
 *
 * This struct is designed to perform conditional maximum operations on
 * matrices, supporting various matrix types, penalty outputs, and activity
 * flags. The behavior can be customized via template parameters, including
 * matrix types, penalty types, activity types, matrix dimensions, and a flag to
 * indicate if limits are valid.
 *
 * @tparam Y_Mat_Type         Type representing the input matrix.
 * @tparam Y_Max_Matrix_Type  Type representing the matrix used for maximum
 * value computation.
 * @tparam Out_Penalty_Type   Type representing the output penalty.
 * @tparam Active_Type        Type representing the activity flag or mask.
 * @tparam I                  Row dimension or index.
 * @tparam J_idx              Column dimension or index.
 * @tparam limit_valid_flag   Boolean flag indicating if limit checking is
 * enabled.
 */
template <typename Y_Mat_Type, typename Y_Max_Matrix_Type,
          typename Out_Penalty_Type, typename Active_Type, std::size_t I,
          std::size_t J_idx, bool limit_valid_flag>
struct MaxConditional {};

template <typename Y_Mat_Type, typename Y_Max_Matrix_Type,
          typename Out_Penalty_Type, typename Active_Type, std::size_t I,
          std::size_t J_idx>
struct MaxConditional<Y_Mat_Type, Y_Max_Matrix_Type, Out_Penalty_Type,
                      Active_Type, I, J_idx, true> {
  static inline void compute(const Y_Mat_Type &Y_horizon,
                             const Y_Max_Matrix_Type &Y_max_matrix,
                             Out_Penalty_Type &Y_limit_penalty,
                             Active_Type &Y_limit_active) {

    const auto y = Y_horizon.template get<I, J_idx>();
    const auto y_max = Y_max_matrix.template get<I, J_idx>();

    if (y > y_max) {
      Y_limit_penalty.template set<I, J_idx>(y - y_max);
      Y_limit_active.template set<I, J_idx>(
          static_cast<typename Active_Type::Value_Type>(1));
    }
  }
};

/**
 * @brief Specialization of MaxConditional struct for the case when the
 * condition is false.
 *
 * This specialization provides a no-op implementation of the compute function.
 * All input parameters are marked as unused to avoid compiler warnings.
 * No operations are performed in this specialization.
 *
 * @tparam Y_Mat_Type         Type of the input matrix Y_horizon.
 * @tparam Y_Max_Matrix_Type  Type of the input matrix Y_max_matrix.
 * @tparam Out_Penalty_Type   Type of the output penalty variable.
 * @tparam Active_Type        Type of the output active variable.
 * @tparam I                  Compile-time index parameter.
 * @tparam J_idx              Compile-time index parameter.
 *
 * @note This specialization is selected when the last template parameter is
 * false.
 */
template <typename Y_Mat_Type, typename Y_Max_Matrix_Type,
          typename Out_Penalty_Type, typename Active_Type, std::size_t I,
          std::size_t J_idx>
struct MaxConditional<Y_Mat_Type, Y_Max_Matrix_Type, Out_Penalty_Type,
                      Active_Type, I, J_idx, false> {

  static inline void compute(const Y_Mat_Type &Y_horizon,
                             const Y_Max_Matrix_Type &Y_max_matrix,
                             Out_Penalty_Type &Y_limit_penalty,
                             Active_Type &Y_limit_active) {
    static_cast<void>(Y_horizon);
    static_cast<void>(Y_max_matrix);
    static_cast<void>(Y_limit_penalty);
    static_cast<void>(Y_limit_active);
    /* Do Nothing */
  }
};

// Column recursion when J_idx > 0
/**
 * @brief Recursively computes column-wise penalty and activity for matrix
 * constraints.
 *
 * This struct template processes a single column (indexed by J_idx) of the
 * input matrices, applying minimum and maximum conditional checks for each
 * element at position (I, J_idx). It updates the output penalty and activity
 * matrices accordingly. The recursion proceeds by decrementing the column index
 * (J_idx) until the base case is reached (not shown here).
 *
 * @tparam Y_Mat_Type         Type of the main input matrix (Y_horizon).
 * @tparam Y_Min_Matrix_Type  Type of the minimum constraint matrix.
 * @tparam Y_Max_Matrix_Type  Type of the maximum constraint matrix.
 * @tparam Out_Penalty_Type   Type of the output penalty matrix.
 * @tparam Active_Type        Type of the output activity matrix.
 * @tparam M                  Number of rows in the matrices.
 * @tparam N                  Number of columns in the matrices.
 * @tparam I                  Current row index being processed.
 * @tparam J_idx              Current column index being processed.
 *
 * @note This struct is intended for use in template metaprogramming and
 * recursion. The base case specialization for J_idx == 0 should be defined
 * elsewhere.
 *
 * @param Y_horizon       The main input matrix.
 * @param Y_min_matrix    The matrix containing minimum constraints.
 * @param Y_max_matrix    The matrix containing maximum constraints.
 * @param Y_limit_penalty Output matrix to accumulate penalty values.
 * @param Y_limit_active  Output matrix to accumulate activity flags.
 */
template <typename Y_Mat_Type, typename Y_Min_Matrix_Type,
          typename Y_Max_Matrix_Type, typename Out_Penalty_Type,
          typename Active_Type, std::size_t M, std::size_t N, std::size_t I,
          std::size_t J_idx>
struct Column {
  static void compute(const Y_Mat_Type &Y_horizon,
                      const Y_Min_Matrix_Type &Y_min_matrix,
                      const Y_Max_Matrix_Type &Y_max_matrix,
                      Out_Penalty_Type &Y_limit_penalty,
                      Active_Type &Y_limit_active) {

    MinConditional<Y_Mat_Type, Y_Min_Matrix_Type, Out_Penalty_Type, Active_Type,
                   I, J_idx,
                   Y_Min_Matrix_Type::SparseAvailable_Type::lists[I][J_idx]>::
        compute(Y_horizon, Y_min_matrix, Y_limit_penalty, Y_limit_active);

    MaxConditional<Y_Mat_Type, Y_Max_Matrix_Type, Out_Penalty_Type, Active_Type,
                   I, J_idx,
                   Y_Max_Matrix_Type::SparseAvailable_Type::lists[I][J_idx]>::
        compute(Y_horizon, Y_max_matrix, Y_limit_penalty, Y_limit_active);

    Column<Y_Mat_Type, Y_Min_Matrix_Type, Y_Max_Matrix_Type, Out_Penalty_Type,
           Active_Type, M, N, I, (J_idx - 1)>::compute(Y_horizon, Y_min_matrix,
                                                       Y_max_matrix,
                                                       Y_limit_penalty,
                                                       Y_limit_active);
  }
};

// Column recursion termination for J_idx == 0
/**
 * @brief Specialization of the Column struct for the case where the column
 * index J is 0.
 *
 * This struct provides a static compute function that applies minimum and
 * maximum conditional operations for the I-th row and 0-th column of the input
 * matrices. It invokes the MinConditional and MaxConditional functors to update
 * the penalty and active status based on the provided horizon, minimum, and
 * maximum matrices.
 *
 * @tparam Y_Mat_Type         Type of the input horizon matrix.
 * @tparam Y_Min_Matrix_Type  Type of the minimum constraint matrix.
 * @tparam Y_Max_Matrix_Type  Type of the maximum constraint matrix.
 * @tparam Out_Penalty_Type   Type used to accumulate penalty values.
 * @tparam Active_Type        Type used to track active constraints.
 * @tparam M                  Number of rows in the matrices.
 * @tparam N                  Number of columns in the matrices.
 * @tparam I                  Row index for which the operation is performed.
 *
 * @note This specialization is for the base case where the column index J is 0.
 *
 * @param Y_horizon        The input horizon matrix.
 * @param Y_min_matrix     The matrix containing minimum constraints.
 * @param Y_max_matrix     The matrix containing maximum constraints.
 * @param Y_limit_penalty  Output parameter to accumulate penalty values.
 * @param Y_limit_active   Output parameter to track active constraints.
 */
template <typename Y_Mat_Type, typename Y_Min_Matrix_Type,
          typename Y_Max_Matrix_Type, typename Out_Penalty_Type,
          typename Active_Type, std::size_t M, std::size_t N, std::size_t I>
struct Column<Y_Mat_Type, Y_Min_Matrix_Type, Y_Max_Matrix_Type,
              Out_Penalty_Type, Active_Type, M, N, I, 0> {
  static void compute(const Y_Mat_Type &Y_horizon,
                      const Y_Min_Matrix_Type &Y_min_matrix,
                      const Y_Max_Matrix_Type &Y_max_matrix,
                      Out_Penalty_Type &Y_limit_penalty,
                      Active_Type &Y_limit_active) {

    MinConditional<Y_Mat_Type, Y_Min_Matrix_Type, Out_Penalty_Type, Active_Type,
                   I, 0, Y_Min_Matrix_Type::SparseAvailable_Type::lists[I][0]>::
        compute(Y_horizon, Y_min_matrix, Y_limit_penalty, Y_limit_active);

    MaxConditional<Y_Mat_Type, Y_Max_Matrix_Type, Out_Penalty_Type, Active_Type,
                   I, 0, Y_Max_Matrix_Type::SparseAvailable_Type::lists[I][0]>::
        compute(Y_horizon, Y_max_matrix, Y_limit_penalty, Y_limit_active);
  }
};

// Row recursion when I_idx > 0
/**
 * @brief Recursively computes penalties and active flags for each row in a
 * matrix.
 *
 * This struct template processes a specific row (indexed by I_idx) of the input
 * matrices, applying the compute operation for each column in the row using the
 * Column struct, and then recursively processes the previous row until the base
 * case is reached.
 *
 * @tparam Y_Mat_Type         Type of the input matrix containing horizon
 * values.
 * @tparam Y_Min_Matrix_Type  Type of the matrix containing minimum limits.
 * @tparam Y_Max_Matrix_Type  Type of the matrix containing maximum limits.
 * @tparam Out_Penalty_Type   Type of the output penalty accumulator.
 * @tparam Active_Type        Type of the output active flag accumulator.
 * @tparam M                  Number of rows in the matrices.
 * @tparam N                  Number of columns in the matrices.
 * @tparam I_idx              Current row index being processed (recursively
 * decremented).
 *
 * @param Y_horizon       The input matrix of horizon values.
 * @param Y_min_matrix    The matrix of minimum allowed values.
 * @param Y_max_matrix    The matrix of maximum allowed values.
 * @param Y_limit_penalty Output accumulator for penalty values.
 * @param Y_limit_active  Output accumulator for active flags.
 */
template <typename Y_Mat_Type, typename Y_Min_Matrix_Type,
          typename Y_Max_Matrix_Type, typename Out_Penalty_Type,
          typename Active_Type, std::size_t M, std::size_t N, std::size_t I_idx>
struct Row {
  static void compute(const Y_Mat_Type &Y_horizon,
                      const Y_Min_Matrix_Type &Y_min_matrix,
                      const Y_Max_Matrix_Type &Y_max_matrix,
                      Out_Penalty_Type &Y_limit_penalty,
                      Active_Type &Y_limit_active) {
    Column<Y_Mat_Type, Y_Min_Matrix_Type, Y_Max_Matrix_Type, Out_Penalty_Type,
           Active_Type, M, N, I_idx, (N - 1)>::compute(Y_horizon, Y_min_matrix,
                                                       Y_max_matrix,
                                                       Y_limit_penalty,
                                                       Y_limit_active);
    Row<Y_Mat_Type, Y_Min_Matrix_Type, Y_Max_Matrix_Type, Out_Penalty_Type,
        Active_Type, M, N, (I_idx - 1)>::compute(Y_horizon, Y_min_matrix,
                                                 Y_max_matrix, Y_limit_penalty,
                                                 Y_limit_active);
  }
};

// Row recursion termination for I_idx == 0
/**
 * @brief Specialization of the Row struct for the case when the row index is 0.
 *
 * This struct provides a static compute function that processes the first row
 * (row index 0) of a matrix operation in the context of sequential quadratic
 * programming (SQP) optimization. The function delegates the computation to the
 * corresponding Column specialization for the first row and all columns (from
 * column 0 to N-1).
 *
 * @tparam Y_Mat_Type         Type of the input matrix Y_horizon.
 * @tparam Y_Min_Matrix_Type  Type of the minimum constraint matrix
 * Y_min_matrix.
 * @tparam Y_Max_Matrix_Type  Type of the maximum constraint matrix
 * Y_max_matrix.
 * @tparam Out_Penalty_Type   Type for the output penalty accumulator.
 * @tparam Active_Type        Type for the active set indicator.
 * @tparam M                  Number of rows in the matrices.
 * @tparam N                  Number of columns in the matrices.
 *
 * @param Y_horizon           The input matrix representing the optimization
 * horizon.
 * @param Y_min_matrix        The matrix of minimum constraints.
 * @param Y_max_matrix        The matrix of maximum constraints.
 * @param Y_limit_penalty     Output parameter to accumulate penalty values for
 * constraint violations.
 * @param Y_limit_active      Output parameter to indicate which constraints are
 * active.
 */
template <typename Y_Mat_Type, typename Y_Min_Matrix_Type,
          typename Y_Max_Matrix_Type, typename Out_Penalty_Type,
          typename Active_Type, std::size_t M, std::size_t N>
struct Row<Y_Mat_Type, Y_Min_Matrix_Type, Y_Max_Matrix_Type, Out_Penalty_Type,
           Active_Type, M, N, 0> {
  static void compute(const Y_Mat_Type &Y_horizon,
                      const Y_Min_Matrix_Type &Y_min_matrix,
                      const Y_Max_Matrix_Type &Y_max_matrix,
                      Out_Penalty_Type &Y_limit_penalty,
                      Active_Type &Y_limit_active) {
    Column<Y_Mat_Type, Y_Min_Matrix_Type, Y_Max_Matrix_Type, Out_Penalty_Type,
           Active_Type, M, N, 0, (N - 1)>::compute(Y_horizon, Y_min_matrix,
                                                   Y_max_matrix,
                                                   Y_limit_penalty,
                                                   Y_limit_active);
  }
};

} // namespace CalculateY_LimitPenaltyAndActive

/**
 * @brief Calculates the penalty and active status for Y matrix limits over a
 * horizon.
 *
 * This function computes the penalty values and active flags for each element
 * in the Y matrix, based on provided minimum and maximum limit matrices. The
 * results are stored in the output penalty and active matrices. The function is
 * templated to support various matrix types and sizes, and performs
 * compile-time checks to ensure matrix dimension compatibility.
 *
 * @tparam Y_Mat_Type         Type of the input Y matrix over the horizon.
 * @tparam Y_Min_Matrix_Type  Type of the minimum limit matrix for Y.
 * @tparam Y_Max_Matrix_Type  Type of the maximum limit matrix for Y.
 * @tparam Out_Penalty_Type   Type of the output penalty matrix.
 * @tparam Active_Type        Type of the output active matrix.
 *
 * @param Y_horizon       Input Y matrix over the horizon.
 * @param Y_min_matrix    Minimum limit matrix for Y.
 * @param Y_max_matrix    Maximum limit matrix for Y.
 * @param Y_limit_penalty Output matrix to store penalty values for Y limits.
 * @param Y_limit_active  Output matrix to store active status flags for Y
 * limits.
 */
template <typename Y_Mat_Type, typename Y_Min_Matrix_Type,
          typename Y_Max_Matrix_Type, typename Out_Penalty_Type,
          typename Active_Type>
inline void calculate_Y_limit_penalty_and_active(
    const Y_Mat_Type &Y_horizon, const Y_Min_Matrix_Type &Y_min_matrix,
    const Y_Max_Matrix_Type &Y_max_matrix, Out_Penalty_Type &Y_limit_penalty,
    Active_Type &Y_limit_active) {

  constexpr std::size_t M = Out_Penalty_Type::COLS;
  constexpr std::size_t N = Out_Penalty_Type::ROWS;
  static_assert(M > 0 && N > 0, "Matrix dimensions must be positive");
  static_assert(Out_Penalty_Type::COLS == Active_Type::COLS,
                "Penalty and Active COLS mismatch");
  static_assert(Out_Penalty_Type::ROWS == Active_Type::ROWS,
                "Penalty and Active ROWS mismatch");

  CalculateY_LimitPenaltyAndActive::Row<
      Y_Mat_Type, Y_Min_Matrix_Type, Y_Max_Matrix_Type, Out_Penalty_Type,
      Active_Type, M, N, (M - 1)>::compute(Y_horizon, Y_min_matrix,
                                           Y_max_matrix, Y_limit_penalty,
                                           Y_limit_active);
}

/* fx xx lambda contract */

namespace FxxLambdaContract {

// K-accumulation: recursively accumulate over k (STATE_SIZE dimension)
/**
 * @brief Helper struct to recursively accumulate the product of elements from a
 * Hessian-like matrix and a vector.
 *
 * This template struct performs a recursive accumulation over the index K_idx,
 * multiplying elements from the Hf_xx matrix and the dX vector, and adding the
 * result to the accumulator acc. The recursion proceeds by decrementing K_idx
 * until the base case is reached (not shown in this snippet).
 *
 * @tparam Fxx_Type    Type of the Hessian-like matrix (must provide a template
 * get<I, J>() method).
 * @tparam dX_Type     Type of the vector (must provide a template get<I, J>()
 * method).
 * @tparam Value_Type  Type of the accumulator variable.
 * @tparam OUTPUT_SIZE Output size (not directly used in this struct).
 * @tparam STATE_SIZE  State size, used for matrix indexing.
 * @tparam I           Row index for the matrix.
 * @tparam J           Column index for the matrix.
 * @tparam K_idx       Current index for recursion and element selection.
 */
template <typename Fxx_Type, typename dX_Type, typename Value_Type,
          std::size_t OUTPUT_SIZE, std::size_t STATE_SIZE, std::size_t I,
          std::size_t J, std::size_t K_idx>
struct AccumulateK {
  static void compute(const Fxx_Type &Hf_xx, const dX_Type &dX,
                      Value_Type &acc) {
    acc += Hf_xx.template get<I * STATE_SIZE + J, K_idx>() *
           dX.template get<K_idx, 0>();
    AccumulateK<Fxx_Type, dX_Type, Value_Type, OUTPUT_SIZE, STATE_SIZE, I, J,
                (K_idx - 1)>::compute(Hf_xx, dX, acc);
  }
};

// K-accumulation termination (K_idx == 0)
/**
 * @brief Specialization of the AccumulateK struct for the case when K = 0.
 *
 * This struct provides a static compute function that performs an accumulation
 * operation for a specific element in a matrix or tensor operation, typically
 * used in optimization routines.
 *
 * @tparam Fxx_Type   Type representing the Hessian or second derivative matrix.
 * @tparam dX_Type    Type representing the delta or change in state vector.
 * @tparam Value_Type Type of the accumulator variable.
 * @tparam OUTPUT_SIZE Size of the output dimension.
 * @tparam STATE_SIZE  Size of the state dimension.
 * @tparam I           Row index for the operation.
 * @tparam J           Column index for the operation.
 *
 * @param Hf_xx  Reference to the Hessian or second derivative matrix.
 * @param dX     Reference to the delta state vector.
 * @param acc    Reference to the accumulator variable to be updated.
 *
 * The function adds to acc the product of the (I * STATE_SIZE + J, 0) element
 * of Hf_xx and the (0, 0) element of dX.
 */
template <typename Fxx_Type, typename dX_Type, typename Value_Type,
          std::size_t OUTPUT_SIZE, std::size_t STATE_SIZE, std::size_t I,
          std::size_t J>
struct AccumulateK<Fxx_Type, dX_Type, Value_Type, OUTPUT_SIZE, STATE_SIZE, I, J,
                   0> {
  static void compute(const Fxx_Type &Hf_xx, const dX_Type &dX,
                      Value_Type &acc) {
    acc +=
        Hf_xx.template get<I * STATE_SIZE + J, 0>() * dX.template get<0, 0>();
  }
};

// Column recursion over j (STATE_SIZE): computes contribution to out(j,0)
/**
 * @brief Computes and updates a single column in an output matrix as part of a
 * recursive template meta-programming operation.
 *
 * This struct template recursively computes the contribution of a specific
 * column (indexed by J_idx) for a given row (indexed by I) in the output matrix
 * `out`. The computation involves accumulating a weighted sum using the Hessian
 * matrix `Hf_xx`, a delta vector `dX`, and a weight vector `lam_next`. The
 * result is added to the current value in the output matrix at position (J_idx,
 * 0).
 *
 * @tparam Fxx_Type     Type of the Hessian matrix input.
 * @tparam dX_Type      Type of the delta vector input.
 * @tparam Weight_Type  Type of the weight vector input.
 * @tparam Out_Type     Type of the output matrix.
 * @tparam OUTPUT_SIZE  Number of rows in the output matrix.
 * @tparam STATE_SIZE   Number of columns in the state (and Hessian) matrix.
 * @tparam I            Row index for which the computation is performed.
 * @tparam J_idx        Column index being processed recursively.
 *
 * @param Hf_xx     The Hessian matrix input.
 * @param dX        The delta vector input.
 * @param lam_next  The weight vector input.
 * @param out       The output matrix to be updated.
 */
template <typename Fxx_Type, typename dX_Type, typename Weight_Type,
          typename Out_Type, std::size_t OUTPUT_SIZE, std::size_t STATE_SIZE,
          std::size_t I, std::size_t J_idx>
struct Column {
  static void compute(const Fxx_Type &Hf_xx, const dX_Type &dX,
                      const Weight_Type &lam_next, Out_Type &out) {
    using Value = typename Out_Type::Value_Type;
    Value acc = static_cast<Value>(0);
    AccumulateK<Fxx_Type, dX_Type, Value, OUTPUT_SIZE, STATE_SIZE, I, J_idx,
                (STATE_SIZE - 1)>::compute(Hf_xx, dX, acc);

    const auto w = lam_next.template get<I, 0>();
    const auto updated = out.template get<J_idx, 0>() + w * acc;
    out.template set<J_idx, 0>(updated);

    Column<Fxx_Type, dX_Type, Weight_Type, Out_Type, OUTPUT_SIZE, STATE_SIZE, I,
           (J_idx - 1)>::compute(Hf_xx, dX, lam_next, out);
  }
};

// Column recursion termination for j == 0
/**
 * @brief Specialization of the Column struct for the case where J = 0.
 *
 * This struct provides a static compute function that performs a matrix-vector
 * operation as part of a sequential quadratic programming (SQP) optimization
 * routine. It accumulates the result of multiplying a Hessian-like matrix
 * (Hf_xx) with a direction vector (dX), scales the result by a weight
 * (lam_next), and updates the output matrix (out).
 *
 * @tparam Fxx_Type     Type of the Hessian-like matrix input.
 * @tparam dX_Type      Type of the direction vector input.
 * @tparam Weight_Type  Type of the weight input (typically a Lagrange
 * multiplier).
 * @tparam Out_Type     Type of the output matrix.
 * @tparam OUTPUT_SIZE  Number of rows in the output matrix.
 * @tparam STATE_SIZE   Number of columns in the Hessian-like matrix and size of
 * dX.
 * @tparam I            Index of the current column being processed.
 *
 * The compute function:
 *   - Accumulates the dot product of the I-th column of Hf_xx and dX.
 *   - Multiplies the accumulated value by the corresponding weight from
 * lam_next.
 *   - Adds the result to the (0, 0) entry of the output matrix.
 *
 * @param Hf_xx     The Hessian-like matrix.
 * @param dX        The direction vector.
 * @param lam_next  The weight (e.g., Lagrange multiplier) for the current
 * column.
 * @param out       The output matrix to be updated.
 */
template <typename Fxx_Type, typename dX_Type, typename Weight_Type,
          typename Out_Type, std::size_t OUTPUT_SIZE, std::size_t STATE_SIZE,
          std::size_t I>
struct Column<Fxx_Type, dX_Type, Weight_Type, Out_Type, OUTPUT_SIZE, STATE_SIZE,
              I, 0> {
  static void compute(const Fxx_Type &Hf_xx, const dX_Type &dX,
                      const Weight_Type &lam_next, Out_Type &out) {
    using Value = typename Out_Type::Value_Type;
    Value acc = static_cast<Value>(0);
    AccumulateK<Fxx_Type, dX_Type, Value, OUTPUT_SIZE, STATE_SIZE, I, 0,
                (STATE_SIZE - 1)>::compute(Hf_xx, dX, acc);

    const auto w = lam_next.template get<I, 0>();
    const auto updated = out.template get<0, 0>() + w * acc;
    out.template set<0, 0>(updated);
  }
};

// Row recursion over i (OUTPUT_SIZE): iterates outputs
/**
 * @brief Recursive template struct to compute a row of a matrix operation.
 *
 * This struct template recursively computes a row of a matrix operation by
 * invoking the corresponding Column computation for the current row index
 * (I_idx) and then recursively calling itself for the previous row index (I_idx
 * - 1).
 *
 * @tparam Fxx_Type    Type of the Hessian or second derivative matrix.
 * @tparam dX_Type     Type of the delta X or state difference vector.
 * @tparam Weight_Type Type of the weight or multiplier (e.g., lambda).
 * @tparam Out_Type    Type of the output container.
 * @tparam OUTPUT_SIZE Number of rows in the output.
 * @tparam STATE_SIZE  Number of columns (state size).
 * @tparam I_idx       Current row index being processed (recursion parameter).
 *
 * @note This struct is intended to be used as part of a recursive template
 * metaprogramming pattern for efficient compile-time matrix operations,
 * typically in optimization routines.
 */
template <typename Fxx_Type, typename dX_Type, typename Weight_Type,
          typename Out_Type, std::size_t OUTPUT_SIZE, std::size_t STATE_SIZE,
          std::size_t I_idx>
struct Row {
  static void compute(const Fxx_Type &Hf_xx, const dX_Type &dX,
                      const Weight_Type &lam_next, Out_Type &out) {
    Column<Fxx_Type, dX_Type, Weight_Type, Out_Type, OUTPUT_SIZE, STATE_SIZE,
           I_idx, (STATE_SIZE - 1)>::compute(Hf_xx, dX, lam_next, out);
    Row<Fxx_Type, dX_Type, Weight_Type, Out_Type, OUTPUT_SIZE, STATE_SIZE,
        (I_idx - 1)>::compute(Hf_xx, dX, lam_next, out);
  }
};

// Row recursion termination for i == 0
/**
 * This struct provides a static compute function that delegates the computation
 * to the corresponding Column specialization for the first row (index 0) and
 * the last column (index STATE_SIZE - 1).
 *
 * @tparam Fxx_Type      Type representing the Hessian or second derivative
 * matrix.
 * @tparam dX_Type       Type representing the state difference or increment.
 * @tparam Weight_Type   Type representing the weight or multiplier (e.g.,
 * lambda).
 * @tparam Out_Type      Type representing the output container.
 * @tparam OUTPUT_SIZE   Compile-time constant for the output size.
 * @tparam STATE_SIZE    Compile-time constant for the state size.
 *
 * @param Hf_xx      Reference to the Hessian or second derivative matrix.
 * @param dX         Reference to the state difference or increment.
 * @param lam_next   Reference to the weight or multiplier for the next step.
 * @param out        Reference to the output container to store the result.
 */
template <typename Fxx_Type, typename dX_Type, typename Weight_Type,
          typename Out_Type, std::size_t OUTPUT_SIZE, std::size_t STATE_SIZE>
struct Row<Fxx_Type, dX_Type, Weight_Type, Out_Type, OUTPUT_SIZE, STATE_SIZE,
           0> {
  static void compute(const Fxx_Type &Hf_xx, const dX_Type &dX,
                      const Weight_Type &lam_next, Out_Type &out) {
    Column<Fxx_Type, dX_Type, Weight_Type, Out_Type, OUTPUT_SIZE, STATE_SIZE, 0,
           (STATE_SIZE - 1)>::compute(Hf_xx, dX, lam_next, out);
  }
};

} // namespace FxxLambdaContract

/**
 * @brief Computes the contraction of the Hessian matrix Hf_xx with the
 * direction vector dX and the weight vector lam_next, and stores the result in
 * the output vector out.
 *
 * This function performs a specialized matrix operation used in Sequential
 * Quadratic Programming (SQP) optimization routines. It contracts the Hessian
 * matrix with the direction and weight vectors, enforcing strict compile-time
 * checks on the dimensions of all input and output types to ensure correctness.
 *
 * @tparam Fxx_Type      Type representing the Hessian matrix (Hf_xx), expected
 * to have dimensions [STATE_SIZE x OUTPUT_SIZE * STATE_SIZE].
 * @tparam dX_Type       Type representing the direction vector (dX), expected
 * to have dimensions [1 x STATE_SIZE].
 * @tparam Weight_Type   Type representing the weight vector (lam_next),
 * expected to have dimensions [1 x OUTPUT_SIZE].
 * @tparam Out_Type      Type representing the output vector (out), expected to
 * have dimensions [1 x STATE_SIZE].
 *
 * @param Hf_xx          The Hessian matrix of the objective function with
 * respect to state variables.
 * @param dX             The direction vector for the optimization step.
 * @param lam_next       The weight vector (typically Lagrange multipliers for
 * constraints).
 * @param out            Output vector to store the result of the contraction.
 *
 * @note All types must satisfy the required static dimension checks. The
 * computation is delegated to FxxLambdaContract::Row.
 */
template <typename Fxx_Type, typename dX_Type, typename Weight_Type,
          typename Out_Type>
inline void
compute_fxx_lambda_contract(const Fxx_Type &Hf_xx, const dX_Type &dX,
                            const Weight_Type &lam_next, Out_Type &out) {

  static_assert(dX_Type::ROWS == 1, "dX must be a (STATE_SIZE x 1) vector");
  static_assert(Weight_Type::ROWS == 1,
                "lam_next must be a (OUTPUT_SIZE x 1) vector");
  static_assert(Out_Type::ROWS == 1,
                "out must be a (STATE_SIZE x 1) vector (ROWS == 1)");

  constexpr std::size_t STATE_SIZE = dX_Type::COLS;
  constexpr std::size_t OUTPUT_SIZE = Weight_Type::COLS;

  static_assert(STATE_SIZE > 0 && OUTPUT_SIZE > 0,
                "STATE_SIZE and OUTPUT_SIZE must be positive");
  static_assert(Fxx_Type::ROWS == STATE_SIZE,
                "Hf_xx ROWS must equal STATE_SIZE");
  static_assert(Fxx_Type::COLS == OUTPUT_SIZE * STATE_SIZE,
                "Hf_xx COLS must equal OUTPUT_SIZE * STATE_SIZE");
  static_assert(Out_Type::COLS == STATE_SIZE, "out COLS must equal STATE_SIZE");

  FxxLambdaContract::Row<Fxx_Type, dX_Type, Weight_Type, Out_Type, OUTPUT_SIZE,
                         STATE_SIZE, (OUTPUT_SIZE - 1)>::compute(Hf_xx, dX,
                                                                 lam_next, out);
}

/* fx xu lambda contract */

namespace FxuLambdaContract {

// K-accumulation: recursively accumulate over k (INPUT_SIZE dimension)
/**
 * @brief Recursive template struct to accumulate a sum over the K_idx
 * dimension.
 *
 * This struct defines a static compute function that recursively accumulates
 * the product of elements from Hf_xu and dU into the provided accumulator
 * variable `acc`. The recursion proceeds by decrementing K_idx until the base
 * case is reached (which should be specialized elsewhere).
 *
 * @tparam Fxu_Type   Type of the Hf_xu matrix-like object, expected to provide
 * a templated get<I, J>() method.
 * @tparam dU_Type    Type of the dU matrix-like object, expected to provide a
 * templated get<I, J>() method.
 * @tparam Value_Type Type of the accumulator variable.
 * @tparam OUTPUT_SIZE Size of the output dimension (not directly used here).
 * @tparam STATE_SIZE  Size of the state dimension, used for index calculation.
 * @tparam INPUT_SIZE  Size of the input dimension (not directly used here).
 * @tparam I           Row index for the outer operation.
 * @tparam J           Column index for the outer operation.
 * @tparam K_idx       Current index for the recursive accumulation.
 */
template <typename Fxu_Type, typename dU_Type, typename Value_Type,
          std::size_t OUTPUT_SIZE, std::size_t STATE_SIZE,
          std::size_t INPUT_SIZE, std::size_t I, std::size_t J,
          std::size_t K_idx>
struct AccumulateK {
  static void compute(const Fxu_Type &Hf_xu, const dU_Type &dU,
                      Value_Type &acc) {
    acc += Hf_xu.template get<I * STATE_SIZE + J, K_idx>() *
           dU.template get<K_idx, 0>();
    AccumulateK<Fxu_Type, dU_Type, Value_Type, OUTPUT_SIZE, STATE_SIZE,
                INPUT_SIZE, I, J, (K_idx - 1)>::compute(Hf_xu, dU, acc);
  }
};

// K-accumulation termination (K_idx == 0)
/**
 * @brief Specialization of the AccumulateK struct for the case when K = 0.
 *
 * This struct provides a static compute function that accumulates the product
 * of a specific element from the Hf_xu matrix and a specific element from the
 * dU matrix into the acc variable.
 *
 * @tparam Fxu_Type   Type of the Hf_xu matrix-like object.
 * @tparam dU_Type    Type of the dU matrix-like object.
 * @tparam Value_Type Type of the accumulator variable.
 * @tparam OUTPUT_SIZE Size of the output (not used in this specialization).
 * @tparam STATE_SIZE  Size of the state dimension.
 * @tparam INPUT_SIZE  Size of the input dimension (not used in this
 * specialization).
 * @tparam I           Row index for the operation.
 * @tparam J           Column index for the operation.
 *
 * @note This specialization is typically used as the base case in recursive
 * template meta-programming.
 *
 * @param Hf_xu Reference to the matrix-like object containing values to be
 * multiplied.
 * @param dU    Reference to the matrix-like object containing values to be
 * multiplied.
 * @param acc   Reference to the accumulator where the result is added.
 */
template <typename Fxu_Type, typename dU_Type, typename Value_Type,
          std::size_t OUTPUT_SIZE, std::size_t STATE_SIZE,
          std::size_t INPUT_SIZE, std::size_t I, std::size_t J>
struct AccumulateK<Fxu_Type, dU_Type, Value_Type, OUTPUT_SIZE, STATE_SIZE,
                   INPUT_SIZE, I, J, 0> {
  static void compute(const Fxu_Type &Hf_xu, const dU_Type &dU,
                      Value_Type &acc) {
    acc +=
        Hf_xu.template get<I * STATE_SIZE + J, 0>() * dU.template get<0, 0>();
  }
};

// Column recursion over j (STATE_SIZE): computes contribution to out(j,0)
/**
 * @brief Computes and updates a column in an output matrix as part of a matrix
 * operation, typically used in Sequential Quadratic Programming (SQP)
 * optimization routines.
 *
 * This struct template recursively processes columns of the output matrix by
 * accumulating weighted contributions from the provided matrices and vectors.
 * For each column index J_idx, it computes an accumulated value using the
 * AccumulateK helper, multiplies it by a weight from lam_next, and updates the
 * corresponding entry in the output matrix 'out'.
 *
 * @tparam Fxu_Type     Type of the matrix Hf_xu (e.g., Hessian or Jacobian).
 * @tparam dU_Type      Type of the vector dU (e.g., control input increments).
 * @tparam Weight_Type  Type of the weight matrix/vector lam_next.
 * @tparam Out_Type     Type of the output matrix to be updated.
 * @tparam OUTPUT_SIZE  Number of rows in the output matrix.
 * @tparam STATE_SIZE   Number of state variables.
 * @tparam INPUT_SIZE   Number of input variables.
 * @tparam I            Current row index being processed.
 * @tparam J_idx        Current column index being processed (recursively
 * decremented).
 *
 * @note This struct is intended for use in template metaprogramming and
 * recursion, and is typically specialized for the base case where J_idx == 0.
 *
 * @param Hf_xu     The input matrix containing partial derivatives or Hessian
 * values.
 * @param dU        The input vector representing increments or changes in input
 * variables.
 * @param lam_next  The weight vector/matrix for the next step in the
 * optimization.
 * @param out       The output matrix to be updated with the computed values.
 */
template <typename Fxu_Type, typename dU_Type, typename Weight_Type,
          typename Out_Type, std::size_t OUTPUT_SIZE, std::size_t STATE_SIZE,
          std::size_t INPUT_SIZE, std::size_t I, std::size_t J_idx>
struct Column {
  static void compute(const Fxu_Type &Hf_xu, const dU_Type &dU,
                      const Weight_Type &lam_next, Out_Type &out) {
    using Value = typename Out_Type::Value_Type;
    Value acc = static_cast<Value>(0);
    AccumulateK<Fxu_Type, dU_Type, Value, OUTPUT_SIZE, STATE_SIZE, INPUT_SIZE,
                I, J_idx, (INPUT_SIZE - 1)>::compute(Hf_xu, dU, acc);

    const auto w = lam_next.template get<I, 0>();
    const auto updated = out.template get<J_idx, 0>() + w * acc;
    out.template set<J_idx, 0>(updated);

    Column<Fxu_Type, dU_Type, Weight_Type, Out_Type, OUTPUT_SIZE, STATE_SIZE,
           INPUT_SIZE, I, (J_idx - 1)>::compute(Hf_xu, dU, lam_next, out);
  }
};

// Column recursion termination for j == 0

/**
 * @brief Specialization of the Column struct for matrix operations with I
 * columns and 0 rows.
 *
 * This struct provides a static compute function that performs a matrix
 * operation involving the Hessian-vector product, an input vector, and a
 * weighting factor. The result is accumulated and used to update the output
 * matrix at position (0, 0).
 *
 * @tparam Fxu_Type    Type representing the Hessian or Jacobian matrix.
 * @tparam dU_Type     Type representing the input vector or matrix.
 * @tparam Weight_Type Type representing the weighting factor (e.g., Lagrange
 * multipliers).
 * @tparam Out_Type    Type representing the output matrix.
 * @tparam OUTPUT_SIZE Number of output rows.
 * @tparam STATE_SIZE  Number of state variables.
 * @tparam INPUT_SIZE  Number of input variables.
 * @tparam I           Current column index (compile-time constant).
 *
 * @note This specialization is for the case where the row index is 0.
 */
template <typename Fxu_Type, typename dU_Type, typename Weight_Type,
          typename Out_Type, std::size_t OUTPUT_SIZE, std::size_t STATE_SIZE,
          std::size_t INPUT_SIZE, std::size_t I>
struct Column<Fxu_Type, dU_Type, Weight_Type, Out_Type, OUTPUT_SIZE, STATE_SIZE,
              INPUT_SIZE, I, 0> {
  static void compute(const Fxu_Type &Hf_xu, const dU_Type &dU,
                      const Weight_Type &lam_next, Out_Type &out) {
    using Value = typename Out_Type::Value_Type;
    Value acc = static_cast<Value>(0);
    AccumulateK<Fxu_Type, dU_Type, Value, OUTPUT_SIZE, STATE_SIZE, INPUT_SIZE,
                I, 0, (INPUT_SIZE - 1)>::compute(Hf_xu, dU, acc);

    const auto w = lam_next.template get<I, 0>();
    const auto updated = out.template get<0, 0>() + w * acc;
    out.template set<0, 0>(updated);
  }
};

// Row recursion over i (OUTPUT_SIZE): iterates outputs
/**
 * @brief Recursive template struct to compute a row-wise operation for SQP
 * matrix operations.
 *
 * This struct template recursively processes each row of a matrix operation by
 * invoking the corresponding Column computation for the current row index
 * (I_idx), and then recursively calling itself for the previous row index
 * (I_idx - 1).
 *
 * @tparam Fxu_Type     Type representing the Hessian or Jacobian matrix
 * (Hf_xu).
 * @tparam dU_Type      Type representing the control increment vector (dU).
 * @tparam Weight_Type  Type representing the weight or multiplier (lam_next).
 * @tparam Out_Type     Type representing the output container (out).
 * @tparam OUTPUT_SIZE  Size of the output vector or matrix.
 * @tparam STATE_SIZE   Size of the state vector.
 * @tparam INPUT_SIZE   Size of the input vector.
 * @tparam I_idx        Current row index being processed (compile-time
 * constant).
 *
 * @note This struct is intended to be used in conjunction with a specialized
 * base case to terminate the recursion when I_idx reaches zero or a predefined
 * limit.
 */
template <typename Fxu_Type, typename dU_Type, typename Weight_Type,
          typename Out_Type, std::size_t OUTPUT_SIZE, std::size_t STATE_SIZE,
          std::size_t INPUT_SIZE, std::size_t I_idx>
struct Row {
  static void compute(const Fxu_Type &Hf_xu, const dU_Type &dU,
                      const Weight_Type &lam_next, Out_Type &out) {
    Column<Fxu_Type, dU_Type, Weight_Type, Out_Type, OUTPUT_SIZE, STATE_SIZE,
           INPUT_SIZE, I_idx, (STATE_SIZE - 1)>::compute(Hf_xu, dU, lam_next,
                                                         out);
    Row<Fxu_Type, dU_Type, Weight_Type, Out_Type, OUTPUT_SIZE, STATE_SIZE,
        INPUT_SIZE, (I_idx - 1)>::compute(Hf_xu, dU, lam_next, out);
  }
};

// Row recursion termination for i == 0
/**
 * @brief Specialization of the Row struct for the case where the row index is
 * 0.
 *
 * This struct provides a static compute function that delegates the computation
 * to the corresponding Column specialization for the first row (index 0) and
 * all columns from 0 to STATE_SIZE - 1.
 *
 * @tparam Fxu_Type     Type representing the Hessian or Jacobian matrix block.
 * @tparam dU_Type      Type representing the control increment vector.
 * @tparam Weight_Type  Type representing the weight or multiplier vector.
 * @tparam Out_Type     Type representing the output matrix or vector.
 * @tparam OUTPUT_SIZE  Number of outputs.
 * @tparam STATE_SIZE   Number of states.
 * @tparam INPUT_SIZE   Number of inputs.
 *
 * @param Hf_xu     The Hessian or Jacobian matrix block.
 * @param dU        The control increment vector.
 * @param lam_next  The weight or multiplier vector for the next stage.
 * @param out       The output matrix or vector to store the result.
 */
template <typename Fxu_Type, typename dU_Type, typename Weight_Type,
          typename Out_Type, std::size_t OUTPUT_SIZE, std::size_t STATE_SIZE,
          std::size_t INPUT_SIZE>
struct Row<Fxu_Type, dU_Type, Weight_Type, Out_Type, OUTPUT_SIZE, STATE_SIZE,
           INPUT_SIZE, 0> {
  static void compute(const Fxu_Type &Hf_xu, const dU_Type &dU,
                      const Weight_Type &lam_next, Out_Type &out) {
    Column<Fxu_Type, dU_Type, Weight_Type, Out_Type, OUTPUT_SIZE, STATE_SIZE,
           INPUT_SIZE, 0, (STATE_SIZE - 1)>::compute(Hf_xu, dU, lam_next, out);
  }
};

/**
 * @brief Template struct for conditional operations in SQP matrix optimization.
 *
 * This struct serves as a template for conditional logic based on the template
 * parameters. It is likely specialized for different values of the template
 * arguments to implement specific behaviors in the context of Sequential
 * Quadratic Programming (SQP) matrix operations.
 *
 * @tparam Fxu_Type      Type representing the function or system dynamics.
 * @tparam dU_Type       Type representing the control input increment.
 * @tparam Weight_Type   Type representing the weighting matrix or scalar.
 * @tparam Out_Type      Type representing the output.
 * @tparam OUTPUT_SIZE   Size of the output vector or matrix.
 * @tparam STATE_SIZE    Size of the state vector.
 * @tparam INPUT_SIZE    Size of the input vector.
 * @tparam I_idx         Index parameter, possibly for iteration or selection.
 * @tparam Activate      Boolean flag to activate or deactivate certain
 * behaviors.
 */
template <typename Fxu_Type, typename dU_Type, typename Weight_Type,
          typename Out_Type, std::size_t OUTPUT_SIZE, std::size_t STATE_SIZE,
          std::size_t INPUT_SIZE, std::size_t I_idx, bool Activate>
struct Conditional {};

/**
 * @brief Specialization of the Conditional struct for the case when the boolean
 * template parameter is true.
 *
 * This struct provides a static compute function that performs a recursive
 * computation by calling the compute function of the Row struct with
 * decremented OUTPUT_SIZE.
 *
 * @tparam Fxu_Type      Type representing the Hessian or Jacobian matrix.
 * @tparam dU_Type       Type representing the control input increment.
 * @tparam Weight_Type   Type representing the weight or multiplier (e.g.,
 * Lagrange multiplier).
 * @tparam Out_Type      Type representing the output container.
 * @tparam OUTPUT_SIZE   Size of the output vector or matrix.
 * @tparam STATE_SIZE    Size of the state vector.
 * @tparam INPUT_SIZE    Size of the input vector.
 * @tparam I_idx         Index parameter for recursion or selection.
 *
 * @param Hf_xu      The Hessian or Jacobian matrix input.
 * @param dU         The control input increment.
 * @param lam_next   The next weight or multiplier value.
 * @param out        Output container to store the result.
 */
template <typename Fxu_Type, typename dU_Type, typename Weight_Type,
          typename Out_Type, std::size_t OUTPUT_SIZE, std::size_t STATE_SIZE,
          std::size_t INPUT_SIZE, std::size_t I_idx>
struct Conditional<Fxu_Type, dU_Type, Weight_Type, Out_Type, OUTPUT_SIZE,
                   STATE_SIZE, INPUT_SIZE, I_idx, true> {
  static void compute(const Fxu_Type &Hf_xu, const dU_Type &dU,
                      const Weight_Type &lam_next, Out_Type &out) {

    Row<Fxu_Type, dU_Type, Weight_Type, Out_Type, OUTPUT_SIZE, STATE_SIZE,
        INPUT_SIZE, (OUTPUT_SIZE - 1)>::compute(Hf_xu, dU, lam_next, out);
  }
};

/**
 * @brief Specialization of the Conditional struct for the case when the
 * condition is false.
 *
 * This specialization provides a static compute function that takes references
 * to several input parameters but performs no operations on them. All
 * parameters are explicitly marked as unused to avoid compiler warnings.
 *
 * @tparam Fxu_Type      Type of the Hf_xu parameter.
 * @tparam dU_Type       Type of the dU parameter.
 * @tparam Weight_Type   Type of the lam_next parameter.
 * @tparam Out_Type      Type of the out parameter.
 * @tparam OUTPUT_SIZE   Output size as a compile-time constant.
 * @tparam STATE_SIZE    State size as a compile-time constant.
 * @tparam INPUT_SIZE    Input size as a compile-time constant.
 * @tparam I_idx         Index as a compile-time constant.
 *
 * @note This specialization is intended to be a no-op; the compute function
 * does nothing.
 *
 * @param Hf_xu     Unused input parameter.
 * @param dU        Unused input parameter.
 * @param lam_next  Unused input parameter.
 * @param out       Unused output parameter.
 */
template <typename Fxu_Type, typename dU_Type, typename Weight_Type,
          typename Out_Type, std::size_t OUTPUT_SIZE, std::size_t STATE_SIZE,
          std::size_t INPUT_SIZE, std::size_t I_idx>
struct Conditional<Fxu_Type, dU_Type, Weight_Type, Out_Type, OUTPUT_SIZE,
                   STATE_SIZE, INPUT_SIZE, I_idx, false> {
  static void compute(const Fxu_Type &Hf_xu, const dU_Type &dU,
                      const Weight_Type &lam_next, Out_Type &out) {

    static_cast<void>(Hf_xu);
    static_cast<void>(dU);
    static_cast<void>(lam_next);
    static_cast<void>(out);
    /* Do Nothing */
  }
};

} // namespace FxuLambdaContract

/**
 * @brief Computes the contraction of the Hessian matrix Hf_xu with the input
 * vector dU and the weight vector lam_next, storing the result in the output
 * vector out.
 *
 * This function performs a matrix operation commonly used in Sequential
 * Quadratic Programming (SQP) optimization routines. It contracts the Hessian
 * matrix with the input and weight vectors, producing an output vector
 * representing the result. The function enforces compile-time checks to ensure
 * that the input types have the correct dimensions:
 *   - dU must be a (INPUT_SIZE x 1) vector.
 *   - lam_next must be a (OUTPUT_SIZE x 1) vector.
 *   - out must be a (STATE_SIZE x 1) vector.
 *   - Hf_xu must have dimensions compatible with the contraction operation.
 *
 * @tparam Fxu_Type      Type representing the Hessian matrix (Hf_xu).
 * @tparam dU_Type       Type representing the input vector (dU).
 * @tparam Weight_Type   Type representing the weight vector (lam_next).
 * @tparam Out_Type      Type representing the output vector (out).
 *
 * @param Hf_xu      The Hessian matrix to be contracted.
 * @param dU         The input vector.
 * @param lam_next   The weight vector.
 * @param out        The output vector where the result is stored.
 *
 * @note This function relies on FxuLambdaContract::Conditional for the actual
 * computation.
 * @note All type and dimension checks are performed at compile time using
 * static_assert.
 */
template <typename Fxu_Type, typename dU_Type, typename Weight_Type,
          typename Out_Type>
inline void
compute_fx_xu_lambda_contract(const Fxu_Type &Hf_xu, const dU_Type &dU,
                              const Weight_Type &lam_next, Out_Type &out) {

  static_assert(dU_Type::ROWS == 1, "dU must be a (INPUT_SIZE x 1) vector");
  static_assert(Weight_Type::ROWS == 1,
                "lam_next must be a (OUTPUT_SIZE x 1) vector");
  static_assert(Out_Type::ROWS == 1,
                "out must be a (STATE_SIZE x 1) vector (ROWS == 1)");

  constexpr std::size_t STATE_SIZE = Out_Type::COLS;
  constexpr std::size_t OUTPUT_SIZE = Weight_Type::COLS;
  constexpr std::size_t INPUT_SIZE = dU_Type::COLS;

  static_assert(STATE_SIZE > 0 && INPUT_SIZE > 0,
                "STATE_SIZE, OUTPUT_SIZE and INPUT_SIZE must be positive");
  static_assert(Fxu_Type::COLS == STATE_SIZE * STATE_SIZE,
                "Hf_xu ROWS must equal OUTPUT_SIZE * STATE_SIZE");
  static_assert(Fxu_Type::ROWS == INPUT_SIZE,
                "Hf_xu COLS must equal INPUT_SIZE");
  static_assert(Out_Type::COLS == STATE_SIZE, "out COLS must equal STATE_SIZE");

  FxuLambdaContract::Conditional<
      Fxu_Type, dU_Type, Weight_Type, Out_Type, OUTPUT_SIZE, STATE_SIZE,
      INPUT_SIZE, (OUTPUT_SIZE - 1), (INPUT_SIZE > 0)>::compute(Hf_xu, dU,
                                                                lam_next, out);
}

/* fu xx lambda contract */

namespace FuxxLambdaContract {

// J-accumulation: recursively accumulate over j (STATE_SIZE dimension)
/**
 * @brief Recursively accumulates the product of elements from Hf_ux and dX into
 * acc.
 *
 * This template struct defines a static compute function that performs a
 * recursive accumulation of the product of a specific element from the Hf_ux
 * matrix and a corresponding element from the dX vector into the acc variable.
 * The recursion is controlled by the template parameter J_idx, which decrements
 * with each recursive call.
 *
 * @tparam Fuxx_Type   Type of the Hf_ux matrix-like object, must support
 * template get<row, col>().
 * @tparam dX_Type     Type of the dX vector-like object, must support template
 * get<row, col>().
 * @tparam Value_Type  Type of the accumulator variable.
 * @tparam STATE_SIZE  Size of the state vector (not directly used in this
 * struct).
 * @tparam INPUT_SIZE  Size of the input vector (used for indexing).
 * @tparam I           Row index for Hf_ux.
 * @tparam K           Column offset for Hf_ux.
 * @tparam J_idx       Current index for recursion and element selection.
 *
 * @param Hf_ux  Matrix-like object providing get<I * INPUT_SIZE + K, J_idx>().
 * @param dX     Vector-like object providing get<J_idx, 0>().
 * @param acc    Accumulator variable to which the product is added.
 */
template <typename Fuxx_Type, typename dX_Type, typename Value_Type,
          std::size_t STATE_SIZE, std::size_t INPUT_SIZE, std::size_t I,
          std::size_t K, std::size_t J_idx>
struct AccumulateJ {
  static void compute(const Fuxx_Type &Hf_ux, const dX_Type &dX,
                      Value_Type &acc) {
    acc += Hf_ux.template get<I * INPUT_SIZE + K, J_idx>() *
           dX.template get<J_idx, 0>();
    AccumulateJ<Fuxx_Type, dX_Type, Value_Type, STATE_SIZE, INPUT_SIZE, I, K,
                (J_idx - 1)>::compute(Hf_ux, dX, acc);
  }
};

// J-accumulation termination (J_idx == 0)
/**
 * @brief Specialization of the AccumulateJ struct for the case when J == 0.
 *
 * This struct provides a static compute function that performs an accumulation
 * operation for a single element (when the recursion index J is zero). It
 * multiplies a specific element from the Hf_ux matrix with a corresponding
 * element from the dX vector and adds the result to the accumulator acc.
 *
 * @tparam Fuxx_Type   Type of the Hf_ux matrix.
 * @tparam dX_Type     Type of the dX vector.
 * @tparam Value_Type  Type of the accumulator.
 * @tparam STATE_SIZE  Number of states (size of the state vector).
 * @tparam INPUT_SIZE  Number of inputs (size of the input vector).
 * @tparam I           Row index parameter.
 * @tparam K           Column index parameter.
 *
 * @param Hf_ux  The matrix from which an element is selected.
 * @param dX     The vector from which an element is selected.
 * @param acc    The accumulator to which the computed product is added.
 */
template <typename Fuxx_Type, typename dX_Type, typename Value_Type,
          std::size_t STATE_SIZE, std::size_t INPUT_SIZE, std::size_t I,
          std::size_t K>
struct AccumulateJ<Fuxx_Type, dX_Type, Value_Type, STATE_SIZE, INPUT_SIZE, I, K,
                   0> {
  static void compute(const Fuxx_Type &Hf_ux, const dX_Type &dX,
                      Value_Type &acc) {
    acc +=
        Hf_ux.template get<I * INPUT_SIZE + K, 0>() * dX.template get<0, 0>();
  }
};

// Column recursion over k (INPUT_SIZE): computes contribution to out(k,0)
/**
 * @brief Computes and updates a column in an output matrix using weighted
 * accumulation.
 *
 * This struct template recursively computes a column of an output matrix by
 * accumulating weighted contributions from the provided matrices/vectors. The
 * computation is performed for a specific index `I` and column index `K_idx`,
 * and proceeds recursively for all columns down to zero.
 *
 * @tparam Fuxx_Type   Type of the Hf_ux matrix (e.g., Eigen matrix or custom
 * type).
 * @tparam dX_Type     Type of the dX vector/matrix.
 * @tparam Weight_Type Type of the lam_next weight vector/matrix.
 * @tparam Out_Type    Type of the output matrix.
 * @tparam STATE_SIZE  Number of states (rows).
 * @tparam INPUT_SIZE  Number of inputs (columns).
 * @tparam I           Current row index for computation.
 * @tparam K_idx       Current column index for computation (recursively
 * decremented).
 *
 * @note This struct assumes that the involved types provide `template get<row,
 * col>()` and `template set<row, col>(value)` member functions for element
 * access and assignment.
 *
 * @param Hf_ux     The input matrix containing values to be accumulated.
 * @param dX        The input vector/matrix used in the accumulation.
 * @param lam_next  The weight vector/matrix applied to the accumulated value.
 * @param out       The output matrix to be updated.
 */
template <typename Fuxx_Type, typename dX_Type, typename Weight_Type,
          typename Out_Type, std::size_t STATE_SIZE, std::size_t INPUT_SIZE,
          std::size_t I, std::size_t K_idx>
struct Column {
  static void compute(const Fuxx_Type &Hf_ux, const dX_Type &dX,
                      const Weight_Type &lam_next, Out_Type &out) {
    using Value = typename Out_Type::Value_Type;
    Value acc = static_cast<Value>(0);
    AccumulateJ<Fuxx_Type, dX_Type, Value, STATE_SIZE, INPUT_SIZE, I, K_idx,
                (STATE_SIZE - 1)>::compute(Hf_ux, dX, acc);

    const auto w = lam_next.template get<I, 0>();
    const auto updated = out.template get<K_idx, 0>() + w * acc;
    out.template set<K_idx, 0>(updated);

    Column<Fuxx_Type, dX_Type, Weight_Type, Out_Type, STATE_SIZE, INPUT_SIZE, I,
           (K_idx - 1)>::compute(Hf_ux, dX, lam_next, out);
  }
};

// Column recursion termination for k == 0

/**
 * @brief Specialization of the Column struct for matrix operations in SQP
 * optimization.
 *
 * This specialization handles the case where the second template parameter (J)
 * is 0. It computes a weighted accumulation of the product between the
 * Hessian-like matrix (Hf_ux) and the direction vector (dX), then updates the
 * output matrix (out) at position (0, 0) using the provided weight (lam_next).
 *
 * @tparam Fuxx_Type   Type representing the Hessian-like matrix.
 * @tparam dX_Type     Type representing the direction vector.
 * @tparam Weight_Type Type representing the weight matrix or vector.
 * @tparam Out_Type    Type representing the output matrix.
 * @tparam STATE_SIZE  Number of state variables.
 * @tparam INPUT_SIZE  Number of input variables.
 * @tparam I           Current column index.
 *
 * @param Hf_ux     The Hessian-like matrix.
 * @param dX        The direction vector.
 * @param lam_next  The weight matrix or vector for the next step.
 * @param out       The output matrix to be updated.
 */
template <typename Fuxx_Type, typename dX_Type, typename Weight_Type,
          typename Out_Type, std::size_t STATE_SIZE, std::size_t INPUT_SIZE,
          std::size_t I>
struct Column<Fuxx_Type, dX_Type, Weight_Type, Out_Type, STATE_SIZE, INPUT_SIZE,
              I, 0> {
  static void compute(const Fuxx_Type &Hf_ux, const dX_Type &dX,
                      const Weight_Type &lam_next, Out_Type &out) {
    using Value = typename Out_Type::Value_Type;
    Value acc = static_cast<Value>(0);
    AccumulateJ<Fuxx_Type, dX_Type, Value, STATE_SIZE, INPUT_SIZE, I, 0,
                (STATE_SIZE - 1)>::compute(Hf_ux, dX, acc);

    const auto w = lam_next.template get<I, 0>();
    const auto updated = out.template get<0, 0>() + w * acc;
    out.template set<0, 0>(updated);
  }
};

// Row recursion over i (STATE_SIZE): iterates the outer state index
/**
 * @brief Recursive template struct to compute a row operation for SQP matrix
 * operations.
 *
 * This struct recursively computes operations for a specific row (indexed by
 * I_idx) in a matrix operation, typically used in Sequential Quadratic
 * Programming (SQP) optimization. The computation is performed by invoking the
 * corresponding Column computation for the current index and then recursively
 * calling itself for the previous row index.
 *
 * @tparam Fuxx_Type   Type representing the Hessian or related matrix.
 * @tparam dX_Type     Type representing the delta or update vector.
 * @tparam Weight_Type Type representing the weighting or Lagrange multipliers.
 * @tparam Out_Type    Type representing the output container.
 * @tparam STATE_SIZE  Number of states in the system.
 * @tparam INPUT_SIZE  Number of inputs in the system.
 * @tparam I_idx       Current row index being processed (recursively
 * decremented).
 */
template <typename Fuxx_Type, typename dX_Type, typename Weight_Type,
          typename Out_Type, std::size_t STATE_SIZE, std::size_t INPUT_SIZE,
          std::size_t I_idx>
struct Row {
  static void compute(const Fuxx_Type &Hf_ux, const dX_Type &dX,
                      const Weight_Type &lam_next, Out_Type &out) {
    Column<Fuxx_Type, dX_Type, Weight_Type, Out_Type, STATE_SIZE, INPUT_SIZE,
           I_idx, (INPUT_SIZE - 1)>::compute(Hf_ux, dX, lam_next, out);
    Row<Fuxx_Type, dX_Type, Weight_Type, Out_Type, STATE_SIZE, INPUT_SIZE,
        (I_idx - 1)>::compute(Hf_ux, dX, lam_next, out);
  }
};

// Row recursion termination for i == 0
template <typename Fuxx_Type, typename dX_Type, typename Weight_Type,
          typename Out_Type, std::size_t STATE_SIZE, std::size_t INPUT_SIZE>
struct Row<Fuxx_Type, dX_Type, Weight_Type, Out_Type, STATE_SIZE, INPUT_SIZE,
           0> {
  static void compute(const Fuxx_Type &Hf_ux, const dX_Type &dX,
                      const Weight_Type &lam_next, Out_Type &out) {
    Column<Fuxx_Type, dX_Type, Weight_Type, Out_Type, STATE_SIZE, INPUT_SIZE, 0,
           (INPUT_SIZE - 1)>::compute(Hf_ux, dX, lam_next, out);
  }
};

} // namespace FuxxLambdaContract

/**
 * @brief Computes the contraction of the Hessian matrix Hf_ux with vectors dX
 * and lam_next, and stores the result in the output vector out.
 *
 * This function performs a specialized matrix-vector operation used in SQP
 * optimization, contracting the Hessian with the direction vector and the
 * Lagrange multipliers.
 *
 * @tparam Fuxx_Type Type representing the Hessian matrix (Hf_ux).
 * @tparam dX_Type Type representing the direction vector (dX).
 * @tparam Weight_Type Type representing the Lagrange multipliers vector
 * (lam_next).
 * @tparam Out_Type Type representing the output vector (out).
 *
 * @param Hf_ux The Hessian matrix of size (STATE_SIZE x STATE_SIZE *
 * INPUT_SIZE).
 * @param dX The direction vector of size (1 x STATE_SIZE).
 * @param lam_next The Lagrange multipliers vector of size (1 x STATE_SIZE).
 * @param out The output vector of size (1 x INPUT_SIZE) to store the result.
 *
 * @note All types must satisfy the static assertions regarding their
 * dimensions.
 * @note This function delegates the computation to
 * FuxxLambdaContract::Row::compute.
 */
template <typename Fuxx_Type, typename dX_Type, typename Weight_Type,
          typename Out_Type>
inline void
compute_fu_xx_lambda_contract(const Fuxx_Type &Hf_ux, const dX_Type &dX,
                              const Weight_Type &lam_next, Out_Type &out) {

  static_assert(dX_Type::ROWS == 1, "dX must be a (STATE_SIZE x 1) vector");
  static_assert(Weight_Type::ROWS == 1,
                "lam_next must be a (STATE_SIZE x 1) vector");
  static_assert(Out_Type::ROWS == 1,
                "out must be a (INPUT_SIZE x 1) vector (ROWS == 1)");

  constexpr std::size_t STATE_SIZE = dX_Type::COLS;
  constexpr std::size_t INPUT_SIZE = Out_Type::COLS;

  static_assert(STATE_SIZE > 0 && INPUT_SIZE > 0,
                "STATE_SIZE and INPUT_SIZE must be positive");
  static_assert(Fuxx_Type::COLS == STATE_SIZE * INPUT_SIZE,
                "Hf_ux ROWS must equal STATE_SIZE * INPUT_SIZE");
  static_assert(Fuxx_Type::ROWS == STATE_SIZE,
                "Hf_ux COLS must equal STATE_SIZE");
  static_assert(Weight_Type::COLS == STATE_SIZE,
                "lam_next COLS must equal STATE_SIZE");
  static_assert(Out_Type::COLS == INPUT_SIZE, "out COLS must equal INPUT_SIZE");

  FuxxLambdaContract::Row<Fuxx_Type, dX_Type, Weight_Type, Out_Type, STATE_SIZE,
                          INPUT_SIZE, (STATE_SIZE - 1)>::compute(Hf_ux, dX,
                                                                 lam_next, out);
}

/* fu uu lambda contract */

namespace FuuuLambdaContract {

// K-accumulation: recursively accumulate over k (INPUT_SIZE dimension)
template <typename Fuu_Type, typename dU_Type, typename Value_Type,
          std::size_t STATE_SIZE, std::size_t INPUT_SIZE, std::size_t I,
          std::size_t J, std::size_t K_idx>
struct AccumulateK {
  static void compute(const Fuu_Type &Hf_uu, const dU_Type &dU,
                      Value_Type &acc) {
    acc += Hf_uu.template get<I * INPUT_SIZE + J, K_idx>() *
           dU.template get<K_idx, 0>();
    AccumulateK<Fuu_Type, dU_Type, Value_Type, STATE_SIZE, INPUT_SIZE, I, J,
                (K_idx - 1)>::compute(Hf_uu, dU, acc);
  }
};

// K-accumulation termination (K_idx == 0)
template <typename Fuu_Type, typename dU_Type, typename Value_Type,
          std::size_t STATE_SIZE, std::size_t INPUT_SIZE, std::size_t I,
          std::size_t J>
struct AccumulateK<Fuu_Type, dU_Type, Value_Type, STATE_SIZE, INPUT_SIZE, I, J,
                   0> {
  static void compute(const Fuu_Type &Hf_uu, const dU_Type &dU,
                      Value_Type &acc) {
    acc +=
        Hf_uu.template get<I * INPUT_SIZE + J, 0>() * dU.template get<0, 0>();
  }
};

// Column recursion over j (INPUT_SIZE): computes contribution to out(j,0)
template <typename Fuu_Type, typename dU_Type, typename Weight_Type,
          typename Out_Type, std::size_t STATE_SIZE, std::size_t INPUT_SIZE,
          std::size_t I, std::size_t J_idx>
struct Column {
  static void compute(const Fuu_Type &Hf_uu, const dU_Type &dU,
                      const Weight_Type &lam_next, Out_Type &out) {
    using Value = typename Out_Type::Value_Type;
    Value acc = static_cast<Value>(0);
    AccumulateK<Fuu_Type, dU_Type, Value, STATE_SIZE, INPUT_SIZE, I, J_idx,
                (INPUT_SIZE - 1)>::compute(Hf_uu, dU, acc);

    const auto w = lam_next.template get<I, 0>();
    const auto updated = out.template get<J_idx, 0>() + w * acc;
    out.template set<J_idx, 0>(updated);

    Column<Fuu_Type, dU_Type, Weight_Type, Out_Type, STATE_SIZE, INPUT_SIZE, I,
           (J_idx - 1)>::compute(Hf_uu, dU, lam_next, out);
  }
};

// Column recursion termination for j == 0
template <typename Fuu_Type, typename dU_Type, typename Weight_Type,
          typename Out_Type, std::size_t STATE_SIZE, std::size_t INPUT_SIZE,
          std::size_t I>
struct Column<Fuu_Type, dU_Type, Weight_Type, Out_Type, STATE_SIZE, INPUT_SIZE,
              I, 0> {
  static void compute(const Fuu_Type &Hf_uu, const dU_Type &dU,
                      const Weight_Type &lam_next, Out_Type &out) {
    using Value = typename Out_Type::Value_Type;
    Value acc = static_cast<Value>(0);
    AccumulateK<Fuu_Type, dU_Type, Value, STATE_SIZE, INPUT_SIZE, I, 0,
                (INPUT_SIZE - 1)>::compute(Hf_uu, dU, acc);

    const auto w = lam_next.template get<I, 0>();
    const auto updated = out.template get<0, 0>() + w * acc;
    out.template set<0, 0>(updated);
  }
};

// Row recursion over i (STATE_SIZE): iterates the outer state index
template <typename Fuu_Type, typename dU_Type, typename Weight_Type,
          typename Out_Type, std::size_t STATE_SIZE, std::size_t INPUT_SIZE,
          std::size_t I_idx>
struct Row {
  static void compute(const Fuu_Type &Hf_uu, const dU_Type &dU,
                      const Weight_Type &lam_next, Out_Type &out) {
    Column<Fuu_Type, dU_Type, Weight_Type, Out_Type, STATE_SIZE, INPUT_SIZE,
           I_idx, (INPUT_SIZE - 1)>::compute(Hf_uu, dU, lam_next, out);
    Row<Fuu_Type, dU_Type, Weight_Type, Out_Type, STATE_SIZE, INPUT_SIZE,
        (I_idx - 1)>::compute(Hf_uu, dU, lam_next, out);
  }
};

// Row recursion termination for i == 0
template <typename Fuu_Type, typename dU_Type, typename Weight_Type,
          typename Out_Type, std::size_t STATE_SIZE, std::size_t INPUT_SIZE>
struct Row<Fuu_Type, dU_Type, Weight_Type, Out_Type, STATE_SIZE, INPUT_SIZE,
           0> {
  static void compute(const Fuu_Type &Hf_uu, const dU_Type &dU,
                      const Weight_Type &lam_next, Out_Type &out) {
    Column<Fuu_Type, dU_Type, Weight_Type, Out_Type, STATE_SIZE, INPUT_SIZE, 0,
           (INPUT_SIZE - 1)>::compute(Hf_uu, dU, lam_next, out);
  }
};

template <typename Fuu_Type, typename dU_Type, typename Weight_Type,
          typename Out_Type, std::size_t STATE_SIZE, std::size_t INPUT_SIZE,
          std::size_t I_idx, bool Activate>
struct Conditional {};

template <typename Fuu_Type, typename dU_Type, typename Weight_Type,
          typename Out_Type, std::size_t STATE_SIZE, std::size_t INPUT_SIZE,
          std::size_t I_idx>
struct Conditional<Fuu_Type, dU_Type, Weight_Type, Out_Type, STATE_SIZE,
                   INPUT_SIZE, I_idx, true> {
  static void compute(const Fuu_Type &Hf_uu, const dU_Type &dU,
                      const Weight_Type &lam_next, Out_Type &out) {

    Row<Fuu_Type, dU_Type, Weight_Type, Out_Type, STATE_SIZE, INPUT_SIZE,
        (STATE_SIZE - 1)>::compute(Hf_uu, dU, lam_next, out);
  }
};

template <typename Fuu_Type, typename dU_Type, typename Weight_Type,
          typename Out_Type, std::size_t STATE_SIZE, std::size_t INPUT_SIZE,
          std::size_t I_idx>
struct Conditional<Fuu_Type, dU_Type, Weight_Type, Out_Type, STATE_SIZE,
                   INPUT_SIZE, I_idx, false> {
  static void compute(const Fuu_Type &Hf_uu, const dU_Type &dU,
                      const Weight_Type &lam_next, Out_Type &out) {

    static_cast<void>(Hf_uu);
    static_cast<void>(dU);
    static_cast<void>(lam_next);
    static_cast<void>(out);
    /* Do Nothing */
  }
};

} // namespace FuuuLambdaContract

/**
 * @brief Computes the contraction of the Hessian matrix Hf_uu with the input
 * vector dU and the weight vector lam_next, storing the result in the output
 * vector out.
 *
 * This function performs a specialized matrix operation used in Sequential
 * Quadratic Programming (SQP) optimization routines. It contracts the Hessian
 * matrix Hf_uu with the input update vector dU and the next-step Lagrange
 * multiplier vector lam_next, producing an output vector out. The operation is
 * dispatched to a conditional implementation based on template parameters.
 *
 * @tparam Fuu_Type      Type representing the Hessian matrix (INPUT_SIZE x
 * STATE_SIZE * INPUT_SIZE).
 * @tparam dU_Type       Type representing the input vector (1 x INPUT_SIZE).
 * @tparam Weight_Type   Type representing the weight (Lagrange multiplier)
 * vector (1 x STATE_SIZE).
 * @tparam Out_Type      Type representing the output vector (1 x INPUT_SIZE).
 *
 * @param Hf_uu          The Hessian matrix of the objective function with
 * respect to inputs.
 * @param dU             The input update vector.
 * @param lam_next       The next-step Lagrange multiplier vector.
 * @param out            Output vector to store the result of the contraction.
 *
 * @note All input types must satisfy the specified static assertions regarding
 * their dimensions.
 * @note This function is intended for use in optimization routines where matrix
 * contraction is required.
 */
template <typename Fuu_Type, typename dU_Type, typename Weight_Type,
          typename Out_Type>
inline void
compute_fu_uu_lambda_contract(const Fuu_Type &Hf_uu, const dU_Type &dU,
                              const Weight_Type &lam_next, Out_Type &out) {

  static_assert(dU_Type::ROWS == 1, "dU must be a (INPUT_SIZE x 1) vector");
  static_assert(Weight_Type::ROWS == 1,
                "lam_next must be a (STATE_SIZE x 1) vector");
  static_assert(Out_Type::ROWS == 1,
                "out must be a (INPUT_SIZE x 1) vector (ROWS == 1)");

  constexpr std::size_t INPUT_SIZE = dU_Type::COLS;
  constexpr std::size_t STATE_SIZE = Weight_Type::COLS;

  static_assert(STATE_SIZE > 0 && INPUT_SIZE > 0,
                "STATE_SIZE and INPUT_SIZE must be positive");
  static_assert(Fuu_Type::COLS == STATE_SIZE * INPUT_SIZE,
                "Hf_uu ROWS must equal STATE_SIZE * INPUT_SIZE");
  static_assert(Fuu_Type::ROWS == INPUT_SIZE,
                "Hf_uu COLS must equal INPUT_SIZE");
  static_assert(Out_Type::COLS == INPUT_SIZE, "out COLS must equal INPUT_SIZE");

  FuuuLambdaContract::Conditional<Fuu_Type, dU_Type, Weight_Type, Out_Type,
                                  STATE_SIZE, INPUT_SIZE, (STATE_SIZE - 1),
                                  (INPUT_SIZE > 0)>::compute(Hf_uu, dU,
                                                             lam_next, out);
}

/* hxx lambda contract */

namespace HxxLambdaContract {

// K-accumulation: recursively accumulate over k (STATE_SIZE dimension)
template <typename Hxx_Type, typename dX_Type, typename Value_Type,
          std::size_t OUTPUT_SIZE, std::size_t STATE_SIZE, std::size_t I,
          std::size_t J, std::size_t K_idx>
struct AccumulateK {
  static void compute(const Hxx_Type &Hh_xx, const dX_Type &dX,
                      Value_Type &acc) {
    acc += Hh_xx.template get<I * STATE_SIZE + J, K_idx>() *
           dX.template get<K_idx, 0>();
    AccumulateK<Hxx_Type, dX_Type, Value_Type, OUTPUT_SIZE, STATE_SIZE, I, J,
                (K_idx - 1)>::compute(Hh_xx, dX, acc);
  }
};

// K-accumulation termination (K_idx == 0)
template <typename Hxx_Type, typename dX_Type, typename Value_Type,
          std::size_t OUTPUT_SIZE, std::size_t STATE_SIZE, std::size_t I,
          std::size_t J>
struct AccumulateK<Hxx_Type, dX_Type, Value_Type, OUTPUT_SIZE, STATE_SIZE, I, J,
                   0> {
  static void compute(const Hxx_Type &Hh_xx, const dX_Type &dX,
                      Value_Type &acc) {
    acc +=
        Hh_xx.template get<I * STATE_SIZE + J, 0>() * dX.template get<0, 0>();
  }
};

// Column recursion over j (STATE_SIZE): computes contribution to out(j,0)
template <typename Hxx_Type, typename dX_Type, typename Weight_Type,
          typename Out_Type, std::size_t OUTPUT_SIZE, std::size_t STATE_SIZE,
          std::size_t I, std::size_t J_idx>
struct Column {
  static void compute(const Hxx_Type &Hh_xx, const dX_Type &dX,
                      const Weight_Type &weight, Out_Type &out) {
    using Value = typename Out_Type::Value_Type;
    Value acc = static_cast<Value>(0);
    AccumulateK<Hxx_Type, dX_Type, Value, OUTPUT_SIZE, STATE_SIZE, I, J_idx,
                (STATE_SIZE - 1)>::compute(Hh_xx, dX, acc);

    const auto w = weight.template get<I, 0>();
    const auto updated = out.template get<J_idx, 0>() + w * acc;
    out.template set<J_idx, 0>(updated);

    Column<Hxx_Type, dX_Type, Weight_Type, Out_Type, OUTPUT_SIZE, STATE_SIZE, I,
           (J_idx - 1)>::compute(Hh_xx, dX, weight, out);
  }
};

// Column recursion termination for j == 0
template <typename Hxx_Type, typename dX_Type, typename Weight_Type,
          typename Out_Type, std::size_t OUTPUT_SIZE, std::size_t STATE_SIZE,
          std::size_t I>
struct Column<Hxx_Type, dX_Type, Weight_Type, Out_Type, OUTPUT_SIZE, STATE_SIZE,
              I, 0> {
  static void compute(const Hxx_Type &Hh_xx, const dX_Type &dX,
                      const Weight_Type &weight, Out_Type &out) {
    using Value = typename Out_Type::Value_Type;
    Value acc = static_cast<Value>(0);
    AccumulateK<Hxx_Type, dX_Type, Value, OUTPUT_SIZE, STATE_SIZE, I, 0,
                (STATE_SIZE - 1)>::compute(Hh_xx, dX, acc);

    const auto w = weight.template get<I, 0>();
    const auto updated = out.template get<0, 0>() + w * acc;
    out.template set<0, 0>(updated);
  }
};

// Row recursion over i (OUTPUT_SIZE): iterates outputs
template <typename Hxx_Type, typename dX_Type, typename Weight_Type,
          typename Out_Type, std::size_t OUTPUT_SIZE, std::size_t STATE_SIZE,
          std::size_t I_idx>
struct Row {
  static void compute(const Hxx_Type &Hh_xx, const dX_Type &dX,
                      const Weight_Type &weight, Out_Type &out) {
    Column<Hxx_Type, dX_Type, Weight_Type, Out_Type, OUTPUT_SIZE, STATE_SIZE,
           I_idx, (STATE_SIZE - 1)>::compute(Hh_xx, dX, weight, out);
    Row<Hxx_Type, dX_Type, Weight_Type, Out_Type, OUTPUT_SIZE, STATE_SIZE,
        (I_idx - 1)>::compute(Hh_xx, dX, weight, out);
  }
};

// Row recursion termination for i == 0
template <typename Hxx_Type, typename dX_Type, typename Weight_Type,
          typename Out_Type, std::size_t OUTPUT_SIZE, std::size_t STATE_SIZE>
struct Row<Hxx_Type, dX_Type, Weight_Type, Out_Type, OUTPUT_SIZE, STATE_SIZE,
           0> {
  static void compute(const Hxx_Type &Hh_xx, const dX_Type &dX,
                      const Weight_Type &weight, Out_Type &out) {
    Column<Hxx_Type, dX_Type, Weight_Type, Out_Type, OUTPUT_SIZE, STATE_SIZE, 0,
           (STATE_SIZE - 1)>::compute(Hh_xx, dX, weight, out);
  }
};

} // namespace HxxLambdaContract

/**
 * @brief Computes the contraction of a Hessian matrix with a direction vector
 * and a weight vector.
 *
 * This function performs a specialized matrix operation, contracting the
 * provided Hessian matrix (`Hh_xx`) with the direction vector (`dX`) and the
 * weight vector (`weight`), and stores the result in `out`. The operation is
 * typically used in Sequential Quadratic Programming (SQP) optimization
 * routines.
 *
 * @tparam Hxx_Type Type representing the Hessian matrix, expected to have
 * dimensions (STATE_SIZE x OUTPUT_SIZE * STATE_SIZE).
 * @tparam dX_Type Type representing the direction vector, expected to be a (1 x
 * STATE_SIZE) vector.
 * @tparam Weight_Type Type representing the weight vector, expected to be a (1
 * x OUTPUT_SIZE) vector.
 * @tparam Out_Type Type representing the output vector, expected to be a (1 x
 * STATE_SIZE) vector.
 *
 * @param Hh_xx The Hessian matrix to be contracted.
 * @param dX The direction vector for contraction.
 * @param weight The weight vector for contraction.
 * @param out Output vector to store the result of the contraction.
 *
 * @note All input types are statically checked for correct dimensions.
 * @note This function delegates the actual computation to
 * HxxLambdaContract::Row.
 */
template <typename Hxx_Type, typename dX_Type, typename Weight_Type,
          typename Out_Type>
inline void
compute_hxx_lambda_contract(const Hxx_Type &Hh_xx, const dX_Type &dX,
                            const Weight_Type &weight, Out_Type &out) {

  static_assert(dX_Type::ROWS == 1, "dX must be a (STATE_SIZE x 1) vector");
  static_assert(Weight_Type::ROWS == 1,
                "weight must be a (OUTPUT_SIZE x 1) vector");
  static_assert(Out_Type::ROWS == 1,
                "out must be a (STATE_SIZE x 1) vector (ROWS == 1)");

  constexpr std::size_t STATE_SIZE = dX_Type::COLS;
  constexpr std::size_t OUTPUT_SIZE = Weight_Type::COLS;

  static_assert(STATE_SIZE > 0 && OUTPUT_SIZE > 0,
                "STATE_SIZE and OUTPUT_SIZE must be positive");
  static_assert(Hxx_Type::ROWS == STATE_SIZE,
                "Hh_xx ROWS must equal STATE_SIZE");
  static_assert(Hxx_Type::COLS == OUTPUT_SIZE * STATE_SIZE,
                "Hh_xx COLS must equal OUTPUT_SIZE * STATE_SIZE");
  static_assert(Out_Type::COLS == STATE_SIZE, "out COLS must equal STATE_SIZE");

  HxxLambdaContract::Row<Hxx_Type, dX_Type, Weight_Type, Out_Type, OUTPUT_SIZE,
                         STATE_SIZE, (OUTPUT_SIZE - 1)>::compute(Hh_xx, dX,
                                                                 weight, out);
}

/* free_mask at check */

namespace FreeMaskAtCheck {

// Per-element conditional for lower-bound proximity check
template <typename U_Mat_Type, typename U_Min_Matrix_Type,
          typename AtLower_Type, typename Value_Type, std::size_t I,
          std::size_t J_idx, bool limit_valid_flag>
struct LowerConditional {};

template <typename U_Mat_Type, typename U_Min_Matrix_Type,
          typename AtLower_Type, typename Value_Type, std::size_t I,
          std::size_t J_idx>
struct LowerConditional<U_Mat_Type, U_Min_Matrix_Type, AtLower_Type, Value_Type,
                        I, J_idx, true> {
  static inline void compute(const U_Mat_Type &U_horizon_in,
                             const U_Min_Matrix_Type &U_min_matrix,
                             const Value_Type &atol, AtLower_Type &at_lower) {

    const auto u = U_horizon_in.template get<I, J_idx>();
    const auto u_min = U_min_matrix.template get<I, J_idx>();

    if ((u >= (u_min - atol)) && (u <= (u_min + atol))) {
      at_lower.template set<I, J_idx>(true);
    }
  }
};

template <typename U_Mat_Type, typename U_Min_Matrix_Type,
          typename AtLower_Type, typename Value_Type, std::size_t I,
          std::size_t J_idx>
struct LowerConditional<U_Mat_Type, U_Min_Matrix_Type, AtLower_Type, Value_Type,
                        I, J_idx, false> {
  static inline void compute(const U_Mat_Type &U_horizon_in,
                             const U_Min_Matrix_Type &U_min_matrix,
                             const Value_Type &atol, AtLower_Type &at_lower) {

    static_cast<void>(U_horizon_in);
    static_cast<void>(U_min_matrix);
    static_cast<void>(atol);
    static_cast<void>(at_lower);
    /* Do Nothing */
  }
};

// Per-element conditional for upper-bound proximity check
template <typename U_Mat_Type, typename U_Max_Matrix_Type,
          typename AtUpper_Type, typename Value_Type, std::size_t I,
          std::size_t J_idx, bool limit_valid_flag>
struct UpperConditional {};

template <typename U_Mat_Type, typename U_Max_Matrix_Type,
          typename AtUpper_Type, typename Value_Type, std::size_t I,
          std::size_t J_idx>
struct UpperConditional<U_Mat_Type, U_Max_Matrix_Type, AtUpper_Type, Value_Type,
                        I, J_idx, true> {
  static inline void compute(const U_Mat_Type &U_horizon_in,
                             const U_Max_Matrix_Type &U_max_matrix,
                             const Value_Type &atol, AtUpper_Type &at_upper) {

    const auto u = U_horizon_in.template get<I, J_idx>();
    const auto u_max = U_max_matrix.template get<I, J_idx>();

    if ((u >= (u_max - atol)) && (u <= (u_max + atol))) {
      at_upper.template set<I, J_idx>(true);
    }
  }
};

template <typename U_Mat_Type, typename U_Max_Matrix_Type,
          typename AtUpper_Type, typename Value_Type, std::size_t I,
          std::size_t J_idx>
struct UpperConditional<U_Mat_Type, U_Max_Matrix_Type, AtUpper_Type, Value_Type,
                        I, J_idx, false> {
  static inline void compute(const U_Mat_Type &U_horizon_in,
                             const U_Max_Matrix_Type &U_max_matrix,
                             const Value_Type &atol, AtUpper_Type &at_upper) {

    static_cast<void>(U_horizon_in);
    static_cast<void>(U_max_matrix);
    static_cast<void>(atol);
    static_cast<void>(at_upper);
    /* Do Nothing */
  }
};

// Column recursion for J (0..N-1), when J_idx > 0
template <typename U_Mat_Type, typename U_Min_Matrix_Type,
          typename U_Max_Matrix_Type, typename AtLower_Type,
          typename AtUpper_Type, typename Value_Type, std::size_t M,
          std::size_t N, std::size_t I, std::size_t J_idx>
struct Column {
  static inline void compute(const U_Mat_Type &U_horizon_in,
                             const U_Min_Matrix_Type &U_min_matrix,
                             const U_Max_Matrix_Type &U_max_matrix,
                             const Value_Type &atol, AtLower_Type &at_lower,
                             AtUpper_Type &at_upper) {

    LowerConditional<U_Mat_Type, U_Min_Matrix_Type, AtLower_Type, Value_Type, I,
                     J_idx,
                     U_Min_Matrix_Type::SparseAvailable_Type::lists[I][J_idx]>::
        compute(U_horizon_in, U_min_matrix, atol, at_lower);

    UpperConditional<U_Mat_Type, U_Max_Matrix_Type, AtUpper_Type, Value_Type, I,
                     J_idx,
                     U_Max_Matrix_Type::SparseAvailable_Type::lists[I][J_idx]>::
        compute(U_horizon_in, U_max_matrix, atol, at_upper);

    Column<U_Mat_Type, U_Min_Matrix_Type, U_Max_Matrix_Type, AtLower_Type,
           AtUpper_Type, Value_Type, M, N, I,
           (J_idx - 1)>::compute(U_horizon_in, U_min_matrix, U_max_matrix, atol,
                                 at_lower, at_upper);
  }
};

// Column recursion termination for J_idx == 0
template <typename U_Mat_Type, typename U_Min_Matrix_Type,
          typename U_Max_Matrix_Type, typename AtLower_Type,
          typename AtUpper_Type, typename Value_Type, std::size_t M,
          std::size_t N, std::size_t I>
struct Column<U_Mat_Type, U_Min_Matrix_Type, U_Max_Matrix_Type, AtLower_Type,
              AtUpper_Type, Value_Type, M, N, I, 0> {
  static inline void compute(const U_Mat_Type &U_horizon_in,
                             const U_Min_Matrix_Type &U_min_matrix,
                             const U_Max_Matrix_Type &U_max_matrix,
                             const Value_Type &atol, AtLower_Type &at_lower,
                             AtUpper_Type &at_upper) {

    LowerConditional<U_Mat_Type, U_Min_Matrix_Type, AtLower_Type, Value_Type, I,
                     0, U_Min_Matrix_Type::SparseAvailable_Type::lists[I][0]>::
        compute(U_horizon_in, U_min_matrix, atol, at_lower);

    UpperConditional<U_Mat_Type, U_Max_Matrix_Type, AtUpper_Type, Value_Type, I,
                     0, U_Max_Matrix_Type::SparseAvailable_Type::lists[I][0]>::
        compute(U_horizon_in, U_max_matrix, atol, at_upper);
  }
};

// Row recursion for I (0..M-1), when I_idx > 0
template <typename U_Mat_Type, typename U_Min_Matrix_Type,
          typename U_Max_Matrix_Type, typename AtLower_Type,
          typename AtUpper_Type, typename Value_Type, std::size_t M,
          std::size_t N, std::size_t I_idx>
struct Row {
  static inline void compute(const U_Mat_Type &U_horizon_in,
                             const U_Min_Matrix_Type &U_min_matrix,
                             const U_Max_Matrix_Type &U_max_matrix,
                             const Value_Type &atol, AtLower_Type &at_lower,
                             AtUpper_Type &at_upper) {
    Column<U_Mat_Type, U_Min_Matrix_Type, U_Max_Matrix_Type, AtLower_Type,
           AtUpper_Type, Value_Type, M, N, I_idx,
           (N - 1)>::compute(U_horizon_in, U_min_matrix, U_max_matrix, atol,
                             at_lower, at_upper);

    Row<U_Mat_Type, U_Min_Matrix_Type, U_Max_Matrix_Type, AtLower_Type,
        AtUpper_Type, Value_Type, M, N, (I_idx - 1)>::compute(U_horizon_in,
                                                              U_min_matrix,
                                                              U_max_matrix,
                                                              atol, at_lower,
                                                              at_upper);
  }
};

// Row recursion termination for I_idx == 0
template <typename U_Mat_Type, typename U_Min_Matrix_Type,
          typename U_Max_Matrix_Type, typename AtLower_Type,
          typename AtUpper_Type, typename Value_Type, std::size_t M,
          std::size_t N>
struct Row<U_Mat_Type, U_Min_Matrix_Type, U_Max_Matrix_Type, AtLower_Type,
           AtUpper_Type, Value_Type, M, N, 0> {
  static inline void compute(const U_Mat_Type &U_horizon_in,
                             const U_Min_Matrix_Type &U_min_matrix,
                             const U_Max_Matrix_Type &U_max_matrix,
                             const Value_Type &atol, AtLower_Type &at_lower,
                             AtUpper_Type &at_upper) {
    Column<U_Mat_Type, U_Min_Matrix_Type, U_Max_Matrix_Type, AtLower_Type,
           AtUpper_Type, Value_Type, M, N, 0, (N - 1)>::compute(U_horizon_in,
                                                                U_min_matrix,
                                                                U_max_matrix,
                                                                atol, at_lower,
                                                                at_upper);
  }
};

} // namespace FreeMaskAtCheck

/**
 * @brief Checks and updates mask matrices indicating whether elements of the
 * input matrix are at their lower or upper bounds within a specified tolerance.
 *
 * This function compares each element of the input matrix `U_horizon_in`
 * against the corresponding elements in the lower bound matrix `U_min_matrix`
 * and the upper bound matrix `U_max_matrix`. If an element is within the
 * specified absolute tolerance `atol` of its lower or upper bound, the
 * corresponding entry in the mask matrices `at_lower` or `at_upper` is updated
 * accordingly.
 *
 * Template Parameters:
 * - U_Mat_Type: Type of the input matrix.
 * - U_Min_Matrix_Type: Type of the lower bound matrix.
 * - U_Max_Matrix_Type: Type of the upper bound matrix.
 * - AtLower_Type: Type of the mask matrix for lower bounds.
 * - AtUpper_Type: Type of the mask matrix for upper bounds.
 * - Value_Type: Type of the tolerance value.
 *
 * @param U_horizon_in   Input matrix to check.
 * @param U_min_matrix   Matrix of lower bounds.
 * @param U_max_matrix   Matrix of upper bounds.
 * @param atol           Absolute tolerance for bound checking.
 * @param at_lower       Output mask matrix indicating elements at lower bounds.
 * @param at_upper       Output mask matrix indicating elements at upper bounds.
 *
 * @note All matrix types must have matching dimensions. Static assertions are
 * used to enforce this.
 */
template <typename U_Mat_Type, typename U_Min_Matrix_Type,
          typename U_Max_Matrix_Type, typename AtLower_Type,
          typename AtUpper_Type, typename Value_Type>
inline void free_mask_at_check(const U_Mat_Type &U_horizon_in,
                               const U_Min_Matrix_Type &U_min_matrix,
                               const U_Max_Matrix_Type &U_max_matrix,
                               const Value_Type &atol, AtLower_Type &at_lower,
                               AtUpper_Type &at_upper) {

  constexpr std::size_t M = U_Mat_Type::COLS; // INPUT_SIZE
  constexpr std::size_t N = U_Mat_Type::ROWS; // NP

  static_assert(M > 0 && N > 0, "Matrix dimensions must be positive");
  static_assert(U_Min_Matrix_Type::COLS == M,
                "U_min_matrix COLS mismatch with U_horizon_in");
  static_assert(U_Min_Matrix_Type::ROWS == N,
                "U_min_matrix ROWS mismatch with U_horizon_in");
  static_assert(U_Max_Matrix_Type::COLS == M,
                "U_max_matrix COLS mismatch with U_horizon_in");
  static_assert(U_Max_Matrix_Type::ROWS == N,
                "U_max_matrix ROWS mismatch with U_horizon_in");
  static_assert(AtLower_Type::COLS == M,
                "at_lower COLS mismatch with U_horizon_in");
  static_assert(AtLower_Type::ROWS == N,
                "at_lower ROWS mismatch with U_horizon_in");
  static_assert(AtUpper_Type::COLS == M,
                "at_upper COLS mismatch with U_horizon_in");
  static_assert(AtUpper_Type::ROWS == N,
                "at_upper ROWS mismatch with U_horizon_in");

  FreeMaskAtCheck::Row<U_Mat_Type, U_Min_Matrix_Type, U_Max_Matrix_Type,
                       AtLower_Type, AtUpper_Type, Value_Type, M, N,
                       (M - 1)>::compute(U_horizon_in, U_min_matrix,
                                         U_max_matrix, atol, at_lower,
                                         at_upper);
}

/* free_mask push active */

namespace FreeMaskPushActive {

// Column recursion for J (0..N-1), when J_idx > 0
template <typename Mask_Type, typename Gradient_Type, typename AtLower_Type,
          typename AtUpper_Type, typename ActiveSet_Type, typename Value_Type,
          std::size_t M, std::size_t N, std::size_t I, std::size_t J_idx>
struct Column {
  static inline void compute(Mask_Type &m, const Gradient_Type &gradient,
                             const AtLower_Type &at_lower,
                             const AtUpper_Type &at_upper,
                             const Value_Type &gtol,
                             ActiveSet_Type &active_set) {

    const bool atL = at_lower.template get<I, J_idx>();
    const bool atU = at_upper.template get<I, J_idx>();
    const auto g = gradient.template get<I, J_idx>();

    if ((atL && (g > gtol)) || (atU && (g < -gtol))) {
      m.template set<I, J_idx>(false);
    } else {
      active_set.push_active(I, J_idx);
    }

    Column<Mask_Type, Gradient_Type, AtLower_Type, AtUpper_Type, ActiveSet_Type,
           Value_Type, M, N, I, (J_idx - 1)>::compute(m, gradient, at_lower,
                                                      at_upper, gtol,
                                                      active_set);
  }
};

// Column recursion termination for J_idx == 0
template <typename Mask_Type, typename Gradient_Type, typename AtLower_Type,
          typename AtUpper_Type, typename ActiveSet_Type, typename Value_Type,
          std::size_t M, std::size_t N, std::size_t I>
struct Column<Mask_Type, Gradient_Type, AtLower_Type, AtUpper_Type,
              ActiveSet_Type, Value_Type, M, N, I, 0> {
  static inline void compute(Mask_Type &m, const Gradient_Type &gradient,
                             const AtLower_Type &at_lower,
                             const AtUpper_Type &at_upper,
                             const Value_Type &gtol,
                             ActiveSet_Type &active_set) {

    const bool atL = at_lower.template get<I, 0>();
    const bool atU = at_upper.template get<I, 0>();
    const auto g = gradient.template get<I, 0>();

    if ((atL && (g > gtol)) || (atU && (g < -gtol))) {
      m.template set<I, 0>(false);
    } else {
      active_set.push_active(I, 0);
    }
  }
};

// Row recursion for I (0..M-1), when I_idx > 0
template <typename Mask_Type, typename Gradient_Type, typename AtLower_Type,
          typename AtUpper_Type, typename ActiveSet_Type, typename Value_Type,
          std::size_t M, std::size_t N, std::size_t I_idx>
struct Row {
  static inline void compute(Mask_Type &m, const Gradient_Type &gradient,
                             const AtLower_Type &at_lower,
                             const AtUpper_Type &at_upper,
                             const Value_Type &gtol,
                             ActiveSet_Type &active_set) {

    Column<Mask_Type, Gradient_Type, AtLower_Type, AtUpper_Type, ActiveSet_Type,
           Value_Type, M, N, I_idx, (N - 1)>::compute(m, gradient, at_lower,
                                                      at_upper, gtol,
                                                      active_set);

    Row<Mask_Type, Gradient_Type, AtLower_Type, AtUpper_Type, ActiveSet_Type,
        Value_Type, M, N, (I_idx - 1)>::compute(m, gradient, at_lower, at_upper,
                                                gtol, active_set);
  }
};

// Row recursion termination for I_idx == 0
template <typename Mask_Type, typename Gradient_Type, typename AtLower_Type,
          typename AtUpper_Type, typename ActiveSet_Type, typename Value_Type,
          std::size_t M, std::size_t N>
struct Row<Mask_Type, Gradient_Type, AtLower_Type, AtUpper_Type, ActiveSet_Type,
           Value_Type, M, N, 0> {
  static inline void compute(Mask_Type &m, const Gradient_Type &gradient,
                             const AtLower_Type &at_lower,
                             const AtUpper_Type &at_upper,
                             const Value_Type &gtol,
                             ActiveSet_Type &active_set) {

    Column<Mask_Type, Gradient_Type, AtLower_Type, AtUpper_Type, ActiveSet_Type,
           Value_Type, M, N, 0, (N - 1)>::compute(m, gradient, at_lower,
                                                  at_upper, gtol, active_set);
  }
};

} // namespace FreeMaskPushActive

/**
 * @brief Updates the mask and pushes active constraints based on gradient and
 * bounds.
 *
 * This function iterates over the mask matrix and, for each element, checks the
 * corresponding gradient, lower bound, and upper bound values. If the gradient
 * exceeds the specified tolerance
 * (`gtol`), and the variable is not at its lower or upper bound, the index is
 * pushed to the active set. The function uses compile-time assertions to ensure
 * that the input matrices have compatible dimensions.
 *
 * @tparam Mask_Type        Type of the mask matrix (must define COLS and ROWS).
 * @tparam Gradient_Type    Type of the gradient matrix (must match mask
 * dimensions).
 * @tparam AtLower_Type     Type indicating which variables are at their lower
 * bounds.
 * @tparam AtUpper_Type     Type indicating which variables are at their upper
 * bounds.
 * @tparam ActiveSet_Type   Type of the active set container.
 * @tparam Value_Type       Type of the gradient tolerance value.
 * @param m                 Reference to the mask matrix.
 * @param gradient          Reference to the gradient matrix.
 * @param at_lower          Reference to the lower bound indicator matrix.
 * @param at_upper          Reference to the upper bound indicator matrix.
 * @param gtol              Gradient tolerance value.
 * @param active_set        Reference to the active set container to be updated.
 */
template <typename Mask_Type, typename Gradient_Type, typename AtLower_Type,
          typename AtUpper_Type, typename ActiveSet_Type, typename Value_Type>
inline void free_mask_push_active(Mask_Type &m, const Gradient_Type &gradient,
                                  const AtLower_Type &at_lower,
                                  const AtUpper_Type &at_upper,
                                  const Value_Type &gtol,
                                  ActiveSet_Type &active_set) {

  constexpr std::size_t M = Mask_Type::COLS; // INPUT_SIZE
  constexpr std::size_t N = Mask_Type::ROWS; // NP

  static_assert(M > 0 && N > 0, "Matrix dimensions must be positive");
  static_assert(Gradient_Type::COLS == M, "gradient COLS mismatch with mask m");
  static_assert(Gradient_Type::ROWS == N, "gradient ROWS mismatch with mask m");
  static_assert(AtLower_Type::COLS == M, "at_lower COLS mismatch with mask m");
  static_assert(AtLower_Type::ROWS == N, "at_lower ROWS mismatch with mask m");
  static_assert(AtUpper_Type::COLS == M, "at_upper COLS mismatch with mask m");
  static_assert(AtUpper_Type::ROWS == N, "at_upper ROWS mismatch with mask m");

  FreeMaskPushActive::Row<Mask_Type, Gradient_Type, AtLower_Type, AtUpper_Type,
                          ActiveSet_Type, Value_Type, M, N,
                          (M - 1)>::compute(m, gradient, at_lower, at_upper,
                                            gtol, active_set);
}

/* solver calculate M_inv */

namespace SolverCalculateMInv {

// Column recursion for J (0..N-1), when J_idx > 0
template <typename Out_Mat_Type, typename In_Mat_Type, typename Value_Type,
          std::size_t M, std::size_t N, std::size_t I, std::size_t J_idx>
struct Column {
  static inline void compute(Out_Mat_Type &M_inv,
                             const In_Mat_Type &diag_R_full_lambda_factor,
                             const Value_Type &avoid_zero_limit) {

    const auto denominator = Base::Utility::avoid_zero_divide(
        diag_R_full_lambda_factor.template get<I, J_idx>(), avoid_zero_limit);

    const auto value = static_cast<Value_Type>(1) / denominator;
    M_inv.template set<I, J_idx>(value);

    Column<Out_Mat_Type, In_Mat_Type, Value_Type, M, N, I,
           (J_idx - 1)>::compute(M_inv, diag_R_full_lambda_factor,
                                 avoid_zero_limit);
  }
};

// Column recursion termination for J_idx == 0
template <typename Out_Mat_Type, typename In_Mat_Type, typename Value_Type,
          std::size_t M, std::size_t N, std::size_t I>
struct Column<Out_Mat_Type, In_Mat_Type, Value_Type, M, N, I, 0> {
  static inline void compute(Out_Mat_Type &M_inv,
                             const In_Mat_Type &diag_R_full_lambda_factor,
                             const Value_Type &avoid_zero_limit) {

    const auto denominator = Base::Utility::avoid_zero_divide(
        diag_R_full_lambda_factor.template get<I, 0>(), avoid_zero_limit);

    const auto value = static_cast<Value_Type>(1) / denominator;
    M_inv.template set<I, 0>(value);
  }
};

// Row recursion for I (0..M-1), when I_idx > 0
template <typename Out_Mat_Type, typename In_Mat_Type, typename Value_Type,
          std::size_t M, std::size_t N, std::size_t I_idx>
struct Row {
  static inline void compute(Out_Mat_Type &M_inv,
                             const In_Mat_Type &diag_R_full_lambda_factor,
                             const Value_Type &avoid_zero_limit) {
    Column<Out_Mat_Type, In_Mat_Type, Value_Type, M, N, I_idx,
           (N - 1)>::compute(M_inv, diag_R_full_lambda_factor,
                             avoid_zero_limit);

    Row<Out_Mat_Type, In_Mat_Type, Value_Type, M, N, (I_idx - 1)>::compute(
        M_inv, diag_R_full_lambda_factor, avoid_zero_limit);
  }
};

// Row recursion termination for I_idx == 0
template <typename Out_Mat_Type, typename In_Mat_Type, typename Value_Type,
          std::size_t M, std::size_t N>
struct Row<Out_Mat_Type, In_Mat_Type, Value_Type, M, N, 0> {
  static inline void compute(Out_Mat_Type &M_inv,
                             const In_Mat_Type &diag_R_full_lambda_factor,
                             const Value_Type &avoid_zero_limit) {
    Column<Out_Mat_Type, In_Mat_Type, Value_Type, M, N, 0, (N - 1)>::compute(
        M_inv, diag_R_full_lambda_factor, avoid_zero_limit);
  }
};

} // namespace SolverCalculateMInv

/**
 * @brief Calculates the inverse of a matrix M_inv using the provided diagonal
 * matrix diag_R_full_lambda_factor.
 *
 * This function computes the inverse of a matrix by leveraging a specialized
 * row-wise computation defined in SolverCalculateMInv::Row. It ensures that the
 * matrix dimensions are positive and that the input matrix dimensions match the
 * output matrix dimensions. The computation avoids division by zero by using
 * the provided avoid_zero_limit parameter.
 *
 * @tparam Out_Mat_Type Type of the output matrix (M_inv).
 * @tparam In_Mat_Type Type of the input diagonal matrix
 * (diag_R_full_lambda_factor).
 * @tparam Value_Type Scalar value type used in computation.
 * @param[out] M_inv Output matrix to store the inverse result.
 * @param[in] diag_R_full_lambda_factor Input diagonal matrix used for
 * inversion.
 * @param[in] avoid_zero_limit Value used to avoid division by zero during
 * inversion.
 */
template <typename Out_Mat_Type, typename In_Mat_Type, typename Value_Type>
inline void solver_calculate_M_inv(Out_Mat_Type &M_inv,
                                   const In_Mat_Type &diag_R_full_lambda_factor,
                                   const Value_Type &avoid_zero_limit) {

  constexpr std::size_t M = Out_Mat_Type::COLS;
  constexpr std::size_t N = Out_Mat_Type::ROWS;

  static_assert(M > 0 && N > 0, "Matrix dimensions must be positive");
  static_assert(In_Mat_Type::COLS == M,
                "diag_R_full_lambda_factor COLS mismatch with M_inv");
  static_assert(In_Mat_Type::ROWS == N,
                "diag_R_full_lambda_factor ROWS mismatch with M_inv");

  SolverCalculateMInv::Row<Out_Mat_Type, In_Mat_Type, Value_Type, M, N,
                           (M - 1)>::compute(M_inv, diag_R_full_lambda_factor,
                                             avoid_zero_limit);
}

/* saturate U_horizon */

namespace SaturateU_Horizon {

// Min-limit conditional saturator based on compile-time availability
template <typename U_Mat_Type, typename U_Min_Matrix_Type, std::size_t I,
          std::size_t J_idx, bool limit_valid_flag>
struct MinSaturateConditional {};

template <typename U_Mat_Type, typename U_Min_Matrix_Type, std::size_t I,
          std::size_t J_idx>
struct MinSaturateConditional<U_Mat_Type, U_Min_Matrix_Type, I, J_idx, true> {
  static inline void compute(U_Mat_Type &U_candidate,
                             const U_Min_Matrix_Type &U_min_matrix) {

    const auto u_min = U_min_matrix.template get<I, J_idx>();
    const auto u_val = U_candidate.template get<I, J_idx>();
    if (u_val < u_min) {
      U_candidate.template set<I, J_idx>(u_min);
    }
  }
};

template <typename U_Mat_Type, typename U_Min_Matrix_Type, std::size_t I,
          std::size_t J_idx>
struct MinSaturateConditional<U_Mat_Type, U_Min_Matrix_Type, I, J_idx, false> {
  static inline void compute(U_Mat_Type &U_candidate,
                             const U_Min_Matrix_Type &U_min_matrix) {

    static_cast<void>(U_candidate);
    static_cast<void>(U_min_matrix);
    /* Do Nothing. */
  }
};

// Max-limit conditional saturator based on compile-time availability
template <typename U_Mat_Type, typename U_Max_Matrix_Type, std::size_t I,
          std::size_t J_idx, bool limit_valid_flag>
struct MaxSaturateConditional {};

template <typename U_Mat_Type, typename U_Max_Matrix_Type, std::size_t I,
          std::size_t J_idx>
struct MaxSaturateConditional<U_Mat_Type, U_Max_Matrix_Type, I, J_idx, true> {
  static inline void compute(U_Mat_Type &U_candidate,
                             const U_Max_Matrix_Type &U_max_matrix) {

    const auto u_max = U_max_matrix.template get<I, J_idx>();
    const auto u_val = U_candidate.template get<I, J_idx>();
    if (u_val > u_max) {
      U_candidate.template set<I, J_idx>(u_max);
    }
  }
};

template <typename U_Mat_Type, typename U_Max_Matrix_Type, std::size_t I,
          std::size_t J_idx>
struct MaxSaturateConditional<U_Mat_Type, U_Max_Matrix_Type, I, J_idx, false> {
  static inline void compute(U_Mat_Type &U_candidate,
                             const U_Max_Matrix_Type &U_max_matrix) {

    static_cast<void>(U_candidate);
    static_cast<void>(U_max_matrix);
    /* Do Nothing. */
  }
};

// Column recursion for J (0..N-1), when J_idx > 0
template <typename U_Mat_Type, typename U_Min_Matrix_Type,
          typename U_Max_Matrix_Type, std::size_t M, std::size_t N,
          std::size_t I, std::size_t J_idx>
struct Column {
  static inline void compute(U_Mat_Type &U_candidate,
                             const U_Min_Matrix_Type &U_min_matrix,
                             const U_Max_Matrix_Type &U_max_matrix) {

    MinSaturateConditional<U_Mat_Type, U_Min_Matrix_Type, I, J_idx,
                           (U_Min_Matrix_Type::SparseAvailable_Type::lists
                                [I][J_idx])>::compute(U_candidate,
                                                      U_min_matrix);

    MaxSaturateConditional<U_Mat_Type, U_Max_Matrix_Type, I, J_idx,
                           (U_Max_Matrix_Type::SparseAvailable_Type::lists
                                [I][J_idx])>::compute(U_candidate,
                                                      U_max_matrix);

    Column<U_Mat_Type, U_Min_Matrix_Type, U_Max_Matrix_Type, M, N, I,
           (J_idx - 1)>::compute(U_candidate, U_min_matrix, U_max_matrix);
  }
};

// Column recursion termination for J_idx == 0
template <typename U_Mat_Type, typename U_Min_Matrix_Type,
          typename U_Max_Matrix_Type, std::size_t M, std::size_t N,
          std::size_t I>
struct Column<U_Mat_Type, U_Min_Matrix_Type, U_Max_Matrix_Type, M, N, I, 0> {
  static inline void compute(U_Mat_Type &U_candidate,
                             const U_Min_Matrix_Type &U_min_matrix,
                             const U_Max_Matrix_Type &U_max_matrix) {

    MinSaturateConditional<U_Mat_Type, U_Min_Matrix_Type, I, 0,
                           (U_Min_Matrix_Type::SparseAvailable_Type::lists
                                [I][0])>::compute(U_candidate, U_min_matrix);

    MaxSaturateConditional<U_Mat_Type, U_Max_Matrix_Type, I, 0,
                           (U_Max_Matrix_Type::SparseAvailable_Type::lists
                                [I][0])>::compute(U_candidate, U_max_matrix);
  }
};

// Row recursion for I (0..M-1), when I_idx > 0
template <typename U_Mat_Type, typename U_Min_Matrix_Type,
          typename U_Max_Matrix_Type, std::size_t M, std::size_t N,
          std::size_t I_idx>
struct Row {
  static inline void compute(U_Mat_Type &U_candidate,
                             const U_Min_Matrix_Type &U_min_matrix,
                             const U_Max_Matrix_Type &U_max_matrix) {
    Column<U_Mat_Type, U_Min_Matrix_Type, U_Max_Matrix_Type, M, N, I_idx,
           (N - 1)>::compute(U_candidate, U_min_matrix, U_max_matrix);

    Row<U_Mat_Type, U_Min_Matrix_Type, U_Max_Matrix_Type, M, N,
        (I_idx - 1)>::compute(U_candidate, U_min_matrix, U_max_matrix);
  }
};

// Row recursion termination for I_idx == 0
template <typename U_Mat_Type, typename U_Min_Matrix_Type,
          typename U_Max_Matrix_Type, std::size_t M, std::size_t N>
struct Row<U_Mat_Type, U_Min_Matrix_Type, U_Max_Matrix_Type, M, N, 0> {
  static inline void compute(U_Mat_Type &U_candidate,
                             const U_Min_Matrix_Type &U_min_matrix,
                             const U_Max_Matrix_Type &U_max_matrix) {
    Column<U_Mat_Type, U_Min_Matrix_Type, U_Max_Matrix_Type, M, N, 0,
           (N - 1)>::compute(U_candidate, U_min_matrix, U_max_matrix);
  }
};

} // namespace SaturateU_Horizon

/**
 * @brief Saturates the elements of the input matrix U_candidate within
 * specified minimum and maximum bounds.
 *
 * This function modifies the input matrix U_candidate in-place, ensuring that
 * each element is clamped between the corresponding elements of U_min_matrix
 * and U_max_matrix. The function is templated to support various matrix types
 * and sizes, and performs compile-time checks to ensure that the dimensions of
 * all matrices match.
 *
 * @tparam U_Mat_Type         Type of the candidate matrix to be saturated.
 * @tparam U_Min_Matrix_Type  Type of the matrix specifying minimum bounds.
 * @tparam U_Max_Matrix_Type  Type of the matrix specifying maximum bounds.
 *
 * @param U_candidate   Reference to the matrix whose elements will be
 * saturated.
 * @param U_min_matrix  Matrix specifying the minimum allowable values for each
 * element.
 * @param U_max_matrix  Matrix specifying the maximum allowable values for each
 * element.
 *
 * @note The function relies on the SaturateU_Horizon::Row helper for the actual
 * saturation logic.
 * @note All matrices must have matching dimensions, enforced via static
 * assertions.
 */
template <typename U_Mat_Type, typename U_Min_Matrix_Type,
          typename U_Max_Matrix_Type>
inline void saturate_U_horizon(U_Mat_Type &U_candidate,
                               const U_Min_Matrix_Type &U_min_matrix,
                               const U_Max_Matrix_Type &U_max_matrix) {

  constexpr std::size_t M = U_Mat_Type::COLS; // INPUT_SIZE
  constexpr std::size_t N = U_Mat_Type::ROWS; // NP

  static_assert(M > 0 && N > 0, "Matrix dimensions must be positive");
  static_assert(U_Min_Matrix_Type::COLS == M,
                "U_min_matrix COLS mismatch with U_candidate");
  static_assert(U_Min_Matrix_Type::ROWS == N,
                "U_min_matrix ROWS mismatch with U_candidate");
  static_assert(U_Max_Matrix_Type::COLS == M,
                "U_max_matrix COLS mismatch with U_candidate");
  static_assert(U_Max_Matrix_Type::ROWS == N,
                "U_max_matrix ROWS mismatch with U_candidate");

  SaturateU_Horizon::Row<U_Mat_Type, U_Min_Matrix_Type, U_Max_Matrix_Type, M, N,
                         (M - 1)>::compute(U_candidate, U_min_matrix,
                                           U_max_matrix);
}

} // namespace MatrixOperation

} // namespace PythonOptimization

#endif // __PYTHON_OPTIMIZATION_SQP_MATRIX_OPERATION_HPP__
