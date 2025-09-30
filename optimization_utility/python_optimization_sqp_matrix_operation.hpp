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
template <typename Matrix_Out_Type, typename Matrix_In_Type>
struct Row<Matrix_Out_Type, Matrix_In_Type, 0> {
  /**
   * @brief Base case assignment for I_idx == 0.
   */
  static inline void compute(Matrix_Out_Type &out_matrix,
                             const Matrix_In_Type &in_matrix,
                             const std::size_t &row_index) {

    out_matrix.access(0, row_index) = in_matrix.template get<0, 0>();
  }
};

template <typename Matrix_Out_Type, typename Matrix_In_Type>
inline void compute(Matrix_Out_Type &out_matrix,
                    const Matrix_In_Type &in_matrix,
                    const std::size_t &row_index) {
  static_assert(Matrix_In_Type::COLS > 0,
                "Matrix_In_Type::COLS must be positive");
  Row<Matrix_Out_Type, Matrix_In_Type, (Matrix_In_Type::COLS - 1)>::compute(
      out_matrix, in_matrix, row_index);
}

} // namespace SetRow

/**
 * @brief Sets a specific row of an output matrix from a given input row matrix.
 *
 * This function assigns the values from a single-row input matrix (`in_matrix`)
 * to the specified row (`row_index`) of the output matrix (`out_matrix`).
 *
 * @tparam Matrix_Out_Type Type of the output matrix. Must define a static
 * member `COLS`.
 * @tparam Matrix_In_Type Type of the input matrix. Must define static members
 * `COLS` and `ROWS`.
 * @param[out] out_matrix The matrix whose row will be set.
 * @param[in] in_matrix The single-row matrix providing the values.
 * @param[in] row_index The index of the row in `out_matrix` to be set.
 *
 * @note The function enforces at compile time that the number of columns in
 * both matrices match, and that the input matrix has exactly one row.
 * @throws static_assert if the column counts do not match or if the input
 * matrix is not a single row.
 */
template <typename Matrix_Out_Type, typename Matrix_In_Type>
inline void set_row(Matrix_Out_Type &out_matrix,
                    const Matrix_In_Type &in_matrix,
                    const std::size_t &row_index) {

  static_assert(Matrix_Out_Type::COLS == Matrix_In_Type::COLS,
                "Matrix_Out_Type::COLS != Matrix_In_Type::COLS");
  static_assert(Matrix_In_Type::ROWS == 1, "Matrix_In_Type::ROWS != 1");

  SetRow::compute(out_matrix, in_matrix, row_index);
}

namespace GetRow {

// Column recursion when I_idx > 0
template <typename Matrix_In_Type, typename Out_Type, std::size_t I_idx>
struct Column {
  /**
   * @brief Recursively copies elements from a given row into a column vector.
   *
   * out(I_idx, 0) = in_matrix(I_idx, row_index)
   * then recurses for the next lower column index.
   */
  static inline void compute(const Matrix_In_Type &in_matrix,
                             const std::size_t &row_index, Out_Type &out) {

    out.template set<I_idx, 0>(in_matrix.access(I_idx, row_index));

    Column<Matrix_In_Type, Out_Type, I_idx - 1>::compute(in_matrix, row_index,
                                                         out);
  }
};

// Column recursion termination for I_idx == 0
template <typename Matrix_In_Type, typename Out_Type>
struct Column<Matrix_In_Type, Out_Type, 0> {
  /**
   * @brief Base case: copy the element at column 0.
   */
  static inline void compute(const Matrix_In_Type &in_matrix,
                             const std::size_t &row_index, Out_Type &out) {

    out.template set<0, 0>(in_matrix.access(0, row_index));
  }
};

// Public wrapper to start the unrolled recursion
template <typename Matrix_In_Type, typename Out_Type>
inline void compute(const Matrix_In_Type &in_matrix,
                    const std::size_t &row_index, Out_Type &out) {

  static_assert(Out_Type::ROWS == 1, "Out_Type must be a (COLS x 1) vector");
  static_assert(Out_Type::COLS == Matrix_In_Type::COLS,
                "Output COLS must equal input COLS");

  Column<Matrix_In_Type, Out_Type, (Out_Type::COLS - 1)>::compute(
      in_matrix, row_index, out);
}

} // namespace GetRow

/**
 * @brief Extracts a specific row from the input matrix and returns it as a
 * dense matrix.
 *
 * @tparam Matrix_In_Type Type of the input matrix.
 * @param in_matrix The input matrix from which the row will be extracted.
 * @param row_index The index of the row to extract.
 * @return A dense matrix containing the specified row, with the same value type
 * and number of columns as the input matrix.
 *
 * This function uses the GetRow::compute method to perform the extraction.
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
template <typename Y_Mat_Type, typename Y_Min_Matrix_Type,
          typename Out_Penalty_Type, typename Active_Type, std::size_t I,
          std::size_t J_idx, bool limit_valid_flag>
struct MinConditional {};

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

template <typename Fxu_Type, typename dU_Type, typename Weight_Type,
          typename Out_Type, std::size_t OUTPUT_SIZE, std::size_t STATE_SIZE,
          std::size_t INPUT_SIZE, std::size_t I_idx, bool Activate>
struct Conditional {};

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

    const auto u = U_horizon_in.template get<I, J_idx>();
    const auto u_min = U_min_matrix.template get<I, J_idx>();
    const auto u_max = U_max_matrix.template get<I, J_idx>();

    if ((u >= (u_min - atol)) && (u <= (u_min + atol))) {
      at_lower.template set<I, J_idx>(true);
    }

    if ((u >= (u_max - atol)) && (u <= (u_max + atol))) {
      at_upper.template set<I, J_idx>(true);
    }

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

    const auto u = U_horizon_in.template get<I, 0>();
    const auto u_min = U_min_matrix.template get<I, 0>();
    const auto u_max = U_max_matrix.template get<I, 0>();

    if ((u >= (u_min - atol)) && (u <= (u_min + atol))) {
      at_lower.template set<I, 0>(true);
    }

    if ((u >= (u_max - atol)) && (u <= (u_max + atol))) {
      at_upper.template set<I, 0>(true);
    }
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

// Column recursion for J (0..N-1), when J_idx > 0
template <typename U_Mat_Type, typename U_Min_Matrix_Type,
          typename U_Max_Matrix_Type, std::size_t M, std::size_t N,
          std::size_t I, std::size_t J_idx>
struct Column {
  static inline void compute(U_Mat_Type &U_candidate,
                             const U_Min_Matrix_Type &U_min_matrix,
                             const U_Max_Matrix_Type &U_max_matrix) {

    const auto u_min = U_min_matrix.template get<I, J_idx>();
    const auto u_max = U_max_matrix.template get<I, J_idx>();
    const auto u_val = U_candidate.template get<I, J_idx>();

    if (u_val < u_min) {
      U_candidate.template set<I, J_idx>(u_min);
    } else if (u_val > u_max) {
      U_candidate.template set<I, J_idx>(u_max);
    }

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

    const auto u_min = U_min_matrix.template get<I, 0>();
    const auto u_max = U_max_matrix.template get<I, 0>();
    const auto u_val = U_candidate.template get<I, 0>();

    if (u_val < u_min) {
      U_candidate.template set<I, 0>(u_min);
    } else if (u_val > u_max) {
      U_candidate.template set<I, 0>(u_max);
    }
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
