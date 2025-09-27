#ifndef __PYTHON_OPTIMIZATION_SQP_MATRIX_OPERATION_HPP__
#define __PYTHON_OPTIMIZATION_SQP_MATRIX_OPERATION_HPP__

#include "python_control.hpp"
#include "python_numpy.hpp"

#include <array>
#include <functional>
#include <type_traits>

namespace PythonOptimization {

namespace MatrixOperation {

template <typename Matrix_Out_Type, typename Matrix_In_Type>
inline void set_row(Matrix_Out_Type &out_matrix,
                    const Matrix_In_Type &in_matrix,
                    const std::size_t &row_index) {

  static_assert(Matrix_Out_Type::COLS == Matrix_In_Type::COLS,
                "Matrix_Out_Type::COLS != Matrix_In_Type::COLS");
  static_assert(Matrix_In_Type::ROWS == 1, "Matrix_In_Type::ROWS != 1");

  for (std::size_t i = 0; i < Matrix_In_Type::ROWS; i++) {
    out_matrix(i, row_index) = in_matrix(i, 0);
  }
}

template <typename Matrix_In_Type>
inline auto get_row(const Matrix_In_Type &in_matrix,
                    const std::size_t &row_index)
    -> PythonNumpy::DenseMatrix_Type<typename Matrix_In_Type::Value_Type,
                                     Matrix_In_Type::COLS, 1> {

  PythonNumpy::DenseMatrix_Type<typename Matrix_In_Type::Value_Type,
                                Matrix_In_Type::COLS, 1>
      out;

  for (std::size_t i = 0; i < Matrix_In_Type::COLS; i++) {
    out(i, 0) = in_matrix(i, row_index);
  }

  return out;
}

template <typename X_Type, typename W_Type>
inline auto calculate_quadratic_form(const X_Type &X, const W_Type &W) ->
    typename X_Type::Value_Type {

  static_assert(W_Type::ROWS == X_Type::COLS, "W_Type::ROWS != X_Type::COLS");
  static_assert(X_Type::ROWS == 1, "X_Type::ROWS != 1");

  auto result = PythonNumpy::ATranspose_mul_B(X, W) * X;

  return result.template get<0, 0>();
}

template <typename X_Type>
inline auto calculate_quadratic_no_weighted(const X_Type &X) ->
    typename X_Type::Value_Type {

  static_assert(X_Type::ROWS == 1, "X_Type::ROWS != 1");

  auto result = PythonNumpy::ATranspose_mul_B(X, X);

  return result.template get<0, 0>();
}

/* calculate Y_limit penalty */

namespace CalculateY_LimitPenalty {

// Column recursion when J_idx > 0
template <typename Y_Mat_Type, typename Y_Min_Matrix_Type,
          typename Y_Max_Matrix_Type, typename Out_Type, std::size_t M,
          std::size_t N, std::size_t I, std::size_t J_idx>
struct Column {
  static inline void compute(const Y_Mat_Type &Y_horizon,
                             const Y_Min_Matrix_Type &Y_min_matrix,
                             const Y_Max_Matrix_Type &Y_max_matrix,
                             Out_Type &Y_limit_penalty) {

    const auto y = Y_horizon.template get<I, J_idx>();
    const auto y_min = Y_min_matrix.template get<I, J_idx>();
    const auto y_max = Y_max_matrix.template get<I, J_idx>();

    if (y < y_min) {
      Y_limit_penalty.template set<I, J_idx>(y - y_min);
    } else if (y > y_max) {
      Y_limit_penalty.template set<I, J_idx>(y - y_max);
    } else {
      /* Do Nothing */
    }

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
  static inline void compute(const Y_Mat_Type &Y_horizon,
                             const Y_Min_Matrix_Type &Y_min_matrix,
                             const Y_Max_Matrix_Type &Y_max_matrix,
                             Out_Type &Y_limit_penalty) {

    const auto y = Y_horizon.template get<I, 0>();
    const auto y_min = Y_min_matrix.template get<I, 0>();
    const auto y_max = Y_max_matrix.template get<I, 0>();

    if (y < y_min) {
      Y_limit_penalty.template set<I, 0>(y - y_min);
    } else if (y > y_max) {
      Y_limit_penalty.template set<I, 0>(y - y_max);
    } else {
      /* Do Nothing */
    }
  }
};

// Row recursion when I_idx > 0
template <typename Y_Mat_Type, typename Y_Min_Matrix_Type,
          typename Y_Max_Matrix_Type, typename Out_Type, std::size_t M,
          std::size_t N, std::size_t I_idx>
struct Row {
  static inline void compute(const Y_Mat_Type &Y_horizon,
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
  static inline void compute(const Y_Mat_Type &Y_horizon,
                             const Y_Min_Matrix_Type &Y_min_matrix,
                             const Y_Max_Matrix_Type &Y_max_matrix,
                             Out_Type &Y_limit_penalty) {
    Column<Y_Mat_Type, Y_Min_Matrix_Type, Y_Max_Matrix_Type, Out_Type, M, N, 0,
           (N - 1)>::compute(Y_horizon, Y_min_matrix, Y_max_matrix,
                             Y_limit_penalty);
  }
};

} // namespace CalculateY_LimitPenalty

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

// Column recursion when J_idx > 0
template <typename Y_Mat_Type, typename Y_Min_Matrix_Type,
          typename Y_Max_Matrix_Type, typename Out_Penalty_Type,
          typename Active_Type, std::size_t M, std::size_t N, std::size_t I,
          std::size_t J_idx>
struct Column {
  static inline void compute(const Y_Mat_Type &Y_horizon,
                             const Y_Min_Matrix_Type &Y_min_matrix,
                             const Y_Max_Matrix_Type &Y_max_matrix,
                             Out_Penalty_Type &Y_limit_penalty,
                             Active_Type &Y_limit_active) {

    const auto y = Y_horizon.template get<I, J_idx>();
    const auto y_min = Y_min_matrix.template get<I, J_idx>();
    const auto y_max = Y_max_matrix.template get<I, J_idx>();

    if (y < y_min) {
      Y_limit_penalty.template set<I, J_idx>(y - y_min);
      Y_limit_active.template set<I, J_idx>(
          static_cast<typename Active_Type::Value_Type>(1));
    } else if (y > y_max) {
      Y_limit_penalty.template set<I, J_idx>(y - y_max);
      Y_limit_active.template set<I, J_idx>(
          static_cast<typename Active_Type::Value_Type>(1));
    } else {
      /* Do Nothing */
    }

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
  static inline void compute(const Y_Mat_Type &Y_horizon,
                             const Y_Min_Matrix_Type &Y_min_matrix,
                             const Y_Max_Matrix_Type &Y_max_matrix,
                             Out_Penalty_Type &Y_limit_penalty,
                             Active_Type &Y_limit_active) {

    const auto y = Y_horizon.template get<I, 0>();
    const auto y_min = Y_min_matrix.template get<I, 0>();
    const auto y_max = Y_max_matrix.template get<I, 0>();

    if (y < y_min) {
      Y_limit_penalty.template set<I, 0>(y - y_min);
      Y_limit_active.template set<I, 0>(
          static_cast<typename Active_Type::Value_Type>(1));
    } else if (y > y_max) {
      Y_limit_penalty.template set<I, 0>(y - y_max);
      Y_limit_active.template set<I, 0>(
          static_cast<typename Active_Type::Value_Type>(1));
    } else {
      /* Do Nothing */
    }
  }
};

// Row recursion when I_idx > 0
template <typename Y_Mat_Type, typename Y_Min_Matrix_Type,
          typename Y_Max_Matrix_Type, typename Out_Penalty_Type,
          typename Active_Type, std::size_t M, std::size_t N, std::size_t I_idx>
struct Row {
  static inline void compute(const Y_Mat_Type &Y_horizon,
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
  static inline void compute(const Y_Mat_Type &Y_horizon,
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

} // namespace MatrixOperation

} // namespace PythonOptimization

#endif // __PYTHON_OPTIMIZATION_SQP_MATRIX_OPERATION_HPP__
