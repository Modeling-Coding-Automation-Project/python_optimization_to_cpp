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

} // namespace MatrixOperation

} // namespace PythonOptimization

#endif // __PYTHON_OPTIMIZATION_SQP_MATRIX_OPERATION_HPP__
