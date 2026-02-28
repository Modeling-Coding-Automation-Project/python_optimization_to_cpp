#ifndef __PYTHON_OPTIMIZATION_COMMON_HPP__
#define __PYTHON_OPTIMIZATION_COMMON_HPP__

#include <functional>

namespace PythonOptimization {

namespace MatrixOperation {

/**
 * @brief Adds a scalar value to each element of the given matrix.
 *
 * This function creates a copy of the input matrix and adds the specified
 * scalar to every element. The resulting matrix is returned.
 *
 * @tparam Matrix_Type Type of the matrix, which must define COLS, ROWS,
 * Value_Type, and operator().
 * @param matrix The input matrix to which the scalar will be added.
 * @param scalar The scalar value to add to each element of the matrix.
 * @return A new matrix with the scalar added to each element.
 */
template <typename Matrix_Type>
inline auto add_scalar_to_matrix(Matrix_Type &matrix,
                                 typename Matrix_Type::Value_Type scalar)
    -> Matrix_Type {

  Matrix_Type out = matrix;

  for (std::size_t i = 0; i < Matrix_Type::COLS; i++) {
    for (std::size_t j = 0; j < Matrix_Type::ROWS; j++) {
      out(i, j) += scalar;
    }
  }

  return out;
}

} // namespace MatrixOperation

/* Cost Function Objects */
template <typename X_Type, typename U_Horizon_Type>
using CostFunction_Object = std::function<typename X_Type::Value_Type(
    const X_Type &, const U_Horizon_Type &)>;

template <typename X_Type, typename U_Horizon_Type, typename Gradient_Type>
using CostAndGradientFunction_Object =
    std::function<void(const X_Type &, const U_Horizon_Type &,
                       typename X_Type::Value_Type &, Gradient_Type &)>;

} // namespace PythonOptimization

#endif // __PYTHON_OPTIMIZATION_COMMON_HPP__
