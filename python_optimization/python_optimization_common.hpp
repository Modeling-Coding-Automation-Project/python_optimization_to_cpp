#ifndef __PYTHON_OPTIMIZATION_COMMON_HPP__
#define __PYTHON_OPTIMIZATION_COMMON_HPP__

#include <functional>

namespace PythonOptimization {

namespace MatrixOperation {

namespace InnerProduct {

// Recursion when I_idx > 0
template <typename Vector_Type, std::size_t I_idx> struct Accumulate {
  /**
   * @brief Recursively accumulates the inner product contribution at index
   * I_idx.
   *
   * @tparam Vector_Type The type of the input vectors.
   * @tparam I_idx The current row index in the recursion.
   * @param a The first vector.
   * @param b The second vector.
   * @return The accumulated sum of products from index I_idx down to 0.
   */
  static auto compute(const Vector_Type &a, const Vector_Type &b) ->
      typename Vector_Type::Value_Type {
    return a.template get<I_idx, 0>() * b.template get<I_idx, 0>() +
           Accumulate<Vector_Type, I_idx - 1>::compute(a, b);
  }
};

// Recursion termination for I_idx == 0
template <typename Vector_Type> struct Accumulate<Vector_Type, 0> {
  /**
   * @brief Base case for the inner product accumulation.
   *
   * @tparam Vector_Type The type of the input vectors.
   * @param a The first vector.
   * @param b The second vector.
   * @return The product of the first elements of a and b.
   */
  static auto compute(const Vector_Type &a, const Vector_Type &b) ->
      typename Vector_Type::Value_Type {
    return a.template get<0, 0>() * b.template get<0, 0>();
  }
};

/**
 * @brief Compute inner product using template metaprogramming.
 *
 * This function computes the inner product of two vectors by recursively
 * accumulating the products of their corresponding elements via compile-time
 * loop unrolling.
 *
 * @tparam Vector_Type The type of the input vectors. Must define static member
 * ROWS and nested Value_Type.
 * @param a The first vector.
 * @param b The second vector.
 * @return The scalar inner product value.
 */
template <typename Vector_Type>
inline auto compute(const Vector_Type &a, const Vector_Type &b) ->
    typename Vector_Type::Value_Type {
  return Accumulate<Vector_Type, Vector_Type::ROWS - 1>::compute(a, b);
}

} // namespace InnerProduct

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
