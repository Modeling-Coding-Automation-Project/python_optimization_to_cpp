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

namespace AbsoluteMaxScalarToMatrix {

// Recursion when I_idx > 0
template <typename Vector_Type, std::size_t I_idx> struct Element {
  /**
   * @brief Recursively computes h(I_idx, 0) = max(delta, epsilon * |u(I_idx,
   * 0)|) and recurses down to index 0.
   *
   * @tparam Vector_Type The column-vector type (ROWS x 1).
   * @tparam I_idx       Current row index in the recursion.
   * @param u       Source column vector.
   * @param delta   Minimum perturbation scalar.
   * @param epsilon Relative perturbation scalar.
   * @param h       Output column vector.
   */
  static inline void compute(const Vector_Type &u,
                             const typename Vector_Type::Value_Type &delta,
                             const typename Vector_Type::Value_Type &epsilon,
                             Vector_Type &h) {
    typename Vector_Type::Value_Type abs_ui = u.template get<I_idx, 0>();
    if (abs_ui < static_cast<typename Vector_Type::Value_Type>(0)) {
      abs_ui = -abs_ui;
    }
    typename Vector_Type::Value_Type val = epsilon * abs_ui;
    h.template set<I_idx, 0>((delta > val) ? delta : val);

    Element<Vector_Type, I_idx - 1>::compute(u, delta, epsilon, h);
  }
};

// Recursion termination for I_idx == 0
template <typename Vector_Type> struct Element<Vector_Type, 0> {
  /**
   * @brief Base case: computes h(0, 0) = max(delta, epsilon * |u(0, 0)|).
   *
   * @tparam Vector_Type The column-vector type (ROWS x 1).
   * @param u       Source column vector.
   * @param delta   Minimum perturbation scalar.
   * @param epsilon Relative perturbation scalar.
   * @param h       Output column vector.
   */
  static inline void compute(const Vector_Type &u,
                             const typename Vector_Type::Value_Type &delta,
                             const typename Vector_Type::Value_Type &epsilon,
                             Vector_Type &h) {
    typename Vector_Type::Value_Type abs_ui = u.template get<0, 0>();
    if (abs_ui < static_cast<typename Vector_Type::Value_Type>(0)) {
      abs_ui = -abs_ui;
    }
    typename Vector_Type::Value_Type val = epsilon * abs_ui;
    h.template set<0, 0>((delta > val) ? delta : val);
  }
};

/**
 * @brief Compute h(i, 0) = max(delta, epsilon * |u(i, 0)|) for all i via
 * compile-time loop unrolling.
 *
 * Equivalent runtime loop:
 * @code
 *   for (std::size_t i = 0; i < ROWS; ++i) {
 *     _T abs_ui = u(i, 0);
 *     if (abs_ui < 0) abs_ui = -abs_ui;
 *     _T val = epsilon * abs_ui;
 *     h(i, 0) = (delta > val) ? delta : val;
 *   }
 * @endcode
 *
 * @tparam Vector_Type The column-vector type (ROWS x 1). Must define ROWS,
 *         COLS, Value_Type, get<R,C>(), and set<R,C>().
 * @param u       Source column vector.
 * @param delta   Minimum perturbation value.
 * @param epsilon Relative perturbation coefficient.
 * @param h       Output column vector (overwritten).
 */
template <typename Vector_Type>
inline void
compute(const Vector_Type &u, const typename Vector_Type::Value_Type &delta,
        const typename Vector_Type::Value_Type &epsilon, Vector_Type &h) {
  static_assert(Vector_Type::COLS == 1,
                "Vector_Type must be a column vector (COLS == 1)");
  Element<Vector_Type, Vector_Type::ROWS - 1>::compute(u, delta, epsilon, h);
}

} // namespace AbsoluteMaxScalarToMatrix

} // namespace PythonOptimization

#endif // __PYTHON_OPTIMIZATION_COMMON_HPP__
