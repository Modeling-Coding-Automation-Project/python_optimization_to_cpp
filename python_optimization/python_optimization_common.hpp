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

namespace AbsoluteMaxScalarToMatrix {

// Column recursion when J_idx > 0
template <typename Matrix_Type, std::size_t M, std::size_t N, std::size_t I,
          std::size_t J_idx>
struct Column {
  /**
   * @brief Recursively computes h(I, J_idx) = max(delta, epsilon * |u(I,
   * J_idx)|) and recurses down to J_idx == 0.
   *
   * @tparam Matrix_Type The matrix type (ROWS x COLS).
   * @tparam M           Number of rows in the matrix.
   * @tparam N           Number of columns in the matrix.
   * @tparam I           Current row index (fixed for this Column recursion).
   * @tparam J_idx       Current column index in the recursion.
   * @param U       Source matrix.
   * @param delta   Minimum perturbation scalar.
   * @param epsilon Relative perturbation scalar.
   * @param H       Output matrix.
   */
  static inline void compute(const Matrix_Type &U,
                             const typename Matrix_Type::Value_Type &delta,
                             const typename Matrix_Type::Value_Type &epsilon,
                             Matrix_Type &H) {

    typename Matrix_Type::Value_Type abs_ui =
        Base::Math::abs(U.template get<I, J_idx>());

    typename Matrix_Type::Value_Type val = epsilon * abs_ui;
    H.template set<I, J_idx>((delta > val) ? delta : val);

    Column<Matrix_Type, M, N, I, J_idx - 1>::compute(U, delta, epsilon, H);
  }
};

// Column recursion termination for J_idx == 0
template <typename Matrix_Type, std::size_t M, std::size_t N, std::size_t I>
struct Column<Matrix_Type, M, N, I, 0> {
  /**
   * @brief Base case: computes H(I, 0) = max(delta, epsilon * |U(I, 0)|).
   *
   * @tparam Matrix_Type The matrix type (ROWS x COLS).
   * @tparam M           Number of rows in the matrix.
   * @tparam N           Number of columns in the matrix.
   * @tparam I           Current row index.
   * @param U       Source matrix.
   * @param delta   Minimum perturbation scalar.
   * @param epsilon Relative perturbation scalar.
   * @param H       Output matrix.
   */
  static inline void compute(const Matrix_Type &U,
                             const typename Matrix_Type::Value_Type &delta,
                             const typename Matrix_Type::Value_Type &epsilon,
                             Matrix_Type &H) {
    typename Matrix_Type::Value_Type abs_ui =
        Base::Math::abs(U.template get<I, 0>());

    typename Matrix_Type::Value_Type val = epsilon * abs_ui;
    H.template set<I, 0>((delta > val) ? delta : val);
  }
};

// Row recursion when I_idx > 0
template <typename Matrix_Type, std::size_t M, std::size_t N, std::size_t I_idx>
struct Row {
  /**
   * @brief Processes all columns of row I_idx, then recurses to row I_idx - 1.
   *
   * @tparam Matrix_Type The matrix type (ROWS x COLS).
   * @tparam M           Number of rows in the matrix.
   * @tparam N           Number of columns in the matrix.
   * @tparam I_idx       Current row index in the recursion.
   * @param U       Source matrix.
   * @param delta   Minimum perturbation scalar.
   * @param epsilon Relative perturbation scalar.
   * @param H       Output matrix.
   */
  static inline void compute(const Matrix_Type &U,
                             const typename Matrix_Type::Value_Type &delta,
                             const typename Matrix_Type::Value_Type &epsilon,
                             Matrix_Type &H) {
    Column<Matrix_Type, M, N, I_idx, N - 1>::compute(U, delta, epsilon, H);
    Row<Matrix_Type, M, N, I_idx - 1>::compute(U, delta, epsilon, H);
  }
};

// Row recursion termination for I_idx == 0
template <typename Matrix_Type, std::size_t M, std::size_t N>
struct Row<Matrix_Type, M, N, 0> {
  /**
   * @brief Base case: processes all columns of row 0.
   *
   * @tparam Matrix_Type The matrix type (ROWS x COLS).
   * @tparam M           Number of rows in the matrix.
   * @tparam N           Number of columns in the matrix.
   * @param U       Source matrix.
   * @param delta   Minimum perturbation scalar.
   * @param epsilon Relative perturbation scalar.
   * @param H       Output matrix.
   */
  static inline void compute(const Matrix_Type &U,
                             const typename Matrix_Type::Value_Type &delta,
                             const typename Matrix_Type::Value_Type &epsilon,
                             Matrix_Type &H) {
    Column<Matrix_Type, M, N, 0, N - 1>::compute(U, delta, epsilon, H);
  }
};

/**
 * @brief Compute h(i, j) = max(delta, epsilon * |u(i, j)|) for all (i, j) via
 * compile-time loop unrolling.
 *
 * Equivalent runtime loop:
 * @code
 *   for (std::size_t i = 0; i < ROWS; ++i) {
 *     for (std::size_t j = 0; j < COLS; ++j) {
 *       _T abs_ui = u(i, j);
 *       if (abs_ui < 0) abs_ui = -abs_ui;
 *       _T val = epsilon * abs_ui;
 *       h(i, j) = (delta > val) ? delta : val;
 *     }
 *   }
 * @endcode
 *
 * @tparam Matrix_Type The matrix type (ROWS x COLS). Must define ROWS, COLS,
 *         Value_Type, get<R,C>(), and set<R,C>().
 * @param U       Source matrix.
 * @param delta   Minimum perturbation value.
 * @param epsilon Relative perturbation coefficient.
 * @param H       Output matrix (overwritten).
 */
template <typename Matrix_Type>
inline void
compute(const Matrix_Type &U, const typename Matrix_Type::Value_Type &delta,
        const typename Matrix_Type::Value_Type &epsilon, Matrix_Type &H) {
  Row<Matrix_Type, Matrix_Type::COLS, Matrix_Type::ROWS,
      Matrix_Type::COLS - 1>::compute(U, delta, epsilon, H);
}

} // namespace AbsoluteMaxScalarToMatrix

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
