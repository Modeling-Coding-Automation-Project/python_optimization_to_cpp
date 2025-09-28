/**
 * @file python_optimization_active_set.hpp
 * @brief Provides classes and utilities for managing active sets of constraints
 *        in optimization problems, including 1D and 2D active set management.
 *
 * Features:
 * - Efficient push and removal of active constraints.
 * - Compile-time fixed sizes for safety and performance.
 * - Support for both 1D and 2D active sets.
 * - Matrix operations (element-wise product, dot, norm) restricted to active
 * set elements.
 */
#ifndef __PYTHON_OPTIMIZATION_ACTIVE_SET_HPP__
#define __PYTHON_OPTIMIZATION_ACTIVE_SET_HPP__

#include "python_math.hpp"
#include "python_numpy.hpp"

#include <array>
#include <functional>
#include <type_traits>

namespace PythonOptimization {

/**
 * @class ActiveSet
 * @brief A class that manages an active set of constraints with a
 * fixed number of constraints, ensuring safe management of the active set
 * information.
 *
 * @details
 * Attributes
 * ----------
 * - active_flags : std::array<bool, NumberOfConstraints>
 *     An array indicating whether each constraint is active (length: number of
 * constraints).
 * - active_indices : std::array<std::size_t, NumberOfConstraints>
 *     An array storing the indices of active constraints
 *     (length: number of constraints, unused parts are set to 0, etc.).
 * - number_of_active : std::size_t
 *     The current number of active constraints.
 */
template <std::size_t NumberOfConstraints> class ActiveSet {
protected:
  /* Type */
  using _Active_Flags_Type = std::array<bool, NumberOfConstraints>;
  using _Active_Indices_Type = std::array<std::size_t, NumberOfConstraints>;

public:
  /* Constructor */
  ActiveSet()
      : _active_flags{}, _active_indices{},
        _number_of_active(static_cast<std::size_t>(0)) {}

  /* Copy Constructor */
  ActiveSet(const ActiveSet<NumberOfConstraints> &input)
      : _active_flags(input._active_flags),
        _active_indices(input._active_indices),
        _number_of_active(input._number_of_active) {}

  ActiveSet<NumberOfConstraints> &
  operator=(const ActiveSet<NumberOfConstraints> &input) {
    if (this != &input) {
      this->_active_flags = input._active_flags;
      this->_active_indices = input._active_indices;
      this->_number_of_active = input._number_of_active;
    }
    return *this;
  }

  /* Move Constructor */
  ActiveSet(ActiveSet<NumberOfConstraints> &&input) noexcept
      : _active_flags(std::move(input._active_flags)),
        _active_indices(std::move(input._active_indices)),
        _number_of_active(input._number_of_active) {}

  ActiveSet<NumberOfConstraints> &
  operator=(ActiveSet<NumberOfConstraints> &&input) noexcept {
    if (this != &input) {
      this->_active_flags = std::move(input._active_flags);
      this->_active_indices = std::move(input._active_indices);
      this->_number_of_active = input._number_of_active;
    }
    return *this;
  }

public:
  /* Function */

  /**
   * @brief Adds the specified index to the set of active constraints if it is
   * not already active.
   *
   * This function checks if the constraint at the given index is currently
   * inactive. If so, it marks the constraint as active, adds its index to the
   * list of active indices, and increments the count of active constraints.
   *
   * @param index The index of the constraint to activate.
   */
  inline void push_active(std::size_t index) {
    if (!this->_active_flags[index]) {
      this->_active_flags[index] = true;
      this->_active_indices[this->_number_of_active] = index;

      this->_number_of_active++;
    }
  }

  /**
   * @brief Removes the specified index from the set of active constraints if it
   * is currently active.
   *
   * This function checks if the constraint at the given index is currently
   * active. If so, it marks the constraint as inactive, removes its index from
   * the list of active indices, and decrements the count of active constraints.
   *
   * @param index The index of the constraint to deactivate.
   */
  inline void push_inactive(std::size_t index) {
    if (this->_active_flags[index]) {
      this->_active_flags[index] = false;
      bool found = false;

      for (std::size_t i = 0; i < this->_number_of_active; ++i) {
        if (!found && this->_active_indices[i] == index) {
          found = true;
        }
        if (found && i < this->_number_of_active - 1) {
          this->_active_indices[i] = this->_active_indices[i + 1];
        }
      }
      if (found) {
        this->_active_indices[this->_number_of_active - 1] = 0;
        this->_number_of_active--;
      }
    }
  }

  /**
   * @brief Retrieves the index of the active constraint at the specified index.
   *
   * This function returns the index of the active constraint at the given
   * index. If the index is out of bounds, it returns the last active index.
   *
   * @param index The index of the active constraint to retrieve.
   * @return The index of the active constraint.
   */
  inline auto get_active(std::size_t index) const -> std::size_t {
    if (index >= this->_number_of_active) {
      index = this->_number_of_active - 1;
    }
    return this->_active_indices[index];
  }

  /**
   * @brief Retrieves the list of active indices.
   *
   * This function returns a reference to the array containing the indices of
   * the currently active constraints.
   *
   * @return A reference to the array of active indices.
   */
  inline auto get_active_indices() const -> _Active_Indices_Type & {
    return this->_active_indices;
  }

  /**
   * @brief Retrieves the number of currently active constraints.
   *
   * This function returns the count of active constraints in the active set.
   *
   * @return The number of active constraints.
   */
  inline auto get_number_of_active() const -> std::size_t {
    return this->_number_of_active;
  }

  /**
   * @brief Checks if the constraint at the specified index is currently active.
   *
   * This function checks the active flags to determine if the constraint at
   * the given index is active.
   *
   * @param index The index of the constraint to check.
   * @return True if the constraint is active, false otherwise.
   */
  inline auto is_active(std::size_t index) const -> bool {
    return this->_active_flags[index];
  }

public:
  /* Constant */
  static constexpr std::size_t NUMBER_OF_CONSTRAINTS = NumberOfConstraints;

protected:
  /* variables */
  _Active_Flags_Type _active_flags;
  _Active_Indices_Type _active_indices;
  std::size_t _number_of_active;
};

/* make Active Set */

/**
 * @brief Creates an instance of ActiveSet with the specified number of
 * constraints.
 *
 * @tparam NumberOfConstraints The number of constraints for the active set.
 * @return An instance of ActiveSet with the specified number of constraints.
 */
template <std::size_t NumberOfConstraints>
inline auto make_ActiveSet(void) -> ActiveSet<NumberOfConstraints> {
  return ActiveSet<NumberOfConstraints>();
}

/* Active Set Type */
template <std::size_t NumberOfConstraints>
using ActiveSet_Type = ActiveSet<NumberOfConstraints>;

namespace ActiveSet2D_Operation {

template <std::size_t M, std::size_t N>
using Active_Flags_Type = std::array<std::array<bool, N>, M>;

/**
 * @brief Recursively clears (sets to false) the active flags in the given flags
 * structure.
 *
 * This template struct defines a static method `run` that, for a given index
 * `I`, sets all elements of the `I`-th row of the `flags` object to `false`
 * using the `fill` method, and then recursively calls itself with `I - 1` until
 * the base case is reached.
 *
 * @tparam M Number of rows in the flags structure.
 * @tparam N Number of columns in the flags structure.
 * @tparam I Current row index to clear (should be non-negative).
 * @param flags Reference to the Active_Flags_Type<M, N> object whose flags are
 * to be cleared.
 */
template <std::size_t M, std::size_t N, std::size_t I> struct ClearLoop {

  static inline void run(Active_Flags_Type<M, N> &flags) {

    flags[I].fill(false);
    ClearLoop<M, N, I - 1>::run(flags);
  }
};

/**
 * @brief Specialization of the ClearLoop struct for the case when the loop
 * index is 0.
 *
 * This specialization provides a static inline function `run` that clears (sets
 * to false) all elements of the first row (index 0) of the `flags` object,
 * which is of type `Active_Flags_Type<M, N>`. This is typically used to reset
 * or clear the active set flags for the first row in an optimization algorithm.
 *
 * @tparam M Number of rows in the flags structure.
 * @tparam N Number of columns in the flags structure.
 * @param flags Reference to the flags object to be cleared.
 */
template <std::size_t M, std::size_t N> struct ClearLoop<M, N, 0> {

  static inline void run(Active_Flags_Type<M, N> &flags) {

    flags[0].fill(false);
  }
};

/**
 * @brief Clears all active flags in the provided active_flags structure.
 *
 * This function resets or clears all flags within the given Active_Flags_Type
 * instance by invoking the ClearLoop helper template. It is templated on the
 * number of columns and rows, allowing for compile-time optimization and type
 * safety.
 *
 * @tparam NUMBER_OF_COLUMNS The number of columns in the active flags
 * structure.
 * @tparam NUMBER_OF_ROWS The number of rows in the active flags structure.
 * @param active_flags Reference to the Active_Flags_Type object whose flags are
 * to be cleared.
 */
template <std::size_t NUMBER_OF_COLUMNS, std::size_t NUMBER_OF_ROWS>
inline void clear_all_active_flags(
    Active_Flags_Type<NUMBER_OF_COLUMNS, NUMBER_OF_ROWS> &active_flags) {

  ClearLoop<NUMBER_OF_COLUMNS, NUMBER_OF_ROWS, NUMBER_OF_COLUMNS - 1>::run(
      active_flags);
}

} // namespace ActiveSet2D_Operation

/**
 * @class ActiveSet2D
 * @brief Manages an active set of (col,row) element positions in a fixed size
 * matrix.
 *
 * @details
 * Template parameters define compile-time fixed numbers of columns and rows.
 * Similar to the 1D ActiveSet, this class stores:
 *  - _active_flags  : A 2D array (flattened) of bool indicating if (col,row) is
 * active.
 *  - _active_pairs  : An array of (col,row) index pairs for currently active
 * elements.
 *  - _number_of_active : Current count of active elements.
 *
 * Insertion/removal keeps the order of earlier active pairs (stable erase via
 * shift).
 */
template <std::size_t Number_Of_Columns, std::size_t Number_Of_Rows>
class ActiveSet2D {
public:
  /* Constant */
  static constexpr std::size_t NUMBER_OF_COLUMNS = Number_Of_Columns;
  static constexpr std::size_t NUMBER_OF_ROWS = Number_Of_Rows;
  static constexpr std::size_t NUMBER_OF_ELEMENTS =
      NUMBER_OF_COLUMNS * NUMBER_OF_ROWS;

  static constexpr std::size_t COLUMN = 0;
  static constexpr std::size_t ROW = 1;

protected:
  /* Type */
  using _Active_Flags_Type =
      std::array<std::array<bool, NUMBER_OF_ROWS>, NUMBER_OF_COLUMNS>;
  using _Index_Pair_Type = std::array<std::size_t, 2>;
  using _Active_Pairs_Type = std::array<_Index_Pair_Type, NUMBER_OF_ELEMENTS>;

public:
  /* Constructor */
  ActiveSet2D() : _active_flags{}, _active_pairs{}, _number_of_active(0) {}

  /* Copy Constructor */
  ActiveSet2D(const ActiveSet2D &input)
      : _active_flags(input._active_flags), _active_pairs(input._active_pairs),
        _number_of_active(input._number_of_active) {}

  ActiveSet2D &operator=(const ActiveSet2D &input) {
    if (this != &input) {
      this->_active_flags = input._active_flags;
      this->_active_pairs = input._active_pairs;
      this->_number_of_active = input._number_of_active;
    }
    return *this;
  }

  /* Move Constructor */
  ActiveSet2D(ActiveSet2D &&input) noexcept
      : _active_flags(std::move(input._active_flags)),
        _active_pairs(std::move(input._active_pairs)),
        _number_of_active(input._number_of_active) {}

  ActiveSet2D &operator=(ActiveSet2D &&input) noexcept {
    if (this != &input) {
      this->_active_flags = std::move(input._active_flags);
      this->_active_pairs = std::move(input._active_pairs);
      this->_number_of_active = input._number_of_active;
    }
    return *this;
  }

public:
  /* Function */

  /**
   * @brief Adds the specified (col, row) pair to the active set.
   *
   * This function is typically used in optimization algorithms that maintain
   * an active set of constraints or variables. The active set is updated by
   * pushing the given column and row indices, which represent the position
   * of the constraint or variable to be activated.
   *
   * @param col The column index to be added to the active set.
   * @param row The row index to be added to the active set.
   */
  inline void push_active(const std::size_t &col, const std::size_t &row) {

    std::size_t column_clamped = col;
    std::size_t row_clamped = row;

    this->_check_bounds(column_clamped, row_clamped);

    if (!this->_active_flags[column_clamped][row_clamped]) {
      this->_active_flags[column_clamped][row_clamped] = true;
      this->_active_pairs[this->_number_of_active][COLUMN] = column_clamped;
      this->_active_pairs[this->_number_of_active][ROW] = row_clamped;
      this->_number_of_active++;
    }
  }

  /**
   * @brief Marks the specified (column, row) pair as inactive and removes it
   * from the list of active pairs.
   *
   * This function first clamps the input column and row indices, checks their
   * bounds, and then checks if the specified pair is currently active. If so,
   * it marks the pair as inactive in the _active_flags matrix and removes it
   * from the _active_pairs array by shifting subsequent elements left to fill
   * the gap. The number of active pairs (_number_of_active) is decremented.
   *
   * @param col The column index of the pair to deactivate.
   * @param row The row index of the pair to deactivate.
   */
  inline void push_inactive(const std::size_t &col, const std::size_t &row) {

    std::size_t column_clamped = col;
    std::size_t row_clamped = row;

    this->_check_bounds(column_clamped, row_clamped);

    if (this->_active_flags[column_clamped][row_clamped]) {
      this->_active_flags[column_clamped][row_clamped] = false;

      bool found = false;
      for (std::size_t i = 0; i < this->_number_of_active; ++i) {
        if (!found && this->_active_pairs[i][COLUMN] == column_clamped &&
            this->_active_pairs[i][ROW] == row_clamped) {
          found = true;
        }
        if (found && i < this->_number_of_active - 1) {
          this->_active_pairs[i][COLUMN] = this->_active_pairs[i + 1][COLUMN];
          this->_active_pairs[i][ROW] = this->_active_pairs[i + 1][ROW];
        }
      }
      if (found) {
        this->_active_pairs[this->_number_of_active - 1][COLUMN] = 0;
        this->_active_pairs[this->_number_of_active - 1][ROW] = 0;
        this->_number_of_active--;
      }
    }
  }

  /**
   * @brief Retrieves the active pair at the specified index, clamping the index
   * if out of bounds.
   *
   * If the provided index is greater than or equal to the number of active
   * pairs, the function clamps the index to the last valid position. If there
   * are no active pairs, the index is set to 0. Returns the active pair at the
   * resulting index.
   *
   * @param index The index of the active pair to retrieve.
   * @return _Index_Pair_Type The active pair at the clamped index.
   */
  inline auto get_active(const std::size_t &index) const -> _Index_Pair_Type {

    std::size_t index_clamped = index;

    if (index_clamped >= this->_number_of_active) {
      if (0 == this->_number_of_active) {
        index_clamped = 0;
      } else {
        index_clamped = this->_number_of_active - 1;
      }
    }
    return this->_active_pairs[index_clamped];
  }

  /**
   * @brief Returns a constant reference to the current set of active pairs.
   *
   * This function provides access to the internal collection of active pairs,
   * which are typically used to represent constraints or relationships that are
   * currently active in the optimization process.
   *
   * @return A constant reference to the internal _Active_Pairs_Type container
   *         holding the active pairs.
   */
  inline auto get_active_pairs(void) const -> const _Active_Pairs_Type & {
    return this->_active_pairs;
  }

  /**
   * @brief Returns the current number of active elements.
   *
   * This function provides access to the private member variable that tracks
   * the number of active elements in the active set.
   *
   * @return The number of active elements as a std::size_t.
   */
  inline auto get_number_of_active(void) const -> std::size_t {
    return this->_number_of_active;
  }

  /**
   * @brief Checks if the specified element is active.
   *
   * This function determines whether the element at the given column and row
   * indices is marked as active. The indices are clamped to valid bounds before
   * checking. If the indices are out of range, an internal bounds check is
   * performed.
   *
   * @param col The column index of the element to check.
   * @param row The row index of the element to check.
   * @return true if the element is active; false otherwise.
   */
  inline auto is_active(const std::size_t &col, const std::size_t &row) const
      -> bool {

    std::size_t column_clamped = col;
    std::size_t row_clamped = row;

    this->_check_bounds(column_clamped, row_clamped);

    return this->_active_flags[column_clamped][row_clamped];
  }

  /**
   * @brief Clears the active set by resetting all active flags and active
   * pairs.
   *
   * This function performs the following actions:
   * - Calls ActiveSet2D_Operation::clear_all_active_flags to reset all active
   * flags.
   * - Iterates through the current active pairs and sets their COLUMN and ROW
   * values to 0.
   * - Resets the number of active pairs to zero.
   *
   * After calling this function, the active set will be empty and ready for
   * reuse.
   */
  inline void clear(void) {
    ActiveSet2D_Operation::clear_all_active_flags(this->_active_flags);

    for (std::size_t i = 0; i < this->_number_of_active; ++i) {
      this->_active_pairs[i][COLUMN] = 0;
      this->_active_pairs[i][ROW] = 0;
    }
    this->_number_of_active = 0;
  }

protected:
  /* Function */

  /**
   * @brief Ensures that the given column and row indices do not exceed their
   * respective bounds.
   *
   * This function checks if the provided column (`col`) and row (`row`) indices
   * are within the valid range, defined by `NUMBER_OF_COLUMNS` and
   * `NUMBER_OF_ROWS`. If an index is out of bounds (greater than or equal to
   * the maximum allowed value), it is set to the maximum valid index
   * (`NUMBER_OF_COLUMNS - 1` or `NUMBER_OF_ROWS - 1`).
   *
   * @param col Reference to the column index to check and potentially adjust.
   * @param row Reference to the row index to check and potentially adjust.
   */
  inline void _check_bounds(std::size_t &col, std::size_t &row) {

    if (NUMBER_OF_COLUMNS <= col) {
      col = NUMBER_OF_COLUMNS - 1;
    }
    if (NUMBER_OF_ROWS <= row) {
      row = NUMBER_OF_ROWS - 1;
    }
  }

protected:
  /* variables */
  _Active_Flags_Type _active_flags;
  _Active_Pairs_Type _active_pairs;
  std::size_t _number_of_active;
};

/* make Active Set 2D */

/**
 * @brief Creates and returns an instance of ActiveSet2D with specified
 * dimensions.
 *
 * This function template constructs an ActiveSet2D object with the given number
 * of columns and rows, as specified by the template parameters.
 *
 * @tparam Number_Of_Columns The number of columns for the ActiveSet2D.
 * @tparam Number_Of_Rows The number of rows for the ActiveSet2D.
 * @return An instance of ActiveSet2D<Number_Of_Columns, Number_Of_Rows>.
 */
template <std::size_t Number_Of_Columns, std::size_t Number_Of_Rows>
inline auto make_ActiveSet2D(void)
    -> ActiveSet2D<Number_Of_Columns, Number_Of_Rows> {
  return ActiveSet2D<Number_Of_Columns, Number_Of_Rows>();
}

/* Active Set 2D Type */

/**
 * @brief Alias for ActiveSet2D with specified dimensions.
 */
template <std::size_t Number_Of_Columns, std::size_t Number_Of_Rows>
using ActiveSet2D_Type = ActiveSet2D<Number_Of_Columns, Number_Of_Rows>;

/**
 * @class ActiveSet2D_MatrixOperator
 * @brief Provides static matrix operations restricted to active positions
 * defined by an ActiveSet2D.
 *
 * This class defines a set of static methods for performing element-wise and
 * scalar operations on matrices, but only over positions marked as "active" in
 * a corresponding ActiveSet2D. All operations ignore inactive positions, which
 * are left as zero in the result.
 *
 * @tparam T  Matrix element type.
 * @tparam M  Number of columns in the matrix.
 * @tparam N  Number of rows in the matrix.
 * @tparam MA Number of columns in the ActiveSet2D.
 * @tparam NA Number of rows in the ActiveSet2D.
 *
 * @section Methods
 * - element_wise_product: Computes the element-wise product of two matrices
 * over active positions.
 * - vdot: Computes the sum of element-wise products (dot product) over active
 * positions.
 * - matrix_multiply_scalar: Multiplies active positions of a matrix by a
 * scalar.
 * - norm: Computes the Euclidean (2-)norm over active positions.
 *
 * @note All methods require that the matrix and ActiveSet2D dimensions match.
 */
class ActiveSet2D_MatrixOperator {
public:
  /* Type */
  template <typename T, std::size_t M, std::size_t N>
  using Matrix_Type = PythonNumpy::DenseMatrix_Type<T, M, N>;

  /* Constant */
  static constexpr std::size_t COLUMN = 0;
  static constexpr std::size_t ROW = 1;

public:
  /* Function */

  /**
   * @brief Computes the element-wise product of two matrices at positions
   * specified by an active set.
   *
   * This static inline function multiplies corresponding elements of matrices A
   * and B, but only at the indices specified by the provided ActiveSet2D. The
   * result is a matrix of the same size as A and B, with non-active positions
   * left uninitialized or default-initialized.
   *
   * @tparam T   The type of the matrix elements.
   * @tparam M   Number of rows in the matrices.
   * @tparam N   Number of columns in the matrices.
   * @tparam MA  Number of rows in the active set (must match M).
   * @tparam NA  Number of columns in the active set (must match N).
   * @param A          The first input matrix.
   * @param B          The second input matrix.
   * @param active_set The set of active indices where the element-wise product
   * is computed.
   * @return Matrix_Type<T, M, N> A matrix containing the element-wise products
   * at active positions.
   *
   * @note The function asserts at compile time that the matrix and active set
   * dimensions match.
   */
  template <typename T, std::size_t M, std::size_t N, std::size_t MA,
            std::size_t NA>
  static inline auto element_wise_product(const Matrix_Type<T, M, N> &A,
                                          const Matrix_Type<T, M, N> &B,
                                          const ActiveSet2D<MA, NA> &active_set)
      -> Matrix_Type<T, M, N> {
    static_assert(M == MA && N == NA,
                  "Matrix size and ActiveSet2D size must match.");

    Matrix_Type<T, M, N> result;

    for (std::size_t idx = 0; idx < active_set.get_number_of_active(); ++idx) {
      auto pair = active_set.get_active(idx);

      result.access(pair[COLUMN], pair[ROW]) =
          A.access(pair[COLUMN], pair[ROW]) * B.access(pair[COLUMN], pair[ROW]);
    }
    return result;
  }

  /**
   * @brief Computes the dot product (vdot) of two matrices over an active set
   * of elements.
   *
   * This function calculates the sum of element-wise products of matrices A and
   * B, but only for the indices specified in the provided ActiveSet2D. The
   * active set defines which elements are included in the computation.
   *
   * @tparam T   The type of the matrix elements.
   * @tparam M   Number of rows in matrices A and B.
   * @tparam N   Number of columns in matrices A and B.
   * @tparam MA  Number of rows in the ActiveSet2D.
   * @tparam NA  Number of columns in the ActiveSet2D.
   * @param A          The first input matrix.
   * @param B          The second input matrix.
   * @param active_set The set of active indices to include in the dot product.
   * @return The sum of products of corresponding elements of A and B over the
   * active set.
   * @note   The matrix dimensions and active set dimensions must match.
   */
  template <typename T, std::size_t M, std::size_t N, std::size_t MA,
            std::size_t NA>
  static inline auto vdot(const Matrix_Type<T, M, N> &A,
                          const Matrix_Type<T, M, N> &B,
                          const ActiveSet2D<MA, NA> &active_set) -> T {
    static_assert(M == MA && N == NA,
                  "Matrix size and ActiveSet2D size must match.");

    T total = static_cast<T>(0);
    for (std::size_t idx = 0; idx < active_set.get_number_of_active(); ++idx) {
      auto pair = active_set.get_active(idx);
      total +=
          A.access(pair[COLUMN], pair[ROW]) * B.access(pair[COLUMN], pair[ROW]);
    }
    return total;
  }

  /**
   * @brief Multiplies elements of a matrix by a scalar, restricted to an active
   * set.
   *
   * This static inline function multiplies the elements of the input matrix `A`
   * by the given `scalar`, but only at the positions specified by the provided
   * `ActiveSet2D`. The result is a new matrix of the same size, where only the
   * elements in the active set are updated (others remain default-initialized).
   *
   * @tparam T   The type of the matrix elements.
   * @tparam M   Number of rows in the matrix.
   * @tparam N   Number of columns in the matrix.
   * @tparam MA  Number of rows in the active set.
   * @tparam NA  Number of columns in the active set.
   * @param A         The input matrix to be multiplied.
   * @param scalar    The scalar value to multiply with.
   * @param active_set The set of active positions to apply the multiplication.
   * @return Matrix_Type<T, M, N> A new matrix with updated values at active
   * positions.
   *
   * @note The function asserts at compile time that the matrix and active set
   * sizes match.
   */
  template <typename T, std::size_t M, std::size_t N, std::size_t MA,
            std::size_t NA>
  static inline auto
  matrix_multiply_scalar(const Matrix_Type<T, M, N> &A, const T &scalar,
                         const ActiveSet2D<MA, NA> &active_set)
      -> Matrix_Type<T, M, N> {
    static_assert(M == MA && N == NA,
                  "Matrix size and ActiveSet2D size must match.");

    Matrix_Type<T, M, N> result;
    for (std::size_t idx = 0; idx < active_set.get_number_of_active(); ++idx) {
      auto pair = active_set.get_active(idx);
      result.access(pair[COLUMN], pair[ROW]) =
          A.access(pair[COLUMN], pair[ROW]) * scalar;
    }
    return result;
  }

  /**
   * @brief Computes the Euclidean norm (L2 norm) of the elements in a matrix
   * specified by an active set.
   *
   * This static inline function calculates the square root of the sum of
   * squares of the matrix elements that are marked as active in the provided
   * ActiveSet2D. The function asserts at compile time that the dimensions of
   * the matrix and the active set match.
   *
   * @tparam T   The type of the matrix elements (e.g., float, double).
   * @tparam M   Number of rows in the matrix.
   * @tparam N   Number of columns in the matrix.
   * @tparam MA  Number of rows in the active set.
   * @tparam NA  Number of columns in the active set.
   * @param A           The matrix whose norm is to be computed.
   * @param active_set  The ActiveSet2D object specifying which elements are
   * active.
   * @return T          The Euclidean norm of the active elements in the matrix.
   */
  template <typename T, std::size_t M, std::size_t N, std::size_t MA,
            std::size_t NA>
  static inline auto norm(const Matrix_Type<T, M, N> &A,
                          const ActiveSet2D<MA, NA> &active_set) -> T {
    static_assert(M == MA && N == NA,
                  "Matrix size and ActiveSet2D size must match.");

    T total = static_cast<T>(0);
    for (std::size_t idx = 0; idx < active_set.get_number_of_active(); ++idx) {
      auto pair = active_set.get_active(idx);
      const T v = A.access(pair[COLUMN], pair[ROW]);
      total += v * v;
    }

    return PythonMath::sqrt(total);
  }
};

} // namespace PythonOptimization

#endif // __PYTHON_OPTIMIZATION_ACTIVE_SET_HPP__
