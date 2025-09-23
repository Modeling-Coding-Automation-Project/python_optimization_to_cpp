#ifndef __PYTHON_OPTIMIZATION_ACTIVE_SET_HPP__
#define __PYTHON_OPTIMIZATION_ACTIVE_SET_HPP__

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
      std::array<std::array<std::size_t, NUMBER_OF_ROWS>, NUMBER_OF_COLUMNS>;
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
   * @brief Adds the (col,row) pair if not already active.
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
   * @brief Removes the (col,row) pair if currently active.
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
   * @brief Returns the (col,row) pair at active list index.
   * If index >= number_of_active, it is clamped to last (consistent with 1D
   * ActiveSet style (safe fallback) instead of throwing).
   */
  inline auto get_active(const std::size_t &index) const -> _Index_Pair_Type {

    std::size_t index_clamped = index;

    if (index_clamped >= this->_number_of_active) {
      index_clamped = this->_number_of_active - 1;
    }
    return this->_active_pairs[index_clamped];
  }

  /**
   * @brief Returns reference to active pairs list (including unused tail).
   */
  inline auto get_active_pairs(void) const -> const _Active_Pairs_Type & {
    return this->_active_pairs;
  }

  /**
   * @brief Returns number of active (col,row) elements.
   */
  inline auto get_number_of_active(void) const -> std::size_t {
    return this->_number_of_active;
  }

  /**
   * @brief Returns whether (col,row) is active.
   */
  inline auto is_active(const std::size_t &col, const std::size_t &row) const
      -> bool {

    std::size_t column_clamped = col;
    std::size_t row_clamped = row;

    this->_check_bounds(column_clamped, row_clamped);

    return this->_active_flags[column_clamped][row_clamped];
  }

  /**
   * @brief Clears all active elements.
   */
  inline void clear(void) {
    this->_active_flags.fill(false);

    for (std::size_t i = 0; i < this->_number_of_active; ++i) {
      this->_active_pairs[i][COLUMN] = 0;
      this->_active_pairs[i][ROW] = 0;
    }
    this->_number_of_active = 0;
  }

protected:
  /* Function */

  /**
   * @brief Clamp col,row to valid range if out-of-bounds.
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

/* Factory make */

template <std::size_t NumberOfColumns, std::size_t NumberOfRows>
inline auto make_ActiveSet2D(void)
    -> ActiveSet2D<NumberOfColumns, NumberOfRows> {
  return ActiveSet2D<NumberOfColumns, NumberOfRows>();
}

/* Alias */
template <std::size_t NumberOfColumns, std::size_t NumberOfRows>
using ActiveSet2D_Type = ActiveSet2D<NumberOfColumns, NumberOfRows>;

/**
 * @class ActiveSet2D_MatrixOperator
 * @brief Utility functions operating on matrices restricted by an ActiveSet2D.
 */
class ActiveSet2D_MatrixOperator {
public:
  /**
   * @brief Element-wise product over active positions only.
   * result(col,row) = A(col,row) * B(col,row) for active pairs else 0.
   */
  template <typename T, std::size_t M, std::size_t N, std::size_t MA,
            std::size_t NA>
  static inline auto element_wise_product(
      const PythonNumpy::Matrix<PythonNumpy::DefDense, T, M, N> &A,
      const PythonNumpy::Matrix<PythonNumpy::DefDense, T, M, N> &B,
      const ActiveSet2D<MA, NA> &active_set)
      -> PythonNumpy::Matrix<PythonNumpy::DefDense, T, M, N> {
    static_assert(M == MA && N == NA,
                  "Matrix size and ActiveSet2D size must match.");
    PythonNumpy::Matrix<PythonNumpy::DefDense, T, M, N> result;

    for (std::size_t idx = 0; idx < active_set.get_number_of_active(); ++idx) {
      auto pair = active_set.get_active(idx);
      const std::size_t col = pair[0];
      const std::size_t row = pair[1];
      result(col, row) = A(col, row) * B(col, row);
    }
    return result;
  }

  /**
   * @brief Sum of element-wise product (dot) over active positions.
   */
  template <typename T, std::size_t M, std::size_t N, std::size_t MA,
            std::size_t NA>
  static inline auto
  vdot(const PythonNumpy::Matrix<PythonNumpy::DefDense, T, M, N> &A,
       const PythonNumpy::Matrix<PythonNumpy::DefDense, T, M, N> &B,
       const ActiveSet2D<MA, NA> &active_set) -> T {
    static_assert(M == MA && N == NA,
                  "Matrix size and ActiveSet2D size must match.");
    T total = static_cast<T>(0);
    for (std::size_t idx = 0; idx < active_set.get_number_of_active(); ++idx) {
      auto pair = active_set.get_active(idx);
      total += A(pair[0], pair[1]) * B(pair[0], pair[1]);
    }
    return total;
  }

  /**
   * @brief Active positions multiplied by scalar, others zero.
   */
  template <typename T, std::size_t M, std::size_t N, std::size_t MA,
            std::size_t NA>
  static inline auto matrix_multiply_scalar(
      const PythonNumpy::Matrix<PythonNumpy::DefDense, T, M, N> &A,
      const T &scalar, const ActiveSet2D<MA, NA> &active_set)
      -> PythonNumpy::Matrix<PythonNumpy::DefDense, T, M, N> {
    static_assert(M == MA && N == NA,
                  "Matrix size and ActiveSet2D size must match.");
    PythonNumpy::Matrix<PythonNumpy::DefDense, T, M, N> result;
    for (std::size_t idx = 0; idx < active_set.get_number_of_active(); ++idx) {
      auto pair = active_set.get_active(idx);
      result(pair[0], pair[1]) = A(pair[0], pair[1]) * scalar;
    }
    return result;
  }

  /**
   * @brief Euclidean norm (2-norm) over active positions.
   */
  template <typename T, std::size_t M, std::size_t N, std::size_t MA,
            std::size_t NA>
  static inline auto
  norm(const PythonNumpy::Matrix<PythonNumpy::DefDense, T, M, N> &A,
       const ActiveSet2D<MA, NA> &active_set) -> T {
    static_assert(M == MA && N == NA,
                  "Matrix size and ActiveSet2D size must match.");
    T total = static_cast<T>(0);
    for (std::size_t idx = 0; idx < active_set.get_number_of_active(); ++idx) {
      auto pair = active_set.get_active(idx);
      const T v = A(pair[0], pair[1]);
      total += v * v;
    }
#ifdef __cpp_lib_math_special_functions
    return static_cast<T>(std::sqrt(total));
#else
    return static_cast<T>(std::sqrt(static_cast<double>(total)));
#endif
  }
};

} // namespace PythonOptimization

#endif // __PYTHON_OPTIMIZATION_ACTIVE_SET_HPP__
