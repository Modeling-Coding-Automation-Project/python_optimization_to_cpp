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

} // namespace PythonOptimization

#endif // __PYTHON_OPTIMIZATION_ACTIVE_SET_HPP__
