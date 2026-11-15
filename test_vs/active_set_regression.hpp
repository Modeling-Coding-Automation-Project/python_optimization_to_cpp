#ifndef TEST_ACTIVE_SET_REGRESSION_HPP_
#define TEST_ACTIVE_SET_REGRESSION_HPP_

#include "python_optimization_qp_active_set.hpp"

#include <cmath>
#include <stdexcept>

namespace ActiveSetRegression {

inline void require(bool condition, const char *message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

template <typename Operation>
void expect_out_of_range(Operation operation) {
  bool caught = false;
  try {
    operation();
  } catch (const std::out_of_range &) {
    caught = true;
  }
  require(caught, "Expected an invalid active-set index to be rejected");
}

inline void test_positions_and_constraint_numbers() {
  PythonOptimization::ActiveSet<6> active;
  for (const auto constraint : {5u, 2u, 4u, 1u}) {
    active.push_active(constraint);
  }
  active.push_active(2);
  active.deactivate_at(1);
  require(active.get_number_of_active() == 3 && active.get_active(0) == 5 &&
              active.get_active(1) == 4 && active.get_active(2) == 1 &&
              !active.is_active(2),
          "Position removal must preserve order and deactivate the mapped ID");
  active.push_inactive(4);
  active.push_inactive(0);
  require(active.get_number_of_active() == 2 && active.get_active(1) == 1,
          "push_inactive must continue to accept an original constraint number");
  active.deactivate_at(0);
  require(active.get_active(0) == 1, "Removing the first position changed order");
  active.deactivate_at(0);
  for (const auto value : active.get_active_indices()) {
    require(value == 0, "Unused active positions must be cleared");
  }
  expect_out_of_range([&]() { active.get_active(0); });
  expect_out_of_range([&]() { active.deactivate_at(0); });
  active.push_active(2);
  expect_out_of_range([&]() { active.get_active(1); });
  expect_out_of_range([&]() { active.deactivate_at(1); });
  expect_out_of_range([&]() { active.deactivate_at(std::size_t(-1)); });
  expect_out_of_range([&]() { active.push_active(6); });
  expect_out_of_range([&]() { active.push_inactive(6); });
  expect_out_of_range([&]() { active.is_active(6); });
  require(active.get_number_of_active() == 1 && active.get_active(0) == 2,
          "Invalid operations must leave the set unchanged");
  PythonOptimization::ActiveSet<0> empty;
  expect_out_of_range([&]() { empty.get_active(0); });
}

inline void test_2d_positions_and_coordinates() {
  PythonOptimization::ActiveSet2D<3, 3> active;
  expect_out_of_range([&]() { active.get_active(0); });
  expect_out_of_range([&]() { active.deactivate_at(0); });
  active.push_active(2, 1);
  active.push_active(0, 2);
  active.push_active(1, 0);
  active.push_active(2, 2);
  active.push_active(0, 2);
  active.deactivate_at(1);
  const auto &view = active;
  require(active.get_number_of_active() == 3 && !view.is_active(0, 2) &&
              view.get_active(0)[0] == 2 && view.get_active(1)[0] == 1 &&
              view.get_active(2)[1] == 2,
          "2D position removal must preserve pairs, flags and order");
  active.push_inactive(1, 0);
  active.push_inactive(0, 0);
  active.deactivate_at(0);
  require(active.get_active(0)[0] == 2 && active.get_active(0)[1] == 2,
          "2D coordinate removal removed the wrong pair");
  active.deactivate_at(0);
  for (const auto &pair : active.get_active_pairs()) {
    require(pair[0] == 0 && pair[1] == 0, "Unused 2D positions must be cleared");
  }
  active.push_active(2, 1);
  expect_out_of_range([&]() { active.get_active(1); });
  expect_out_of_range([&]() { active.deactivate_at(1); });
  expect_out_of_range([&]() { active.push_active(3, 0); });
  expect_out_of_range([&]() { active.push_inactive(0, 3); });
  expect_out_of_range([&]() { view.is_active(std::size_t(-1), 0); });
  require(active.get_number_of_active() == 1 && view.is_active(2, 1),
          "Invalid 2D operations must not clamp to a different pair");
  active.clear();
  expect_out_of_range([&]() { active.get_active(0); });
}

template <typename T> void test_qp_constraint_release() {
  using namespace PythonNumpy;
  using namespace PythonOptimization;
  const T tolerance = static_cast<T>(1e-4);

  // Position zero refers to constraint 2, an unnecessary upper bound x <= 1.
  QP_ActiveSetSolver_Type<T, 1, 3> scalar;
  scalar.active_set.push_active(2);
  scalar.set_max_iteration(10);
  scalar.set_tol(static_cast<T>(1e-5));
  DenseMatrix_Type<T, 1, 1> scalar_E;
  scalar_E(0, 0) = T(1);
  auto x = scalar.solve(scalar_E,
                        make_DenseMatrix<1, 1>(T(0)),
                        make_DenseMatrix<3, 1>(T(-1), T(1), T(1)),
                        make_DenseMatrix<3, 1>(T(1), T(3), T(1)));
  require(std::abs(x(0, 0)) < tolerance,
          "QP must release constraint 2 using active position zero");
  require(scalar.get_iteration_count() < scalar.get_max_iteration() &&
              scalar.active_set.get_number_of_active() == 0,
          "QP must finish after releasing the unnecessary bound");

  // Only the second active constraint is unnecessary; retain x <= 1.
  QP_ActiveSetSolver_Type<T, 2, 4> pair;
  pair.active_set.push_active(2);
  pair.active_set.push_active(3);
  pair.set_max_iteration(10);
  pair.set_tol(static_cast<T>(1e-5));
  DenseMatrix_Type<T, 2, 2> E;
  E(0, 0) = T(2);
  E(1, 1) = T(3);
  auto xy = pair.solve(
      E, make_DenseMatrix<2, 1>(T(4), T(1.5)),
      make_DenseMatrix<4, 2>(T(-1), T(0), T(0), T(-1), T(1), T(0), T(0), T(1)),
      make_DenseMatrix<4, 1>(T(0), T(0), T(1), T(1)));
  require(std::abs(xy(0, 0) - T(1)) < tolerance &&
              std::abs(xy(1, 0) - T(0.5)) < tolerance &&
              pair.get_iteration_count() < pair.get_max_iteration(),
          "QP must remove the second bound and re-solve with the first bound");

  auto lambda = make_DenseMatrix<4, 1>(T(1), T(-100), T(-200), T(-300));
  std::size_t position = 0;
  require(!QP_ActiveSetSolverOperation::find_negative_lambda(
              lambda, 1, tolerance, position),
          "Inactive multiplier storage must not affect constraint release");
  require(!QP_ActiveSetSolverOperation::find_negative_lambda(
              lambda, 0, tolerance, position),
          "An empty active prefix must not contain a negative multiplier");
  require(QP_ActiveSetSolverOperation::find_negative_lambda(
              lambda, 2, tolerance, position) && position == 1,
          "Negative multiplier search must return an active-set position");
}

inline void run() {
  test_positions_and_constraint_numbers();
  test_2d_positions_and_coordinates();
  test_qp_constraint_release<double>();
  test_qp_constraint_release<float>();
}

} // namespace ActiveSetRegression
#endif
