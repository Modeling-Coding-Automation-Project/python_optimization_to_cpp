#include <iostream>

#include "python_numpy.hpp"
#include "python_optimization.hpp"

using namespace PythonNumpy;
using namespace PythonOptimization;

int main(void) {
  /* Define Problem */
  constexpr std::size_t NUMBER_OF_VARIABLES = 3;
  constexpr std::size_t NUMBER_OF_CONSTRAINTS = 3;

  auto qp_solver = make_QP_ActiveSetSolver<double, NUMBER_OF_VARIABLES,
                                           NUMBER_OF_CONSTRAINTS>();

  qp_solver.set_max_iteration(10);
  qp_solver.set_tol(1.0e-8);
  qp_solver.set_kkt_inv_solver_division_min(1.0e-5);

  auto E = make_DiagMatrixIdentity<double, NUMBER_OF_VARIABLES>();
  auto L = make_DenseMatrix<NUMBER_OF_VARIABLES, 1>(2.0, 3.0, 1.0);

  auto M = make_DenseMatrix<NUMBER_OF_CONSTRAINTS, NUMBER_OF_VARIABLES>(
      1.0, 1.0, 1.0, 3.0, -2.0, -3.0, 1.0, -3.0, 2.0);

  auto Gamma = make_DenseMatrix<NUMBER_OF_CONSTRAINTS, 1>(1.0, 1.0, 1.0);

  /* Solve QP Problem */
  auto x_opt = qp_solver.solve(E, L, M, Gamma);

  /* Output Result */
  std::cout << "Optimal Solution:" << std::endl;
  for (std::size_t i = 0; i < NUMBER_OF_VARIABLES; ++i) {
    std::cout << "x[" << i << "] = " << x_opt(i, 0) << std::endl;
  }

  return 0;
}
