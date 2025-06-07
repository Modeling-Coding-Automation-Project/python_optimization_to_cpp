#include <iostream>

#include "python_numpy.hpp"
#include "python_optimization.hpp"

using namespace PythonNumpy;
using namespace PythonOptimization;

int main(void) {
  /* Define Problem */
  constexpr std::size_t NUMBER_OF_VARIABLES = 2;
  constexpr std::size_t NUMBER_OF_CONSTRAINTS = 4;

  auto qp_solver = make_QP_ActiveSetSolver<double, NUMBER_OF_VARIABLES,
                                           NUMBER_OF_CONSTRAINTS>();

  auto E_1 = make_DiagMatrixIdentity<double, NUMBER_OF_VARIABLES>();
  auto L_1 = make_DenseMatrix<NUMBER_OF_VARIABLES, 1>(5.0, 8.0);

  auto M_1 = make_DenseMatrix<NUMBER_OF_CONSTRAINTS, NUMBER_OF_VARIABLES>(
      1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, -1.0);

  auto Gamma_1 = make_DenseMatrix<NUMBER_OF_CONSTRAINTS, 1>(4.0, 6.0, 0.0, 0.0);

  /* Solve QP Problem */
  auto x_opt = qp_solver.solve(E_1, L_1, M_1, Gamma_1);

  /* Output Result */
  std::cout << "Optimal Solution:" << std::endl;
  for (std::size_t i = 0; i < NUMBER_OF_VARIABLES; ++i) {
    std::cout << "x[" << i << "] = " << x_opt(i, 0) << std::endl;
  }

  return 0;
}
