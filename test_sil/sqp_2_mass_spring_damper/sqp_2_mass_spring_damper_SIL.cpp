#include "sqp_2_mass_spring_damper_SIL_wrapper.hpp"

#include "sqp_2_mass_spring_damper_demo_SIL_parameter.hpp"

#include "python_numpy.hpp"
#include "python_optimization.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

using namespace PythonNumpy;
using namespace PythonOptimization;

using FLOAT = typename sqp_2_mass_spring_damper_SIL_wrapper::type::Value_Type;

using X_Type = sqp_2_mass_spring_damper_SIL_wrapper::X_Type;

constexpr std::size_t STATE_SIZE =
    sqp_2_mass_spring_damper_SIL_wrapper::STATE_SIZE;
constexpr std::size_t INPUT_SIZE =
    sqp_2_mass_spring_damper_SIL_wrapper::INPUT_SIZE;
constexpr std::size_t NP = sqp_2_mass_spring_damper_SIL_wrapper::NP;

using Reference_Trajectory_Type =
    sqp_2_mass_spring_damper_SIL_wrapper::Reference_Trajectory_Type;

using Cost_Matrices_Type = sqp_2_mass_spring_damper_SIL_wrapper::type;

using U_horizon_Type = DenseMatrix_Type<double, INPUT_SIZE, NP>;
using Gradient_Type = U_horizon_Type;
using V_Horizon_Type = U_horizon_Type;
using HVP_Type = U_horizon_Type;

py::array_t<FLOAT> solve(void) {

  auto cost_matrices = sqp_2_mass_spring_damper_SIL_wrapper::make();

  /* Define functions for solver */
  CostFunction_Object<X_Type, U_horizon_Type> cost_function =
      [&cost_matrices](const X_Type &X, const U_horizon_Type &U) ->
      typename X_Type::Value_Type { return cost_matrices.compute_cost(X, U); };
  CostAndGradientFunction_Object<X_Type, U_horizon_Type, Gradient_Type>
      cost_and_gradient_function =
          [&cost_matrices](const X_Type &X, const U_horizon_Type &U,
                           typename X_Type::Value_Type &J,
                           Gradient_Type &gradient) {
            cost_matrices.compute_cost_and_gradient(X, U, J, gradient);
          };
  HVP_Function_Object<X_Type, U_horizon_Type, V_Horizon_Type, HVP_Type>
      hvp_function = [&cost_matrices](const X_Type &X, const U_horizon_Type &U,
                                      const V_Horizon_Type &V) -> HVP_Type {
    return cost_matrices.hvp_analytic(X, U, V);
  };

  /* Initial variables */
  Reference_Trajectory_Type reference_trajectory;

  DenseMatrix_Type<double, INPUT_SIZE, NP> U_horizon_initial;

  /* SQP Active Set PCG PLS */
  auto solver = make_SQP_ActiveSet_PCG_PLS<Cost_Matrices_Type>();

  /* solve */
  solver.set_solver_max_iteration(30);

  auto X_initial = make_DenseMatrix<STATE_SIZE, 1>(
      static_cast<double>(3.141592653589793 / 4.0), static_cast<double>(0));

  auto U_horizon_opt =
      solver.solve(U_horizon_initial, cost_and_gradient_function, cost_function,
                   hvp_function, X_initial, cost_matrices.get_U_min_matrix(),
                   cost_matrices.get_U_max_matrix());

  /* output U */
  py::array_t<FLOAT> result;
  result.resize({static_cast<int>(INPUT_SIZE), static_cast<int>(NP)});

  for (std::size_t i = 0; i < INPUT_SIZE; ++i) {
    for (std::size_t j = 0; j < NP; ++j) {
      result.mutable_at(i, j) = U_horizon_opt.access(i, j);
    }
  }

  return result;
}

PYBIND11_MODULE(Sqp2MassSpringDamperSIL, m) {
  m.def("solve", &solve, "solve SQP");
}
