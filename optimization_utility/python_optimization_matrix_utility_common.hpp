/**
 * @file python_optimization_matrix_utility_common.hpp
 *
 * @brief Common type definitions for matrix utilities in Python optimization.
 * This header defines common type aliases for function objects related to state
 * space models, including state equations, measurement equations, and their
 * Jacobians and Hessians. These type aliases are used across various
 * optimization modules to ensure consistency and readability when working with
 * state space models in the context of optimization problems.
 */
#ifndef PYTHON_OPTIMIZATION_MATRIX_UTILITY_COMMON_HPP_
#define PYTHON_OPTIMIZATION_MATRIX_UTILITY_COMMON_HPP_

#include <functional>

namespace PythonOptimization {

/* State Space Function Objects */

template <typename State_Type, typename Input_Type, typename Parameter_Type>
using StateEquation_Object = std::function<State_Type(
    const State_Type &, const Input_Type &, const Parameter_Type &)>;

template <typename Output_Type, typename State_Type, typename Input_Type,
          typename Parameter_Type>
using MeasurementEquation_Object = std::function<Output_Type(
    const State_Type &, const Input_Type &, const Parameter_Type &)>;

template <typename State_Jacobian_X_Matrix_Type, typename State_Type,
          typename Input_Type, typename Parameter_Type>
using StateEquationJacobian_X_Object =
    std::function<State_Jacobian_X_Matrix_Type(
        const State_Type &, const Input_Type &, const Parameter_Type &)>;

template <typename State_Jacobian_U_Matrix_Type, typename State_Type,
          typename Input_Type, typename Parameter_Type>
using StateEquationJacobian_U_Object =
    std::function<State_Jacobian_U_Matrix_Type(
        const State_Type &, const Input_Type &, const Parameter_Type &)>;

template <typename Measurement_Jacobian_X_Matrix_Type, typename State_Type,
          typename Input_Type, typename Parameter_Type>
using MeasurementEquationJacobian_X_Object =
    std::function<Measurement_Jacobian_X_Matrix_Type(
        const State_Type &, const Input_Type &, const Parameter_Type &)>;

template <typename State_Hessian_XX_Matrix_Type, typename State_Type,
          typename Input_Type, typename Parameter_Type>
using StateEquationHessian_XX_Object =
    std::function<State_Hessian_XX_Matrix_Type(
        const State_Type &, const Input_Type &, const Parameter_Type &)>;

template <typename State_Hessian_XU_Matrix_Type, typename State_Type,
          typename Input_Type, typename Parameter_Type>
using StateEquationHessian_XU_Object =
    std::function<State_Hessian_XU_Matrix_Type(
        const State_Type &, const Input_Type &, const Parameter_Type &)>;

template <typename State_Hessian_UX_Matrix_Type, typename State_Type,
          typename Input_Type, typename Parameter_Type>
using StateEquationHessian_UX_Object =
    std::function<State_Hessian_UX_Matrix_Type(
        const State_Type &, const Input_Type &, const Parameter_Type &)>;

template <typename State_Hessian_UU_Matrix_Type, typename State_Type,
          typename Input_Type, typename Parameter_Type>
using StateEquationHessian_UU_Object =
    std::function<State_Hessian_UU_Matrix_Type(
        const State_Type &, const Input_Type &, const Parameter_Type &)>;

template <typename Measurement_Hessian_XX_Matrix_Type, typename State_Type,
          typename Input_Type, typename Parameter_Type>
using MeasurementEquationHessian_XX_Object =
    std::function<Measurement_Hessian_XX_Matrix_Type(
        const State_Type &, const Input_Type &, const Parameter_Type &)>;

} // namespace PythonOptimization

#endif // PYTHON_OPTIMIZATION_MATRIX_UTILITY_COMMON_HPP_
