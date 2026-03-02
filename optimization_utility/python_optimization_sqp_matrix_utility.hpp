/**
 * @file python_optimization_sqp_matrix_utility.hpp
 *
 * @brief Utility definitions for SQP cost matrices in Nonlinear MPC (NMPC)
 * optimization.
 *
 * This header provides type definitions, function objects, and the main class
 * for handling cost matrices and related operations in Sequential Quadratic
 * Programming (SQP) for Nonlinear Model Predictive Control (NMPC) problems.
 */
#ifndef __PYTHON_OPTIMIZATION_SQP_MATRIX_UTILITY_HPP__
#define __PYTHON_OPTIMIZATION_SQP_MATRIX_UTILITY_HPP__

#include "python_optimization_common_matrix_operation.hpp"
#include "python_optimization_matrix_utility_common.hpp"

#include "python_control.hpp"
#include "python_numpy.hpp"

#include <array>
#include <functional>
#include <type_traits>

namespace PythonOptimization {

static constexpr double Y_MIN_MAX_RHO_FACTOR_DEFAULT = 1.0e2;

/* SQP cost matrices for Nonlinear MPC */

/**
 * @brief Sequential Quadratic Programming (SQP) Cost Matrices for Nonlinear
 * Model Predictive Control (NMPC)
 *
 * This class encapsulates the cost matrices and related operations for
 * SQP-based NMPC problems. It provides interfaces for setting up cost weights,
 * constraints, system and measurement functions, and their derivatives
 * (Jacobian and Hessian). The class supports simulation of system trajectories,
 * cost and gradient computation, and analytic Hessian-vector product (HVP)
 * calculations.
 *
 * @tparam T Value type (e.g., double, float)
 * @tparam Np_In Prediction horizon length
 * @tparam Parameter_Type_In Type for system parameters
 * @tparam U_Min_Type_In Type for input lower bound
 * @tparam U_Max_Type_In Type for input upper bound
 * @tparam Y_Min_Type_In Type for output lower bound
 * @tparam Y_Max_Type_In Type for output upper bound
 * @tparam State_Jacobian_X_Matrix_Type_In Type for state Jacobian w.r.t. state
 * @tparam State_Jacobian_U_Matrix_Type_In Type for state Jacobian w.r.t. input
 * @tparam Measurement_Jacobian_X_Matrix_Type_In Type for measurement Jacobian
 * w.r.t. state
 * @tparam State_Hessian_XX_Matrix_Type_In Type for state Hessian w.r.t.
 * state-state
 * @tparam State_Hessian_XU_Matrix_Type_In Type for state Hessian w.r.t.
 * state-input
 * @tparam State_Hessian_UX_Matrix_Type_In Type for state Hessian w.r.t.
 * input-state
 * @tparam State_Hessian_UU_Matrix_Type_In Type for state Hessian w.r.t.
 * input-input
 * @tparam Measurement_Hessian_XX_Matrix_Type_In Type for measurement Hessian
 * w.r.t. state-state
 */
template <typename T, std::size_t Np_In, typename Parameter_Type_In,
          typename U_Min_Type_In, typename U_Max_Type_In,
          typename Y_Min_Type_In, typename Y_Max_Type_In,
          typename State_Jacobian_X_Matrix_Type_In,
          typename State_Jacobian_U_Matrix_Type_In,
          typename Measurement_Jacobian_X_Matrix_Type_In,
          typename State_Hessian_XX_Matrix_Type_In,
          typename State_Hessian_XU_Matrix_Type_In,
          typename State_Hessian_UX_Matrix_Type_In,
          typename State_Hessian_UU_Matrix_Type_In,
          typename Measurement_Hessian_XX_Matrix_Type_In>
class SQP_CostMatrices_NMPC {
public:
  /* Constant */
  static constexpr std::size_t STATE_SIZE =
      State_Jacobian_X_Matrix_Type_In::ROWS;
  static constexpr std::size_t INPUT_SIZE =
      State_Jacobian_U_Matrix_Type_In::ROWS;
  static constexpr std::size_t OUTPUT_SIZE =
      Measurement_Jacobian_X_Matrix_Type_In::COLS;

  // static constexpr std::size_t NP = Np_In;
  // To avoid ODR violation in C++11/14, use enum hack.
  enum : std::size_t { NP = Np_In };

  static constexpr std::size_t STATE_JACOBIAN_X_COLS = STATE_SIZE;
  static constexpr std::size_t STATE_JACOBIAN_X_ROWS = STATE_SIZE;

  static constexpr std::size_t STATE_JACOBIAN_U_COLS = STATE_SIZE;
  static constexpr std::size_t STATE_JACOBIAN_U_ROWS = INPUT_SIZE;

  static constexpr std::size_t MEASUREMENT_JACOBIAN_X_COLS = OUTPUT_SIZE;
  static constexpr std::size_t MEASUREMENT_JACOBIAN_X_ROWS = STATE_SIZE;

  static constexpr std::size_t STATE_HESSIAN_XX_COLS = STATE_SIZE * STATE_SIZE;
  static constexpr std::size_t STATE_HESSIAN_XX_ROWS = STATE_SIZE;

  static constexpr std::size_t STATE_HESSIAN_XU_COLS = STATE_SIZE * STATE_SIZE;
  static constexpr std::size_t STATE_HESSIAN_XU_ROWS = INPUT_SIZE;

  static constexpr std::size_t STATE_HESSIAN_UX_COLS = INPUT_SIZE * STATE_SIZE;
  static constexpr std::size_t STATE_HESSIAN_UX_ROWS = STATE_SIZE;

  static constexpr std::size_t STATE_HESSIAN_UU_COLS = INPUT_SIZE * STATE_SIZE;
  static constexpr std::size_t STATE_HESSIAN_UU_ROWS = INPUT_SIZE;

  static constexpr std::size_t MEASUREMENT_HESSIAN_XX_COLS =
      OUTPUT_SIZE * STATE_SIZE;
  static constexpr std::size_t MEASUREMENT_HESSIAN_XX_ROWS = STATE_SIZE;

public:
  /* Type */
  using Value_Type = T;

  using X_Type = PythonControl::StateSpaceState_Type<T, STATE_SIZE>;
  using U_Type = PythonControl::StateSpaceInput_Type<T, INPUT_SIZE>;
  using Y_Type = PythonControl::StateSpaceOutput_Type<T, OUTPUT_SIZE>;

  using X_Horizon_Type = PythonNumpy::Tile_Type<1, (NP + 1), X_Type>;
  using U_Horizon_Type = PythonNumpy::Tile_Type<1, NP, U_Type>;
  using Y_Horizon_Type = PythonNumpy::Tile_Type<1, (NP + 1), Y_Type>;

  using U_Min_Type = U_Min_Type_In;
  using U_Max_Type = U_Max_Type_In;
  using Y_Min_Type = Y_Min_Type_In;
  using Y_Max_Type = Y_Max_Type_In;

  /* Check Compatibility */
  static_assert(std::is_same<typename U_Min_Type_In::Value_Type, T>::value,
                "U_Min_Type_In::Value_Type != T");
  static_assert(std::is_same<typename U_Max_Type_In::Value_Type, T>::value,
                "U_Max_Type_In::Value_Type != T");
  static_assert(std::is_same<typename Y_Min_Type_In::Value_Type, T>::value,
                "Y_Min_Type_In::Value_Type != T");
  static_assert(std::is_same<typename Y_Max_Type_In::Value_Type, T>::value,
                "Y_Max_Type_In::Value_Type != T");

  static_assert(
      std::is_same<typename State_Jacobian_X_Matrix_Type_In::Value_Type,
                   T>::value,
      "State_Jacobian_X_Matrix_Type::Value_Type != T");
  static_assert(
      std::is_same<typename State_Jacobian_U_Matrix_Type_In::Value_Type,
                   T>::value,
      "State_Jacobian_U_Matrix_Type::Value_Type != T");
  static_assert(
      std::is_same<typename Measurement_Jacobian_X_Matrix_Type_In::Value_Type,
                   T>::value,
      "Measurement_Jacobian_X_Matrix_Type::Value_Type != T");

  static_assert(
      std::is_same<typename State_Hessian_XX_Matrix_Type_In::Value_Type,
                   T>::value,
      "State_Hessian_XX_Matrix_Type::Value_Type != T");
  static_assert(
      std::is_same<typename State_Hessian_XU_Matrix_Type_In::Value_Type,
                   T>::value,
      "State_Hessian_XU_Matrix_Type::Value_Type != T");
  static_assert(
      std::is_same<typename State_Hessian_UX_Matrix_Type_In::Value_Type,
                   T>::value,
      "State_Hessian_UX_Matrix_Type::Value_Type != T");
  static_assert(
      std::is_same<typename State_Hessian_UU_Matrix_Type_In::Value_Type,
                   T>::value,
      "State_Hessian_UU_Matrix_Type::Value_Type != T");
  static_assert(
      std::is_same<typename Measurement_Hessian_XX_Matrix_Type_In::Value_Type,
                   T>::value,
      "Measurement_Hessian_XX_Matrix_Type::Value_Type != T");

  static_assert(U_Min_Type_In::COLS == INPUT_SIZE,
                "U_Min_Type_In::COLS != INPUT_SIZE");
  static_assert(U_Min_Type_In::ROWS == 1, "U_Min_Type_In::ROWS != 1");

  static_assert(U_Max_Type_In::COLS == INPUT_SIZE,
                "U_Max_Type_In::COLS != INPUT_SIZE");
  static_assert(U_Max_Type_In::ROWS == 1, "U_Max_Type_In::ROWS != 1");

  static_assert(Y_Min_Type_In::COLS == OUTPUT_SIZE,
                "Y_Min_Type_In::COLS != OUTPUT_SIZE");
  static_assert(Y_Min_Type_In::ROWS == 1, "Y_Min_Type_In::ROWS != 1");

  static_assert(Y_Max_Type_In::COLS == OUTPUT_SIZE,
                "Y_Max_Type_In::COLS != OUTPUT_SIZE");
  static_assert(Y_Max_Type_In::ROWS == 1, "Y_Max_Type_In::ROWS != 1");

  static_assert(State_Jacobian_X_Matrix_Type_In::COLS == STATE_JACOBIAN_X_COLS,
                "State_Jacobian_X_Matrix_Type::COLS != STATE_JACOBIAN_X_COLS");
  static_assert(State_Jacobian_X_Matrix_Type_In::ROWS == STATE_JACOBIAN_X_ROWS,
                "State_Jacobian_X_Matrix_Type::ROWS != STATE_JACOBIAN_X_ROWS");

  static_assert(State_Jacobian_U_Matrix_Type_In::COLS == STATE_JACOBIAN_U_COLS,
                "State_Jacobian_U_Matrix_Type::COLS != STATE_JACOBIAN_U_COLS");
  static_assert(State_Jacobian_U_Matrix_Type_In::ROWS == STATE_JACOBIAN_U_ROWS,
                "State_Jacobian_U_Matrix_Type::ROWS != STATE_JACOBIAN_U_ROWS");

  static_assert(Measurement_Jacobian_X_Matrix_Type_In::COLS ==
                    MEASUREMENT_JACOBIAN_X_COLS,
                "Measurement_Jacobian_X_Matrix_Type::COLS != "
                "MEASUREMENT_JACOBIAN_X_COLS");
  static_assert(Measurement_Jacobian_X_Matrix_Type_In::ROWS ==
                    MEASUREMENT_JACOBIAN_X_ROWS,
                "Measurement_Jacobian_X_Matrix_Type::ROWS != "
                "MEASUREMENT_JACOBIAN_X_ROWS");

  static_assert(State_Hessian_XX_Matrix_Type_In::COLS == STATE_HESSIAN_XX_COLS,
                "State_Hessian_XX_Matrix_Type::COLS != STATE_HESSIAN_XX_COLS");
  static_assert(State_Hessian_XX_Matrix_Type_In::ROWS == STATE_HESSIAN_XX_ROWS,
                "State_Hessian_XX_Matrix_Type::ROWS != STATE_HESSIAN_XX_ROWS");

  static_assert(State_Hessian_XU_Matrix_Type_In::COLS == STATE_HESSIAN_XU_COLS,
                "State_Hessian_XU_Matrix_Type::COLS != STATE_HESSIAN_XU_COLS");
  static_assert(State_Hessian_XU_Matrix_Type_In::ROWS == STATE_HESSIAN_XU_ROWS,
                "State_Hessian_XU_Matrix_Type::ROWS != STATE_HESSIAN_XU_ROWS");

  static_assert(State_Hessian_UX_Matrix_Type_In::COLS == STATE_HESSIAN_UX_COLS,
                "State_Hessian_UX_Matrix_Type::COLS != STATE_HESSIAN_UX_COLS");
  static_assert(State_Hessian_UX_Matrix_Type_In::ROWS == STATE_HESSIAN_UX_ROWS,
                "State_Hessian_UX_Matrix_Type::ROWS != STATE_HESSIAN_UX_ROWS");

  static_assert(State_Hessian_UU_Matrix_Type_In::COLS == STATE_HESSIAN_UU_COLS,
                "State_Hessian_UU_Matrix_Type::COLS != STATE_HESSIAN_UU_COLS");
  static_assert(State_Hessian_UU_Matrix_Type_In::ROWS == STATE_HESSIAN_UU_ROWS,
                "State_Hessian_UU_Matrix_Type::ROWS != STATE_HESSIAN_UU_ROWS");

  static_assert(Measurement_Hessian_XX_Matrix_Type_In::COLS ==
                    MEASUREMENT_HESSIAN_XX_COLS,
                "Measurement_Hessian_XX_Matrix_Type::COLS != "
                "MEASUREMENT_HESSIAN_XX_COLS");
  static_assert(Measurement_Hessian_XX_Matrix_Type_In::ROWS ==
                    MEASUREMENT_HESSIAN_XX_ROWS,
                "Measurement_Hessian_XX_Matrix_Type::ROWS != "
                "MEASUREMENT_HESSIAN_XX_ROWS");

protected:
  /* Type */
  using _T = T;
  using _Parameter_Type = Parameter_Type_In;

  using _Qx_Type = PythonNumpy::DiagMatrix_Type<_T, STATE_SIZE>;
  using _R_Type = PythonNumpy::DiagMatrix_Type<_T, INPUT_SIZE>;
  using _Qy_Type = PythonNumpy::DiagMatrix_Type<_T, OUTPUT_SIZE>;

  using _U_Min_Type = U_Min_Type_In;
  using _U_Max_Type = U_Max_Type_In;
  using _Y_Min_Type = Y_Min_Type_In;
  using _Y_Max_Type = Y_Max_Type_In;

  using _U_Min_Matrix_Type = PythonNumpy::Tile_Type<1, NP, U_Min_Type_In>;
  using _U_Max_Matrix_Type = PythonNumpy::Tile_Type<1, NP, U_Max_Type_In>;
  using _Y_Min_Matrix_Type = PythonNumpy::Tile_Type<1, (NP + 1), Y_Min_Type_In>;
  using _Y_Max_Matrix_Type = PythonNumpy::Tile_Type<1, (NP + 1), Y_Max_Type_In>;

  using _StateFunction_Out_Type = X_Type;
  using _MeasurementFunction_Out_Type = Y_Type;

  using _StateFunctionJacobian_X_Out_Type = State_Jacobian_X_Matrix_Type_In;
  using _StateFunctionJacobian_U_Out_Type = State_Jacobian_U_Matrix_Type_In;
  using _MeasurementFunctionJacobian_X_Out_Type =
      Measurement_Jacobian_X_Matrix_Type_In;

  using _StateFunctionHessian_XX_Out_Type = State_Hessian_XX_Matrix_Type_In;
  using _StateFunctionHessian_XU_Out_Type = State_Hessian_XU_Matrix_Type_In;
  using _StateFunctionHessian_UX_Out_Type = State_Hessian_UX_Matrix_Type_In;
  using _StateFunctionHessian_UU_Out_Type = State_Hessian_UU_Matrix_Type_In;
  using _MeasurementFunctionHessian_XX_Out_Type =
      Measurement_Hessian_XX_Matrix_Type_In;

  using _StateFunction_Object =
      StateFunction_Object<X_Type, U_Type, _Parameter_Type>;
  using _MeasurementFunction_Object =
      MeasurementFunction_Object<Y_Type, X_Type, U_Type, _Parameter_Type>;

  using _StateFunctionJacobian_X_Object =
      StateFunctionJacobian_X_Object<_StateFunctionJacobian_X_Out_Type, X_Type,
                                     U_Type, _Parameter_Type>;
  using _StateFunctionJacobian_U_Object =
      StateFunctionJacobian_U_Object<_StateFunctionJacobian_U_Out_Type, X_Type,
                                     U_Type, _Parameter_Type>;
  using _MeasurementFunctionJacobian_X_Object =
      MeasurementFunctionJacobian_X_Object<
          _MeasurementFunctionJacobian_X_Out_Type, X_Type, U_Type,
          _Parameter_Type>;

  using _StateFunctionHessian_XX_Object =
      StateFunctionHessian_XX_Object<_StateFunctionHessian_XX_Out_Type, X_Type,
                                     U_Type, _Parameter_Type>;
  using _StateFunctionHessian_XU_Object =
      StateFunctionHessian_XU_Object<_StateFunctionHessian_XU_Out_Type, X_Type,
                                     U_Type, _Parameter_Type>;
  using _StateFunctionHessian_UX_Object =
      StateFunctionHessian_UX_Object<_StateFunctionHessian_UX_Out_Type, X_Type,
                                     U_Type, _Parameter_Type>;
  using _StateFunctionHessian_UU_Object =
      StateFunctionHessian_UU_Object<_StateFunctionHessian_UU_Out_Type, X_Type,
                                     U_Type, _Parameter_Type>;
  using _MeasurementFunctionHessian_XX_Object =
      MeasurementFunctionHessian_XX_Object<
          _MeasurementFunctionHessian_XX_Out_Type, X_Type, U_Type,
          _Parameter_Type>;

  using _Reference_Trajectory_Type = Y_Horizon_Type;

  using _Gradient_Type = U_Horizon_Type;
  using _V_Horizon_Type = U_Horizon_Type;
  using _HVP_Type = U_Horizon_Type;

public:
  /* Constructor */
  SQP_CostMatrices_NMPC()
      : _Y_min_max_rho(), _Y_offset(), _Qx(), _R(), _Qy(), _Px(), _Py(),
        _U_min_matrix(), _U_max_matrix(), _Y_min_matrix(), _Y_max_matrix(),
        _state_function(), _measurement_function(),
        _state_function_jacobian_x(), _state_function_jacobian_u(),
        _measurement_function_jacobian_x(), _state_function_hessian_xx(),
        _state_function_hessian_xu(), _state_function_hessian_ux(),
        _state_function_hessian_uu(), _measurement_function_hessian_xx() {}

  SQP_CostMatrices_NMPC(const _Qx_Type &Qx, const _R_Type &R,
                        const _Qy_Type &Qy, const U_Min_Type &U_min,
                        const U_Max_Type &U_max, const Y_Min_Type &Y_min,
                        const Y_Max_Type &Y_max)
      : _Y_min_max_rho(static_cast<_T>(Y_MIN_MAX_RHO_FACTOR_DEFAULT)),
        _Y_offset(), _Qx(Qx), _R(R), _Qy(Qy), _Px(Qx), _Py(Qy), _U_min_matrix(),
        _U_max_matrix(), _Y_min_matrix(), _Y_max_matrix(), _state_function(),
        _measurement_function(), _state_function_jacobian_x(),
        _state_function_jacobian_u(), _measurement_function_jacobian_x(),
        _state_function_hessian_xx(), _state_function_hessian_xu(),
        _state_function_hessian_ux(), _state_function_hessian_uu(),
        _measurement_function_hessian_xx() {

    this->set_U_min(U_min);
    this->set_U_max(U_max);
    this->set_Y_min(Y_min);
    this->set_Y_max(Y_max);
  }

  /* Copy Constructor */
  SQP_CostMatrices_NMPC(const SQP_CostMatrices_NMPC &input)
      : state_space_parameters(input.state_space_parameters),
        reference_trajectory(input.reference_trajectory),
        _Y_min_max_rho(input._Y_min_max_rho), _Y_offset(input._Y_offset),
        _Qx(input._Qx), _R(input._R), _Qy(input._Qy), _Px(input._Px),
        _Py(input._Py), _U_min_matrix(input._U_min_matrix),
        _U_max_matrix(input._U_max_matrix), _Y_min_matrix(input._Y_min_matrix),
        _Y_max_matrix(input._Y_max_matrix),
        _state_function(input._state_function),
        _measurement_function(input._measurement_function),
        _state_function_jacobian_x(input._state_function_jacobian_x),
        _state_function_jacobian_u(input._state_function_jacobian_u),
        _measurement_function_jacobian_x(
            input._measurement_function_jacobian_x),
        _state_function_hessian_xx(input._state_function_hessian_xx),
        _state_function_hessian_xu(input._state_function_hessian_xu),
        _state_function_hessian_ux(input._state_function_hessian_ux),
        _state_function_hessian_uu(input._state_function_hessian_uu),
        _measurement_function_hessian_xx(
            input._measurement_function_hessian_xx) {}

  SQP_CostMatrices_NMPC &operator=(const SQP_CostMatrices_NMPC &input) {
    if (this != &input) {
      this->state_space_parameters = input.state_space_parameters;
      this->reference_trajectory = input.reference_trajectory;
      this->_Y_min_max_rho = input._Y_min_max_rho;
      this->_Y_offset = input._Y_offset;

      this->_Qx = input._Qx;
      this->_R = input._R;
      this->_Qy = input._Qy;
      this->_Px = input._Px;
      this->_Py = input._Py;

      this->_U_min_matrix = input._U_min_matrix;
      this->_U_max_matrix = input._U_max_matrix;
      this->_Y_min_matrix = input._Y_min_matrix;
      this->_Y_max_matrix = input._Y_max_matrix;

      this->_state_function = input._state_function;
      this->_measurement_function = input._measurement_function;

      this->_state_function_jacobian_x = input._state_function_jacobian_x;
      this->_state_function_jacobian_u = input._state_function_jacobian_u;
      this->_measurement_function_jacobian_x =
          input._measurement_function_jacobian_x;

      this->_state_function_hessian_xx = input._state_function_hessian_xx;
      this->_state_function_hessian_xu = input._state_function_hessian_xu;
      this->_state_function_hessian_ux = input._state_function_hessian_ux;
      this->_state_function_hessian_uu = input._state_function_hessian_uu;
      this->_measurement_function_hessian_xx =
          input._measurement_function_hessian_xx;
    }
    return *this;
  }

  /* Move Constructor */
  SQP_CostMatrices_NMPC(SQP_CostMatrices_NMPC &&input) noexcept
      : state_space_parameters(std::move(input.state_space_parameters)),
        reference_trajectory(std::move(input.reference_trajectory)),
        _Y_min_max_rho(input._Y_min_max_rho),
        _Y_offset(std::move(input._Y_offset)), _Qx(std::move(input._Qx)),
        _R(std::move(input._R)), _Qy(std::move(input._Qy)),
        _Px(std::move(input._Px)), _Py(std::move(input._Py)),
        _U_min_matrix(std::move(input._U_min_matrix)),
        _U_max_matrix(std::move(input._U_max_matrix)),
        _Y_min_matrix(std::move(input._Y_min_matrix)),
        _Y_max_matrix(std::move(input._Y_max_matrix)),
        _state_function(std::move(input._state_function)),
        _measurement_function(std::move(input._measurement_function)),
        _state_function_jacobian_x(std::move(input._state_function_jacobian_x)),
        _state_function_jacobian_u(std::move(input._state_function_jacobian_u)),
        _measurement_function_jacobian_x(
            std::move(input._measurement_function_jacobian_x)),
        _state_function_hessian_xx(std::move(input._state_function_hessian_xx)),
        _state_function_hessian_xu(std::move(input._state_function_hessian_xu)),
        _state_function_hessian_ux(std::move(input._state_function_hessian_ux)),
        _state_function_hessian_uu(std::move(input._state_function_hessian_uu)),
        _measurement_function_hessian_xx(
            std::move(input._measurement_function_hessian_xx)) {}

  SQP_CostMatrices_NMPC &operator=(SQP_CostMatrices_NMPC &&input) noexcept {
    if (this != &input) {
      this->state_space_parameters = std::move(input.state_space_parameters);
      this->reference_trajectory = std::move(input.reference_trajectory);
      this->_Y_min_max_rho = input._Y_min_max_rho;
      this->_Y_offset = std::move(input._Y_offset);

      this->_Qx = std::move(input._Qx);
      this->_R = std::move(input._R);
      this->_Qy = std::move(input._Qy);
      this->_Px = std::move(input._Px);
      this->_Py = std::move(input._Py);

      this->_U_min_matrix = std::move(input._U_min_matrix);
      this->_U_max_matrix = std::move(input._U_max_matrix);
      this->_Y_min_matrix = std::move(input._Y_min_matrix);
      this->_Y_max_matrix = std::move(input._Y_max_matrix);

      this->_state_function = std::move(input._state_function);
      this->_measurement_function = std::move(input._measurement_function);

      this->_state_function_jacobian_x =
          std::move(input._state_function_jacobian_x);
      this->_state_function_jacobian_u =
          std::move(input._state_function_jacobian_u);
      this->_measurement_function_jacobian_x =
          std::move(input._measurement_function_jacobian_x);

      this->_state_function_hessian_xx =
          std::move(input._state_function_hessian_xx);
      this->_state_function_hessian_xu =
          std::move(input._state_function_hessian_xu);
      this->_state_function_hessian_ux =
          std::move(input._state_function_hessian_ux);
      this->_state_function_hessian_uu =
          std::move(input._state_function_hessian_uu);
      this->_measurement_function_hessian_xx =
          std::move(input._measurement_function_hessian_xx);
    }
    return *this;
  }

public:
  /* Setters */
  /**
   * @brief Sets the function objects required for optimization computations.
   *
   * This method assigns the provided state and measurement function objects,
   * along with their respective Jacobians and Hessians, to the internal members
   * of the class. These function objects are used in various optimization
   * routines, such as SQP (Sequential Quadratic Programming).
   *
   * @param state_function The state transition function object.
   * @param measurement_function The measurement function object.
   * @param state_function_jacobian_x Jacobian of the state function with
   * respect to state variables.
   * @param state_function_jacobian_u Jacobian of the state function with
   * respect to control variables.
   * @param measurement_function_jacobian_x Jacobian of the measurement function
   * with respect to state variables.
   * @param state_function_hessian_xx Hessian of the state function with respect
   * to state variables (XX).
   * @param state_function_hessian_xu Hessian of the state function with respect
   * to state and control variables (XU).
   * @param state_function_hessian_ux Hessian of the state function with respect
   * to control and state variables (UX).
   * @param state_function_hessian_uu Hessian of the state function with respect
   * to control variables (UU).
   * @param measurement_function_hessian_xx Hessian of the measurement function
   * with respect to state variables (XX).
   */
  inline void set_function_objects(
      const _StateFunction_Object &state_function,
      const _MeasurementFunction_Object &measurement_function,
      const _StateFunctionJacobian_X_Object &state_function_jacobian_x,
      const _StateFunctionJacobian_U_Object &state_function_jacobian_u,
      const _MeasurementFunctionJacobian_X_Object
          &measurement_function_jacobian_x,
      const _StateFunctionHessian_XX_Object &state_function_hessian_xx,
      const _StateFunctionHessian_XU_Object &state_function_hessian_xu,
      const _StateFunctionHessian_UX_Object &state_function_hessian_ux,
      const _StateFunctionHessian_UU_Object &state_function_hessian_uu,
      const _MeasurementFunctionHessian_XX_Object
          &measurement_function_hessian_xx) {

    this->_state_function = state_function;
    this->_measurement_function = measurement_function;

    this->_state_function_jacobian_x = state_function_jacobian_x;
    this->_state_function_jacobian_u = state_function_jacobian_u;
    this->_measurement_function_jacobian_x = measurement_function_jacobian_x;

    this->_state_function_hessian_xx = state_function_hessian_xx;
    this->_state_function_hessian_xu = state_function_hessian_xu;
    this->_state_function_hessian_ux = state_function_hessian_ux;
    this->_state_function_hessian_uu = state_function_hessian_uu;
    this->_measurement_function_hessian_xx = measurement_function_hessian_xx;
  }

  inline void set_U_min(const _U_Min_Type &U_min) {

    PythonNumpy::update_tile_concatenated_matrix<1, NP, _U_Min_Type>(
        this->_U_min_matrix, U_min);
  }

  inline void set_U_max(const _U_Max_Type &U_max) {

    PythonNumpy::update_tile_concatenated_matrix<1, NP, _U_Max_Type>(
        this->_U_max_matrix, U_max);
  }

  inline void set_Y_min(const _Y_Min_Type &Y_min) {

    PythonNumpy::update_tile_concatenated_matrix<1, (NP + 1), _Y_Min_Type>(
        this->_Y_min_matrix, Y_min);
  }

  inline void set_Y_max(const _Y_Max_Type &Y_max) {

    PythonNumpy::update_tile_concatenated_matrix<1, (NP + 1), _Y_Max_Type>(
        this->_Y_max_matrix, Y_max);
  }

  inline void set_Y_offset(Y_Type Y_offset) { this->_Y_offset = Y_offset; }

  inline void set_Qx(const _Qx_Type &Qx) {
    this->_Qx = Qx;
    this->_Px = Qx;
  }

  inline void set_R(const _R_Type &R) { this->_R = R; }

  inline void set_Qy(const _Qy_Type &Qy) {
    this->_Qy = Qy;
    this->_Py = Qy;
  }

  inline void set_Y_min_max_rho(const _T &Y_min_max_rho) {
    this->_Y_min_max_rho = Y_min_max_rho;
  }

  /* Getters */
  inline _U_Min_Matrix_Type get_U_min_matrix(void) const {
    return this->_U_min_matrix;
  }

  inline _U_Max_Matrix_Type get_U_max_matrix(void) const {
    return this->_U_max_matrix;
  }

  inline _Y_Min_Matrix_Type get_Y_min_matrix(void) const {
    return this->_Y_min_matrix;
  }

  inline _Y_Max_Matrix_Type get_Y_max_matrix(void) const {
    return this->_Y_max_matrix;
  }

  inline _Qx_Type get_Qx(void) const { return this->_Qx; }

  inline _R_Type get_R(void) const { return this->_R; }

  inline _Qy_Type get_Qy(void) const { return this->_Qy; }

  /* Function */
  /**
   * @brief Computes the second derivative of the objective function with
   * respect to the state variables.
   *
   * This function returns twice the value of the internal state weighting
   * matrix (_Qx). The input parameters X and U are not used in the computation.
   *
   * @param X State variables (unused).
   * @param U Control variables (unused).
   * @return _Qx_Type The result of 2 * _Qx.
   */
  inline auto l_xx(const X_Type &X, const U_Type &U) -> _Qx_Type {
    static_cast<void>(X);
    static_cast<void>(U);

    return static_cast<_T>(2) * this->_Qx;
  }

  /**
   * @brief Computes the second derivative of the Lagrangian with respect to the
   * control variable U.
   *
   * This function ignores its input parameters X and U, and returns twice the
   * value of the member variable _R, cast to type _T. It is typically used in
   * optimization routines where the Hessian with respect to U is constant.
   *
   * @tparam X_Type Type of the state variable X.
   * @tparam U_Type Type of the control variable U.
   * @tparam _R_Type Return type of the function.
   * @tparam _T Type used for casting the result.
   *
   * @param X State variable (unused).
   * @param U Control variable (unused).
   * @return _R_Type Twice the value of _R, cast to type _T.
   */
  inline auto l_uu(const X_Type &X, const U_Type &U) -> _R_Type {
    static_cast<void>(X);
    static_cast<void>(U);

    return static_cast<_T>(2) * this->_R;
  }

  /**
   * @brief Creates and returns an empty sparse matrix of type
   * PythonNumpy::SparseMatrixEmpty_Type.
   *
   * This function generates an empty sparse matrix with the specified template
   * parameters. The input arguments X and U are provided for interface
   * consistency but are not used in the function.
   *
   * @tparam _T         The data type of the matrix elements.
   * @tparam STATE_SIZE Number of rows in the matrix.
   * @tparam INPUT_SIZE Number of columns in the matrix.
   * @param X           State vector (unused).
   * @param U           Input vector (unused).
   * @return PythonNumpy::SparseMatrixEmpty_Type<_T, STATE_SIZE, INPUT_SIZE> An
   * empty sparse matrix.
   */
  inline auto l_xu(const X_Type &X, const U_Type &U)
      -> PythonNumpy::SparseMatrixEmpty_Type<_T, STATE_SIZE, INPUT_SIZE> {
    static_cast<void>(X);
    static_cast<void>(U);

    return PythonNumpy::make_SparseMatrixEmpty<_T, STATE_SIZE, INPUT_SIZE>();
  }

  /**
   * @brief Creates an empty sparse matrix representing the partial derivative
   * of the cost function with respect to the state (X) and input (U) variables.
   *
   * This function returns a sparse matrix of type
   * PythonNumpy::SparseMatrixEmpty_Type with template parameters _T,
   * INPUT_SIZE, and STATE_SIZE. The input arguments X and U are not used in the
   * computation and are only present to match the expected function signature.
   *
   * @tparam _T         The data type of the matrix elements.
   * @tparam INPUT_SIZE The number of input variables (columns).
   * @tparam STATE_SIZE The number of state variables (rows).
   * @param X           The state vector (unused).
   * @param U           The input vector (unused).
   * @return PythonNumpy::SparseMatrixEmpty_Type<_T, INPUT_SIZE, STATE_SIZE>
   *         An empty sparse matrix of the specified type and dimensions.
   */
  inline auto l_ux(const X_Type &X, const U_Type &U)
      -> PythonNumpy::SparseMatrixEmpty_Type<_T, INPUT_SIZE, STATE_SIZE> {
    static_cast<void>(X);
    static_cast<void>(U);

    return PythonNumpy::make_SparseMatrixEmpty<_T, INPUT_SIZE, STATE_SIZE>();
  }

  /**
   * @brief Calculates the state function using the provided inputs.
   *
   * This function evaluates the internal state function with the given state
   * vector `X`, control vector `U`, and parameter set `parameter`, returning
   * the computed state.
   *
   * @param X The current state vector.
   * @param U The control input vector.
   * @param parameter The set of parameters required for the state function.
   * @return _StateFunction_Out_Type The result of the state function
   * evaluation.
   */
  inline auto calculate_state_function(const X_Type &X, const U_Type &U,
                                       const _Parameter_Type &parameter)
      -> _StateFunction_Out_Type {

    return this->_state_function(X, U, parameter);
  }

  /**
   * @brief Calculates the measurement function using the provided state, input,
   * and parameters.
   *
   * This function invokes the internal measurement function with the given
   * arguments to compute the output of the measurement model.
   *
   * @param X The current state vector.
   * @param U The current input/control vector.
   * @param parameter The parameter set required for the measurement function.
   * @return The output of the measurement function.
   */
  inline auto calculate_measurement_function(const X_Type &X, const U_Type &U,
                                             const _Parameter_Type &parameter)
      -> _MeasurementFunction_Out_Type {

    return this->_measurement_function(X, U, parameter);
  }

  /**
   * @brief Calculates the Jacobian of the state function with respect to the
   * state vector X.
   *
   * This function computes the partial derivatives of the state function with
   * respect to the state variables, given the current state vector X, control
   * input vector U, and system parameters.
   *
   * @param X The current state vector.
   * @param U The current control input vector.
   * @param parameter The system parameters required for the computation.
   * @return The Jacobian matrix of the state function with respect to X.
   */
  inline auto calculate_state_jacobian_x(const X_Type &X, const U_Type &U,
                                         const _Parameter_Type &parameter)
      -> _StateFunctionJacobian_X_Out_Type {

    return this->_state_function_jacobian_x(X, U, parameter);
  }

  /**
   * @brief Calculates the Jacobian of the state function with respect to the
   * control input.
   *
   * This function computes the partial derivatives of the state function with
   * respect to the control input `U`, given the current state `X`, control
   * input `U`, and system parameters `parameter`.
   *
   * @param X The current state vector.
   * @param U The current control input vector.
   * @param parameter The system parameters required for the Jacobian
   * calculation.
   * @return The Jacobian matrix of the state function with respect to the
   * control input.
   */
  inline auto calculate_state_jacobian_u(const X_Type &X, const U_Type &U,
                                         const _Parameter_Type &parameter)
      -> _StateFunctionJacobian_U_Out_Type {

    return this->_state_function_jacobian_u(X, U, parameter);
  }

  /**
   * @brief Calculates the Jacobian of the measurement function with respect to
   * the state vector X.
   *
   * This function computes the partial derivatives of the measurement function
   * with respect to the state variables, given the current state X, input U,
   * and model parameters. It delegates the actual computation to the
   * _measurement_function_jacobian_x member function.
   *
   * @param X The current state vector.
   * @param U The current input/control vector.
   * @param parameter The model parameters required for the Jacobian
   * calculation.
   * @return The Jacobian matrix of the measurement function with respect to X.
   */
  inline auto calculate_measurement_jacobian_x(const X_Type &X, const U_Type &U,
                                               const _Parameter_Type &parameter)
      -> _MeasurementFunctionJacobian_X_Out_Type {

    return this->_measurement_function_jacobian_x(X, U, parameter);
  }

  /**
   * @brief Contracts the Hessian of the state function with the given direction
   * and lambda vector.
   *
   * This function computes the contraction of the Hessian of the state function
   * with respect to the state variables (X), using the provided direction
   * vector (dX) and the lambda vector (lam_next). The contraction is performed
   * by first obtaining the Hessian matrix via `_state_function_hessian_xx`, and
   * then applying the contraction operation using
   * `MatrixOperation::compute_fxx_lambda_contract`.
   *
   * @tparam Lambda_Vector_Type Type of the lambda vector, must be a row vector
   * with STATE_SIZE columns.
   * @param X Current state vector.
   * @param U Current control vector.
   * @param parameter Additional parameters required for the state function.
   * @param lam_next Lambda vector for the next time step (row vector of size
   * STATE_SIZE).
   * @param dX Direction vector for contraction.
   * @return X_Type Result of the contraction operation.
   *
   * @note Lambda_Vector_Type must have COLS == STATE_SIZE and ROWS == 1.
   */
  template <typename Lambda_Vector_Type>
  inline auto fx_xx_lambda_contract(const X_Type &X, const U_Type &U,
                                    const _Parameter_Type &parameter,
                                    const Lambda_Vector_Type &lam_next,
                                    const X_Type &dX) -> X_Type {

    static_assert(Lambda_Vector_Type::COLS == STATE_SIZE,
                  "Lambda_Vector_Type::COLS != STATE_SIZE");
    static_assert(Lambda_Vector_Type::ROWS == 1,
                  "Lambda_Vector_Type::ROWS != 1");

    auto Hf_xx = this->_state_function_hessian_xx(X, U, parameter);

    X_Type out;

    MatrixOperation::compute_fxx_lambda_contract(Hf_xx, dX, lam_next, out);

    return out;
  }

  /**
   * @brief Contracts the Hessian of the state function with the input direction
   * and lambda vector.
   *
   * This function computes the contraction of the Hessian of the state function
   * with respect to state and input variables (`Hf_xu`), the input direction
   * (`dU`), and the lambda vector (`lam_next`). The result is stored in an
   * output state vector and returned.
   *
   * @tparam Lambda_Vector_Type Type of the lambda vector, must have COLS ==
   * STATE_SIZE and ROWS == 1.
   * @param X Current state vector.
   * @param U Current input vector.
   * @param parameter Model or optimization parameters.
   * @param lam_next Lambda vector for the next step.
   * @param dU Input direction vector.
   * @return X_Type Resulting contracted state vector.
   *
   * @note The function asserts that the lambda vector has the correct
   * dimensions.
   */
  template <typename Lambda_Vector_Type>
  inline auto fx_xu_lambda_contract(const X_Type &X, const U_Type &U,
                                    const _Parameter_Type &parameter,
                                    const Lambda_Vector_Type &lam_next,
                                    const U_Type &dU) -> X_Type {

    static_assert(Lambda_Vector_Type::COLS == STATE_SIZE,
                  "Lambda_Vector_Type::COLS != STATE_SIZE");
    static_assert(Lambda_Vector_Type::ROWS == 1,
                  "Lambda_Vector_Type::ROWS != 1");

    auto Hf_xu = this->_state_function_hessian_xu(X, U, parameter);

    X_Type out;

    MatrixOperation::compute_fx_xu_lambda_contract(Hf_xu, dU, lam_next, out);

    return out;
  }

  /**
   * @brief Contracts the Hessian of the state function with the provided
   * direction and lambda vector.
   *
   * This function computes the contraction of the Hessian of the state function
   * with respect to state and control variables (`Hf_ux`), the direction vector
   * `dX`, and the next lambda vector `lam_next`. The result is stored in an
   * output variable of type `U_Type`.
   *
   * @tparam Lambda_Vector_Type Type of the lambda vector, must be a row vector
   * with STATE_SIZE columns.
   * @param X Current state vector.
   * @param U Current control vector.
   * @param parameter Model or optimization parameters.
   * @param lam_next Lambda vector at the next time step (row vector of size
   * STATE_SIZE).
   * @param dX Direction vector for contraction.
   * @return U_Type Result of the contraction operation.
   *
   * @note The function asserts that `Lambda_Vector_Type` is a row vector with
   * the correct size.
   * @note Relies on `MatrixOperation::compute_fu_xx_lambda_contract` for the
   * actual computation.
   */
  template <typename Lambda_Vector_Type>
  inline auto fu_xx_lambda_contract(const X_Type &X, const U_Type &U,
                                    const _Parameter_Type &parameter,
                                    const Lambda_Vector_Type &lam_next,
                                    const X_Type &dX) -> U_Type {

    static_assert(Lambda_Vector_Type::COLS == STATE_SIZE,
                  "Lambda_Vector_Type::COLS != STATE_SIZE");
    static_assert(Lambda_Vector_Type::ROWS == 1,
                  "Lambda_Vector_Type::ROWS != 1");

    auto Hf_ux = this->_state_function_hessian_ux(X, U, parameter);

    U_Type out;

    MatrixOperation::compute_fu_xx_lambda_contract(Hf_ux, dX, lam_next, out);

    return out;
  }

  /**
   * @brief Contracts the Hessian of the state function with the input direction
   * and lambda vector.
   *
   * This function computes the contraction of the Hessian of the state function
   * with respect to the input variables (`U`) using the provided state (`X`),
   * input (`U`), parameters (`parameter`), next-step lambda vector
   * (`lam_next`), and input direction (`dU`). The contraction is performed by
   * calling `MatrixOperation::compute_fu_uu_lambda_contract`, which combines
   * the Hessian, input direction, and lambda vector to produce the output.
   *
   * @tparam Lambda_Vector_Type Type of the lambda vector, must be a row vector
   * with `STATE_SIZE` columns.
   * @param X The current state vector.
   * @param U The current input vector.
   * @param parameter The parameter set for the state function.
   * @param lam_next The lambda vector for the next time step (row vector of
   * size `STATE_SIZE`).
   * @param dU The direction vector for the input.
   * @return U_Type The result of the contraction operation.
   *
   * @note Compile-time assertions ensure that `lam_next` is a row vector with
   * the correct number of columns.
   */
  template <typename Lambda_Vector_Type>
  inline auto fu_uu_lambda_contract(const X_Type &X, const U_Type &U,
                                    const _Parameter_Type &parameter,
                                    const Lambda_Vector_Type &lam_next,
                                    const U_Type &dU) -> U_Type {

    static_assert(Lambda_Vector_Type::COLS == STATE_SIZE,
                  "Lambda_Vector_Type::COLS != STATE_SIZE");
    static_assert(Lambda_Vector_Type::ROWS == 1,
                  "Lambda_Vector_Type::ROWS != 1");

    auto Hf_uu = this->_state_function_hessian_uu(X, U, parameter);

    U_Type out;

    MatrixOperation::compute_fu_uu_lambda_contract(Hf_uu, dU, lam_next, out);

    return out;
  }

  /**
   * @brief Contracts the Hessian of the measurement function with a direction
   * vector and a weight vector.
   *
   * This function computes the contraction of the Hessian of the measurement
   * function (with respect to X) using the provided direction vector `dX` and
   * the weight vector `weight`. The result is stored in `out` and returned. The
   * function asserts that the weight vector has the correct dimensions.
   *
   * @tparam Weight_Vector_Type Type of the weight vector, must have COLS ==
   * OUTPUT_SIZE and ROWS == 1.
   * @param X The input variable for which the Hessian is computed.
   * @param parameter Additional parameters required for the measurement
   * function.
   * @param weight The weight vector used in the contraction.
   * @param dX The direction vector for contraction.
   * @return X_Type The contracted result.
   */
  template <typename Weight_Vector_Type>
  inline auto hxx_lambda_contract(const X_Type &X,
                                  const _Parameter_Type &parameter,
                                  const Weight_Vector_Type &weight,
                                  const X_Type &dX) -> X_Type {

    static_assert(Weight_Vector_Type::COLS == OUTPUT_SIZE,
                  "Weight_Vector_Type::COLS != OUTPUT_SIZE");
    static_assert(Weight_Vector_Type::ROWS == 1,
                  "Weight_Vector_Type::ROWS != 1");

    U_Type U;
    auto Hh_xx = this->_measurement_function_hessian_xx(X, U, parameter);

    X_Type out;

    MatrixOperation::compute_hxx_lambda_contract(Hh_xx, dX, weight, out);

    return out;
  }

  /**
   * @brief Simulates the state trajectory over a time horizon given initial
   * state and control inputs.
   *
   * This function computes the evolution of the system state starting from an
   * initial state (`X_initial`) and applying a sequence of control inputs
   * (`U_horizon`) over a fixed number of steps (`NP`). At each step, the state
   * is updated using the `calculate_state_function`, which models the system
   * dynamics. The resulting trajectory of states is stored in `X_horizon`.
   *
   * @param X_initial The initial state vector.
   * @param U_horizon The matrix of control inputs for each time step in the
   * horizon.
   * @param parameter Additional parameters required for the state update
   * function.
   * @return X_Horizon_Type The matrix containing the state trajectory over the
   * horizon.
   */
  inline auto simulate_trajectory(const X_Type &X_initial,
                                  const U_Horizon_Type &U_horizon,
                                  const _Parameter_Type &parameter)
      -> X_Horizon_Type {

    X_Horizon_Type X_horizon;
    X_Type X = X_initial;

    MatrixOperation::set_row(X_horizon, X, 0);

    for (std::size_t k = 0; k < NP; k++) {
      auto U = MatrixOperation::get_row(U_horizon, k);
      X = this->calculate_state_function(X, U, parameter);

      MatrixOperation::set_row(X_horizon, X, k + 1);
    }

    return X_horizon;
  }

  /**
   * @brief Calculates the penalty for violating Y horizon limits.
   *
   * This function computes a penalty matrix based on the provided Y_horizon
   * values, comparing them against the minimum and maximum Y limits stored in
   * the class. The penalty is calculated using
   * MatrixOperation::calculate_Y_limit_penalty and returned as a new matrix.
   *
   * @param Y_horizon The matrix or vector representing the Y horizon values to
   * be checked.
   * @return Y_Horizon_Type The resulting penalty matrix for Y horizon limit
   * violations.
   */
  inline auto calculate_Y_limit_penalty(const Y_Horizon_Type &Y_horizon)
      -> Y_Horizon_Type {
    Y_Horizon_Type Y_limit_penalty;

    MatrixOperation::calculate_Y_limit_penalty(
        Y_horizon, this->_Y_min_matrix, this->_Y_max_matrix, Y_limit_penalty);

    return Y_limit_penalty;
  }

  /**
   * @brief Calculates the penalty and active status for Y horizon limits.
   *
   * This function initializes the penalty and active status matrices for the Y
   * horizon, then computes the penalty and active status based on the provided
   * Y horizon values and the internal minimum and maximum Y matrices.
   *
   * @param Y_horizon The input matrix representing the Y horizon values.
   * @param Y_limit_penalty Output matrix to store the calculated penalty for Y
   * limits.
   * @param Y_limit_active Output matrix to store the active status of Y limits.
   */
  inline void
  calculate_Y_limit_penalty_and_active(const Y_Horizon_Type &Y_horizon,
                                       Y_Horizon_Type &Y_limit_penalty,
                                       Y_Horizon_Type &Y_limit_active) {

    Y_limit_penalty = Y_Horizon_Type();
    Y_limit_active = Y_Horizon_Type();

    MatrixOperation::calculate_Y_limit_penalty_and_active(
        Y_horizon, this->_Y_min_matrix, this->_Y_max_matrix, Y_limit_penalty,
        Y_limit_active);
  }

  /**
   * @brief Computes the cost function for a given initial state and control
   * horizon.
   *
   * This function simulates the system trajectory over the prediction horizon
   * using the provided initial state and control inputs. It then calculates the
   * measurement outputs, applies penalties for constraint violations, and
   * evaluates the quadratic cost terms for state, output tracking error,
   * control effort, and output limit penalties. The total cost is accumulated
   * over the prediction horizon and returned.
   *
   * @param X_initial Initial state vector of the system.
   * @param U_horizon Control input sequence over the prediction horizon.
   * @return _T The computed cost value.
   */
  inline auto compute_cost(const X_Type X_initial,
                           const U_Horizon_Type &U_horizon) -> _T {

    auto X_horizon = this->simulate_trajectory(X_initial, U_horizon,
                                               this->state_space_parameters);

    Y_Horizon_Type Y_horizon;
    PythonNumpy::update_tile_concatenated_matrix<1, (NP + 1), Y_Type>(
        Y_horizon, this->_Y_offset);
    U_Type U_dummy;

    for (std::size_t k = 0; k < (NP + 1); k++) {
      auto Y_k = this->calculate_measurement_function(
          MatrixOperation::get_row(X_horizon, k), U_dummy,
          this->state_space_parameters);

      MatrixOperation::set_row(Y_horizon, Y_k, k);
    }

    auto Y_limit_penalty = this->calculate_Y_limit_penalty(Y_horizon);

    _T J = static_cast<_T>(0);
    for (std::size_t k = 0; k < NP; k++) {
      auto e_y_r = MatrixOperation::get_row(Y_horizon, k) -
                   MatrixOperation::get_row(this->reference_trajectory, k);

      auto X_T_Qx_X = MatrixOperation::calculate_quadratic_form(
          MatrixOperation::get_row(X_horizon, k), this->_Qx);
      auto e_y_r_T_Qy_e_y_r =
          MatrixOperation::calculate_quadratic_form(e_y_r, this->_Qy);

      auto U_T_R_U = MatrixOperation::calculate_quadratic_form(
          MatrixOperation::get_row(U_horizon, k), this->_R);

      auto Y_limit_penalty_T_Y_limit_penalty =
          MatrixOperation::calculate_quadratic_no_weighted(
              MatrixOperation::get_row(Y_limit_penalty, k));

      J += X_T_Qx_X + e_y_r_T_Qy_e_y_r + U_T_R_U +
           this->_Y_min_max_rho * Y_limit_penalty_T_Y_limit_penalty;
    }

    auto eN_y_r = MatrixOperation::get_row(Y_horizon, NP) -
                  MatrixOperation::get_row(this->reference_trajectory, NP);

    auto XN_T_Qx_XN = MatrixOperation::calculate_quadratic_form(
        MatrixOperation::get_row(X_horizon, NP), this->_Qx);
    auto eN_y_r_T_Qy_eN_y_r =
        MatrixOperation::calculate_quadratic_form(eN_y_r, this->_Qy);
    auto YN_limit_penalty_T_YN_limit_penalty =
        MatrixOperation::calculate_quadratic_no_weighted(
            MatrixOperation::get_row(Y_limit_penalty, NP));

    J += XN_T_Qx_XN + eN_y_r_T_Qy_eN_y_r +
         this->_Y_min_max_rho * YN_limit_penalty_T_YN_limit_penalty;

    return J;
  }

  /**
   * @brief Computes the cost function value and its gradient for a given
   * initial state and control horizon.
   *
   * This function simulates the system trajectory starting from the initial
   * state using the provided control horizon. It then calculates the
   * measurement trajectory, applies penalties for measurement limits, and
   * evaluates the cost function over the prediction horizon. The gradient of
   * the cost function with respect to the control inputs is computed using
   * adjoint sensitivity analysis.
   *
   * @tparam X_Type Type representing the initial state vector.
   * @tparam U_Horizon_Type Type representing the control input horizon matrix.
   * @tparam _T Scalar type used for cost and gradient calculations.
   * @tparam _Gradient_Type Type representing the gradient vector/matrix.
   *
   * @param X_initial Initial state vector.
   * @param U_horizon Control input horizon matrix.
   * @param[out] J Computed cost function value (accumulated over the horizon).
   * @param[out] gradient Gradient of the cost function with respect to the
   * control inputs.
   *
   * @note
   * - Assumes that all required system parameters and matrices (e.g., Qx, Qy,
   * R, Px, Py) are initialized.
   * - The function is intended for use in sequential quadratic programming
   * (SQP) or similar optimization routines.
   */
  inline void compute_cost_and_gradient(const X_Type X_initial,
                                        const U_Horizon_Type &U_horizon, _T &J,
                                        _Gradient_Type &gradient) {

    U_Type U_dummy;

    auto X_horizon = this->simulate_trajectory(X_initial, U_horizon,
                                               this->state_space_parameters);

    Y_Horizon_Type Y_horizon;
    PythonNumpy::update_tile_concatenated_matrix<1, (NP + 1), Y_Type>(
        Y_horizon, this->_Y_offset);

    for (std::size_t k = 0; k < (NP + 1); k++) {
      auto Y_k = this->calculate_measurement_function(
          MatrixOperation::get_row(X_horizon, k), U_dummy,
          this->state_space_parameters);

      MatrixOperation::set_row(Y_horizon, Y_k, k);
    }

    auto Y_limit_penalty = this->calculate_Y_limit_penalty(Y_horizon);

    J = static_cast<_T>(0);
    for (std::size_t k = 0; k < NP; k++) {
      auto e_y_r = MatrixOperation::get_row(Y_horizon, k) -
                   MatrixOperation::get_row(this->reference_trajectory, k);

      auto X_T_Qx_X = MatrixOperation::calculate_quadratic_form(
          MatrixOperation::get_row(X_horizon, k), this->_Qx);
      auto e_y_r_T_Qy_e_y_r =
          MatrixOperation::calculate_quadratic_form(e_y_r, this->_Qy);

      auto U_T_R_U = MatrixOperation::calculate_quadratic_form(
          MatrixOperation::get_row(U_horizon, k), this->_R);

      auto Y_limit_penalty_T_Y_limit_penalty =
          MatrixOperation::calculate_quadratic_no_weighted(
              MatrixOperation::get_row(Y_limit_penalty, k));

      J += X_T_Qx_X + e_y_r_T_Qy_e_y_r + U_T_R_U +
           this->_Y_min_max_rho * Y_limit_penalty_T_Y_limit_penalty;
    }

    auto eN_y_r = MatrixOperation::get_row(Y_horizon, NP) -
                  MatrixOperation::get_row(this->reference_trajectory, NP);

    auto XN_T_Qx_XN = MatrixOperation::calculate_quadratic_form(
        MatrixOperation::get_row(X_horizon, NP), this->_Qx);
    auto eN_y_r_T_Qy_eN_y_r =
        MatrixOperation::calculate_quadratic_form(eN_y_r, this->_Qy);
    auto YN_limit_penalty_T_YN_limit_penalty =
        MatrixOperation::calculate_quadratic_no_weighted(
            MatrixOperation::get_row(Y_limit_penalty, NP));

    J += XN_T_Qx_XN + eN_y_r_T_Qy_eN_y_r +
         this->_Y_min_max_rho * YN_limit_penalty_T_YN_limit_penalty;

    // terminal adjoint
    auto C_N = this->calculate_measurement_jacobian_x(
        MatrixOperation::get_row(X_horizon, NP), U_dummy,
        this->state_space_parameters);

    auto Px_X = this->_Px * MatrixOperation::get_row(X_horizon, NP);
    auto Py_eN_y_r = this->_Py * eN_y_r;
    auto Y_min_max_rho_YN_limit_penalty =
        this->_Y_min_max_rho * MatrixOperation::get_row(Y_limit_penalty, NP);

    auto lam_next =
        static_cast<_T>(2) *
        (Px_X + PythonNumpy::ATranspose_mul_B(
                    C_N, (Py_eN_y_r + Y_min_max_rho_YN_limit_penalty)));

    gradient = _Gradient_Type();

    for (std::size_t k = NP; k-- > 0;) {

      auto Cx_k = this->calculate_measurement_jacobian_x(
          MatrixOperation::get_row(X_horizon, k), U_dummy,
          this->state_space_parameters);

      auto ek_y = MatrixOperation::get_row(Y_horizon, k) -
                  MatrixOperation::get_row(this->reference_trajectory, k);

      auto A_k = this->calculate_state_jacobian_x(
          MatrixOperation::get_row(X_horizon, k),
          MatrixOperation::get_row(U_horizon, k), this->state_space_parameters);

      auto B_k = this->calculate_state_jacobian_u(
          MatrixOperation::get_row(X_horizon, k),
          MatrixOperation::get_row(U_horizon, k), this->state_space_parameters);

      auto _2_R_U = static_cast<_T>(2) * this->_R *
                    MatrixOperation::get_row(U_horizon, k);
      auto B_k_T_lam_next = PythonNumpy::ATranspose_mul_B(B_k, lam_next);

      MatrixOperation::set_row(gradient, _2_R_U + B_k_T_lam_next, k);

      auto Qx_X = this->_Qx * MatrixOperation::get_row(X_horizon, k);
      auto Qy_ek_y = this->_Qy * ek_y;
      auto Y_min_max_rho_Yk_limit_penalty =
          static_cast<_T>(2) * this->_Y_min_max_rho *
          MatrixOperation::get_row(Y_limit_penalty, k);

      auto A_k_T_lam_next = PythonNumpy::ATranspose_mul_B(A_k, lam_next);

      lam_next =
          static_cast<_T>(2) *
              (Qx_X + PythonNumpy::ATranspose_mul_B(
                          Cx_k, (Qy_ek_y + Y_min_max_rho_Yk_limit_penalty))) +
          A_k_T_lam_next;
    }
  }

  /**
   * @brief Computes the analytic Hessian-vector product (HVP) for the SQP
   * optimization problem.
   *
   * This function calculates the Hessian-vector product of the Lagrangian with
   * respect to the control input trajectory, using analytic derivatives and
   * adjoint methods. It performs the following steps:
   *   1. Simulates the state trajectory forward in time given the initial state
   * and control horizon.
   *   2. Computes the measurement outputs and applies penalties for output
   * constraints.
   *   3. Calculates the first-order adjoint (costate lambda) using backward
   * recursion.
   *   4. Propagates the directional state perturbation (delta_x) forward in
   * time.
   *   5. Computes the second-order adjoint (d_lambda) using backward recursion,
   * including contributions from output penalties and second-order terms.
   *   6. Assembles the Hessian-vector product for the control input trajectory,
   * including all second-order terms from dynamics and cost.
   *
   * @param X_initial Initial state vector.
   * @param U_horizon Control input trajectory over the horizon.
   * @param V_horizon Directional perturbation of the control input trajectory.
   * @return Hessian-vector product with respect to the control input
   * trajectory.
   */
  inline auto hvp_analytic(const X_Type &X_initial,
                           const U_Horizon_Type &U_horizon,
                           const _V_Horizon_Type &V_horizon) -> _HVP_Type {

    U_Type U_dummy;

    /* --- 1) forward states */
    auto X_horizon = this->simulate_trajectory(X_initial, U_horizon,
                                               this->state_space_parameters);

    Y_Horizon_Type Y_horizon;
    for (std::size_t k = 0; k < (NP + 1); k++) {
      auto Y_k = this->calculate_measurement_function(
          MatrixOperation::get_row(X_horizon, k), U_dummy,
          this->state_space_parameters);

      MatrixOperation::set_row(Y_horizon, Y_k, k);
    }

    auto yN = this->calculate_measurement_function(
        MatrixOperation::get_row(X_horizon, NP), U_dummy,
        this->state_space_parameters);

    auto eN_y = yN - MatrixOperation::get_row(this->reference_trajectory, NP);

    Y_Horizon_Type Y_limit_penalty;
    Y_Horizon_Type Y_limit_active;
    this->calculate_Y_limit_penalty_and_active(Y_horizon, Y_limit_penalty,
                                               Y_limit_active);

    // /* --- 2) first-order adjoint (costate lambda) with output terms */
    X_Horizon_Type lam;
    auto Cx_N = this->calculate_measurement_jacobian_x(
        MatrixOperation::get_row(X_horizon, NP), U_dummy,
        this->state_space_parameters);

    auto Px_XN = this->_Px * MatrixOperation::get_row(X_horizon, NP);
    auto Py_eN_y = this->_Py * eN_y;
    auto Y_min_max_rho_YN_limit_penalty =
        this->_Y_min_max_rho * MatrixOperation::get_row(Y_limit_penalty, NP);

    auto lam_input =
        static_cast<_T>(2) *
        (Px_XN + PythonNumpy::ATranspose_mul_B(
                     Cx_N, (Py_eN_y + Y_min_max_rho_YN_limit_penalty)));
    MatrixOperation::set_row(lam, lam_input, NP);

    for (std::size_t k = NP; k-- > 0;) {
      auto A_k = this->calculate_state_jacobian_x(
          MatrixOperation::get_row(X_horizon, k),
          MatrixOperation::get_row(U_horizon, k), this->state_space_parameters);

      auto Cx_k = this->calculate_measurement_jacobian_x(
          MatrixOperation::get_row(X_horizon, k), U_dummy,
          this->state_space_parameters);

      auto ek_y = MatrixOperation::get_row(Y_horizon, k) -
                  MatrixOperation::get_row(this->reference_trajectory, k);

      auto Qx_X = this->_Qx * MatrixOperation::get_row(X_horizon, k);
      auto Qy_ek_y = this->_Qy * ek_y;
      auto Y_min_max_rho_Yk_limit_penalty =
          this->_Y_min_max_rho * MatrixOperation::get_row(Y_limit_penalty, k);
      auto A_k_T_lam = PythonNumpy::ATranspose_mul_B(
          A_k, MatrixOperation::get_row(lam, k + 1));

      auto lam_input_ =
          static_cast<_T>(2) *
              (Qx_X + PythonNumpy::ATranspose_mul_B(
                          Cx_k, (Qy_ek_y + Y_min_max_rho_Yk_limit_penalty))) +
          A_k_T_lam;
      MatrixOperation::set_row(lam, lam_input_, k);
    }

    /* --- 3) forward directional state: delta_x --- */
    X_Horizon_Type dx;
    for (std::size_t k = 0; k < NP; k++) {
      auto A_k = this->calculate_state_jacobian_x(
          MatrixOperation::get_row(X_horizon, k),
          MatrixOperation::get_row(U_horizon, k), this->state_space_parameters);
      auto B_k = this->calculate_state_jacobian_u(
          MatrixOperation::get_row(X_horizon, k),
          MatrixOperation::get_row(U_horizon, k), this->state_space_parameters);

      auto dx_input = A_k * MatrixOperation::get_row(dx, k) +
                      B_k * MatrixOperation::get_row(V_horizon, k);
      MatrixOperation::set_row(dx, dx_input, k + 1);
    }

    /* --- 4) backward second-order adjoint --- */
    X_Horizon_Type d_lambda;

    /*
     * Match the treatment of the terminal term phi_xx = l_xx(X_N,·)
     (currently
     * 2P) Additionally, contributions from pure second-order output and
     second
     * derivatives of output
     */
    auto l_xx_dx =
        this->l_xx(MatrixOperation::get_row(X_horizon, NP), U_dummy) *
        MatrixOperation::get_row(dx, NP);
    auto Cx_N_dx = Cx_N * MatrixOperation::get_row(dx, NP);

    auto CX_N_T_Py_Cx_N_dx = PythonNumpy::ATranspose_mul_B(
        Cx_N, (static_cast<_T>(2) * this->_Py * Cx_N_dx));

    Y_Type YN_limit_active_CX_N_dx;
    PythonNumpy::element_wise_multiply(
        YN_limit_active_CX_N_dx, MatrixOperation::get_row(Y_limit_active, NP),
        Cx_N_dx);

    auto Y_min_max_rho_YN_limit_active_CX_N_dx =
        static_cast<_T>(2) * this->_Y_min_max_rho * YN_limit_active_CX_N_dx;

    auto CX_N_T_penalty_CX_N_dx = PythonNumpy::ATranspose_mul_B(
        Cx_N, Y_min_max_rho_YN_limit_active_CX_N_dx);

    Y_min_max_rho_YN_limit_penalty =
        static_cast<_T>(2) * this->_Y_min_max_rho *
        MatrixOperation::get_row(Y_limit_penalty, NP);

    auto Hxx_penalty_term_N = this->hxx_lambda_contract(
        MatrixOperation::get_row(X_horizon, NP), this->state_space_parameters,
        Y_min_max_rho_YN_limit_penalty, MatrixOperation::get_row(dx, NP));

    auto d_lambda_input = l_xx_dx + CX_N_T_Py_Cx_N_dx + Hxx_penalty_term_N +
                          CX_N_T_penalty_CX_N_dx;
    MatrixOperation::set_row(d_lambda, d_lambda_input, NP);

    _HVP_Type Hu;

    for (std::size_t k = NP; k-- > 0;) {
      auto A_k = this->calculate_state_jacobian_x(
          MatrixOperation::get_row(X_horizon, k),
          MatrixOperation::get_row(U_horizon, k), this->state_space_parameters);

      auto B_k = this->calculate_state_jacobian_u(
          MatrixOperation::get_row(X_horizon, k),
          MatrixOperation::get_row(U_horizon, k), this->state_space_parameters);

      auto Cx_k = this->calculate_measurement_jacobian_x(
          MatrixOperation::get_row(X_horizon, k), U_dummy,
          this->state_space_parameters);

      auto ek_y = MatrixOperation::get_row(Y_horizon, k) -
                  MatrixOperation::get_row(this->reference_trajectory, k);

      auto Cx_dx_k = Cx_k * MatrixOperation::get_row(dx, k);

      auto term_Qy_GN = PythonNumpy::ATranspose_mul_B(
          Cx_k, (static_cast<_T>(2) * this->_Py * Cx_dx_k));

      auto term_Qy_hxx = this->hxx_lambda_contract(
          MatrixOperation::get_row(X_horizon, k), this->state_space_parameters,
          static_cast<_T>(2) * this->_Py * ek_y,
          MatrixOperation::get_row(dx, k));

      Y_Type YN_limit_active_CX_k_dx;
      PythonNumpy::element_wise_multiply(
          YN_limit_active_CX_k_dx, MatrixOperation::get_row(Y_limit_active, k),
          Cx_dx_k);

      auto Y_min_max_rho_Yk_limit_active_Cx_dx_k =
          static_cast<_T>(2) * this->_Y_min_max_rho * YN_limit_active_CX_k_dx;

      auto term_penalty_GN = PythonNumpy::ATranspose_mul_B(
          Cx_k, Y_min_max_rho_Yk_limit_active_Cx_dx_k);

      auto Y_min_max_rho_Yk_limit_penalty =
          static_cast<_T>(2) * this->_Y_min_max_rho *
          MatrixOperation::get_row(Y_limit_penalty, k);

      auto term_penalty_hxx = this->hxx_lambda_contract(
          MatrixOperation::get_row(X_horizon, k), this->state_space_parameters,
          Y_min_max_rho_Yk_limit_penalty, MatrixOperation::get_row(dx, k));

      auto term_xx = this->fx_xx_lambda_contract(
          MatrixOperation::get_row(X_horizon, k),
          MatrixOperation::get_row(U_horizon, k), this->state_space_parameters,
          MatrixOperation::get_row(lam, k + 1),
          MatrixOperation::get_row(dx, k));

      auto term_xu = this->fx_xu_lambda_contract(
          MatrixOperation::get_row(X_horizon, k),
          MatrixOperation::get_row(U_horizon, k), this->state_space_parameters,
          MatrixOperation::get_row(lam, k + 1),
          MatrixOperation::get_row(V_horizon, k));

      auto l_xx_dx_ = this->l_xx(MatrixOperation::get_row(X_horizon, k),
                                 MatrixOperation::get_row(U_horizon, k)) *
                      MatrixOperation::get_row(dx, k);

      auto l_xu_V = this->l_xu(MatrixOperation::get_row(X_horizon, k),
                               MatrixOperation::get_row(U_horizon, k)) *
                    MatrixOperation::get_row(V_horizon, k);

      auto A_k_T_d_lambda = PythonNumpy::ATranspose_mul_B(
          A_k, MatrixOperation::get_row(d_lambda, k + 1));

      auto d_lambda_input_ = l_xx_dx_ + l_xu_V + A_k_T_d_lambda + term_Qy_GN +
                             term_Qy_hxx + term_penalty_GN + term_penalty_hxx +
                             term_xx + term_xu;

      MatrixOperation::set_row(d_lambda, d_lambda_input_, k);

      /*
       * (HV)_k:
       * 2R V + B^T dlambda_{k+1} + second-order terms from dynamics
       * (Cu=0 -> no direct contribution from output terms)
       */
      auto term_ux = this->fu_xx_lambda_contract(
          MatrixOperation::get_row(X_horizon, k),
          MatrixOperation::get_row(U_horizon, k), this->state_space_parameters,
          MatrixOperation::get_row(lam, k + 1),
          MatrixOperation::get_row(dx, k));

      auto term_uu = this->fu_uu_lambda_contract(
          MatrixOperation::get_row(X_horizon, k),
          MatrixOperation::get_row(U_horizon, k), this->state_space_parameters,
          MatrixOperation::get_row(lam, k + 1),
          MatrixOperation::get_row(V_horizon, k));

      auto l_uu_V = this->l_uu(MatrixOperation::get_row(X_horizon, k),
                               MatrixOperation::get_row(U_horizon, k)) *
                    MatrixOperation::get_row(V_horizon, k);

      auto l_ux_dx = this->l_ux(MatrixOperation::get_row(X_horizon, k),
                                MatrixOperation::get_row(U_horizon, k)) *
                     MatrixOperation::get_row(dx, k);

      auto B_k_T_d_lambda = PythonNumpy::ATranspose_mul_B(
          B_k, MatrixOperation::get_row(d_lambda, k + 1));

      auto Hu_input = l_uu_V + l_ux_dx + B_k_T_d_lambda + term_ux + term_uu;
      MatrixOperation::set_row(Hu, Hu_input, k);
    }

    return Hu;
  }

public:
  /* Variable */
  _Parameter_Type state_space_parameters;
  _Reference_Trajectory_Type reference_trajectory;

protected:
  /* Variable */
  _T _Y_min_max_rho;
  Y_Type _Y_offset;

  _Qx_Type _Qx;
  _R_Type _R;
  _Qy_Type _Qy;

  _Qx_Type _Px;
  _Qy_Type _Py;

  _U_Min_Matrix_Type _U_min_matrix;
  _U_Max_Matrix_Type _U_max_matrix;
  _Y_Min_Matrix_Type _Y_min_matrix;
  _Y_Max_Matrix_Type _Y_max_matrix;

  _StateFunction_Object _state_function;
  _MeasurementFunction_Object _measurement_function;

  _StateFunctionJacobian_X_Object _state_function_jacobian_x;
  _StateFunctionJacobian_U_Object _state_function_jacobian_u;
  _MeasurementFunctionJacobian_X_Object _measurement_function_jacobian_x;

  _StateFunctionHessian_XX_Object _state_function_hessian_xx;
  _StateFunctionHessian_XU_Object _state_function_hessian_xu;
  _StateFunctionHessian_UX_Object _state_function_hessian_ux;
  _StateFunctionHessian_UU_Object _state_function_hessian_uu;
  _MeasurementFunctionHessian_XX_Object _measurement_function_hessian_xx;
};

/* make SQP_CostMatrices_NMPC */
/**
 * @brief Constructs an SQP_CostMatrices_NMPC object for Nonlinear Model
 * Predictive Control (NMPC).
 *
 * This function creates and returns an instance of SQP_CostMatrices_NMPC, which
 * encapsulates the cost matrices and constraints required for Sequential
 * Quadratic Programming (SQP) in NMPC.
 *
 * @tparam T Scalar type for matrix elements.
 * @tparam Np Prediction horizon length.
 * @tparam Parameter_Type Type representing additional parameters.
 * @tparam U_Min_Type Type for minimum control input constraints.
 * @tparam U_Max_Type Type for maximum control input constraints.
 * @tparam Y_Min_Type Type for minimum output constraints.
 * @tparam Y_Max_Type Type for maximum output constraints.
 * @tparam State_Jacobian_X_Matrix_Type Type for state Jacobian with respect to
 * states.
 * @tparam State_Jacobian_U_Matrix_Type Type for state Jacobian with respect to
 * inputs.
 * @tparam Measurement_Jacobian_X_Matrix_Type Type for measurement Jacobian with
 * respect to states.
 * @tparam State_Hessian_XX_Matrix_Type Type for state Hessian with respect to
 * states.
 * @tparam State_Hessian_XU_Matrix_Type Type for state Hessian with respect to
 * state and input.
 * @tparam State_Hessian_UX_Matrix_Type Type for state Hessian with respect to
 * input and state.
 * @tparam State_Hessian_UU_Matrix_Type Type for state Hessian with respect to
 * inputs.
 * @tparam Measurement_Hessian_XX_Matrix_Type Type for measurement Hessian with
 * respect to states.
 *
 * @param Qx Diagonal weighting matrix for state error.
 * @param R Diagonal weighting matrix for control input.
 * @param Qy Diagonal weighting matrix for measurement error.
 * @param U_min Minimum control input constraints.
 * @param U_max Maximum control input constraints.
 * @param Y_min Minimum output constraints.
 * @param Y_max Maximum output constraints.
 *
 * @return SQP_CostMatrices_NMPC object initialized with the provided matrices
 * and constraints.
 */
template <typename T, std::size_t Np, typename Parameter_Type,
          typename U_Min_Type, typename U_Max_Type, typename Y_Min_Type,
          typename Y_Max_Type, typename State_Jacobian_X_Matrix_Type,
          typename State_Jacobian_U_Matrix_Type,
          typename Measurement_Jacobian_X_Matrix_Type,
          typename State_Hessian_XX_Matrix_Type,
          typename State_Hessian_XU_Matrix_Type,
          typename State_Hessian_UX_Matrix_Type,
          typename State_Hessian_UU_Matrix_Type,
          typename Measurement_Hessian_XX_Matrix_Type>
inline auto make_SQP_CostMatrices_NMPC(
    const PythonNumpy::DiagMatrix_Type<T, State_Jacobian_X_Matrix_Type::ROWS>
        &Qx,
    const PythonNumpy::DiagMatrix_Type<T, State_Jacobian_U_Matrix_Type::ROWS>
        &R,
    const PythonNumpy::DiagMatrix_Type<
        T, Measurement_Jacobian_X_Matrix_Type::COLS> &Qy,
    U_Min_Type U_min, U_Max_Type U_max, Y_Min_Type Y_min, Y_Max_Type Y_max)
    -> SQP_CostMatrices_NMPC<
        T, Np, Parameter_Type, U_Min_Type, U_Max_Type, Y_Min_Type, Y_Max_Type,
        State_Jacobian_X_Matrix_Type, State_Jacobian_U_Matrix_Type,
        Measurement_Jacobian_X_Matrix_Type, State_Hessian_XX_Matrix_Type,
        State_Hessian_XU_Matrix_Type, State_Hessian_UX_Matrix_Type,
        State_Hessian_UU_Matrix_Type, Measurement_Hessian_XX_Matrix_Type> {

  return SQP_CostMatrices_NMPC<
      T, Np, Parameter_Type, U_Min_Type, U_Max_Type, Y_Min_Type, Y_Max_Type,
      State_Jacobian_X_Matrix_Type, State_Jacobian_U_Matrix_Type,
      Measurement_Jacobian_X_Matrix_Type, State_Hessian_XX_Matrix_Type,
      State_Hessian_XU_Matrix_Type, State_Hessian_UX_Matrix_Type,
      State_Hessian_UU_Matrix_Type, Measurement_Hessian_XX_Matrix_Type>(
      Qx, R, Qy, U_min, U_max, Y_min, Y_max);
}

/* SQP_CostMatrices_NMPC type */
/**
 * @brief Alias template for SQP_CostMatrices_NMPC with multiple type
 * parameters.
 *
 * This alias simplifies the usage of the SQP_CostMatrices_NMPC template by
 * providing a more concise name.
 *
 * @tparam T Scalar or numeric type used in cost matrices.
 * @tparam Np Number of prediction steps or horizon length.
 * @tparam Parameter_Type Type representing optimization parameters.
 * @tparam U_Min_Type Type representing minimum control input constraints.
 * @tparam U_Max_Type Type representing maximum control input constraints.
 * @tparam Y_Min_Type Type representing minimum output constraints.
 * @tparam Y_Max_Type Type representing maximum output constraints.
 * @tparam State_Jacobian_X_Matrix_Type Type for state Jacobian with respect to
 * state variables.
 * @tparam State_Jacobian_U_Matrix_Type Type for state Jacobian with respect to
 * control inputs.
 * @tparam Measurement_Jacobian_X_Matrix_Type Type for measurement Jacobian with
 * respect to state variables.
 * @tparam State_Hessian_XX_Matrix_Type Type for state Hessian with respect to
 * state variables.
 * @tparam State_Hessian_XU_Matrix_Type Type for state Hessian with respect to
 * state and control variables.
 * @tparam State_Hessian_UX_Matrix_Type Type for state Hessian with respect to
 * control and state variables.
 * @tparam State_Hessian_UU_Matrix_Type Type for state Hessian with respect to
 * control variables.
 * @tparam Measurement_Hessian_XX_Matrix_Type Type for measurement Hessian with
 * respect to state variables.
 */
template <typename T, std::size_t Np, typename Parameter_Type,
          typename U_Min_Type, typename U_Max_Type, typename Y_Min_Type,
          typename Y_Max_Type, typename State_Jacobian_X_Matrix_Type,
          typename State_Jacobian_U_Matrix_Type,
          typename Measurement_Jacobian_X_Matrix_Type,
          typename State_Hessian_XX_Matrix_Type,
          typename State_Hessian_XU_Matrix_Type,
          typename State_Hessian_UX_Matrix_Type,
          typename State_Hessian_UU_Matrix_Type,
          typename Measurement_Hessian_XX_Matrix_Type>
using SQP_CostMatrices_NMPC_Type = SQP_CostMatrices_NMPC<
    T, Np, Parameter_Type, U_Min_Type, U_Max_Type, Y_Min_Type, Y_Max_Type,
    State_Jacobian_X_Matrix_Type, State_Jacobian_U_Matrix_Type,
    Measurement_Jacobian_X_Matrix_Type, State_Hessian_XX_Matrix_Type,
    State_Hessian_XU_Matrix_Type, State_Hessian_UX_Matrix_Type,
    State_Hessian_UU_Matrix_Type, Measurement_Hessian_XX_Matrix_Type>;

} // namespace PythonOptimization

#endif // __PYTHON_OPTIMIZATION_SQP_MATRIX_UTILITY_HPP__
