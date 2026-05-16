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
#ifndef PYTHON_OPTIMIZATION_SQP_MATRIX_UTILITY_HPP_
#define PYTHON_OPTIMIZATION_SQP_MATRIX_UTILITY_HPP_

#include "python_optimization_matrix_utility_common.hpp"
#include "python_optimization_utility_matrix_operation.hpp"

#include "python_control.hpp"
#include "python_numpy.hpp"

#include <array>
#include <functional>
#include <tuple>
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
 * constraints, system and measurement equations, and their derivatives
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
      State_Jacobian_X_Matrix_Type_In::COLS;
  static constexpr std::size_t INPUT_SIZE =
      State_Jacobian_U_Matrix_Type_In::COLS;
  static constexpr std::size_t OUTPUT_SIZE =
      Measurement_Jacobian_X_Matrix_Type_In::ROWS;

  // static constexpr std::size_t NP = Np_In;
  // To avoid ODR violation in C++11/14, use enum hack.
  enum : std::size_t { NP = Np_In };

  static constexpr std::size_t STATE_JACOBIAN_X_ROWS = STATE_SIZE;
  static constexpr std::size_t STATE_JACOBIAN_X_COLS = STATE_SIZE;

  static constexpr std::size_t STATE_JACOBIAN_U_ROWS = STATE_SIZE;
  static constexpr std::size_t STATE_JACOBIAN_U_COLS = INPUT_SIZE;

  static constexpr std::size_t MEASUREMENT_JACOBIAN_X_ROWS = OUTPUT_SIZE;
  static constexpr std::size_t MEASUREMENT_JACOBIAN_X_COLS = STATE_SIZE;

  static constexpr std::size_t STATE_HESSIAN_XX_ROWS = STATE_SIZE * STATE_SIZE;
  static constexpr std::size_t STATE_HESSIAN_XX_COLS = STATE_SIZE;

  static constexpr std::size_t STATE_HESSIAN_XU_ROWS = STATE_SIZE * STATE_SIZE;
  static constexpr std::size_t STATE_HESSIAN_XU_COLS = INPUT_SIZE;

  static constexpr std::size_t STATE_HESSIAN_UX_ROWS = INPUT_SIZE * STATE_SIZE;
  static constexpr std::size_t STATE_HESSIAN_UX_COLS = STATE_SIZE;

  static constexpr std::size_t STATE_HESSIAN_UU_ROWS = INPUT_SIZE * STATE_SIZE;
  static constexpr std::size_t STATE_HESSIAN_UU_COLS = INPUT_SIZE;

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

  static_assert(U_Min_Type_In::ROWS == INPUT_SIZE,
                "U_Min_Type_In::ROWS != INPUT_SIZE");
  static_assert(U_Min_Type_In::COLS == 1, "U_Min_Type_In::COLS != 1");

  static_assert(U_Max_Type_In::ROWS == INPUT_SIZE,
                "U_Max_Type_In::ROWS != INPUT_SIZE");
  static_assert(U_Max_Type_In::COLS == 1, "U_Max_Type_In::COLS != 1");

  static_assert(Y_Min_Type_In::ROWS == OUTPUT_SIZE,
                "Y_Min_Type_In::ROWS != OUTPUT_SIZE");
  static_assert(Y_Min_Type_In::COLS == 1, "Y_Min_Type_In::COLS != 1");

  static_assert(Y_Max_Type_In::ROWS == OUTPUT_SIZE,
                "Y_Max_Type_In::ROWS != OUTPUT_SIZE");
  static_assert(Y_Max_Type_In::COLS == 1, "Y_Max_Type_In::COLS != 1");

  static_assert(State_Jacobian_X_Matrix_Type_In::ROWS == STATE_JACOBIAN_X_ROWS,
                "State_Jacobian_X_Matrix_Type::ROWS != STATE_JACOBIAN_X_ROWS");
  static_assert(State_Jacobian_X_Matrix_Type_In::COLS == STATE_JACOBIAN_X_COLS,
                "State_Jacobian_X_Matrix_Type::COLS != STATE_JACOBIAN_X_COLS");

  static_assert(State_Jacobian_U_Matrix_Type_In::ROWS == STATE_JACOBIAN_U_ROWS,
                "State_Jacobian_U_Matrix_Type::ROWS != STATE_JACOBIAN_U_ROWS");
  static_assert(State_Jacobian_U_Matrix_Type_In::COLS == STATE_JACOBIAN_U_COLS,
                "State_Jacobian_U_Matrix_Type::COLS != STATE_JACOBIAN_U_COLS");

  static_assert(Measurement_Jacobian_X_Matrix_Type_In::ROWS ==
                    MEASUREMENT_JACOBIAN_X_ROWS,
                "Measurement_Jacobian_X_Matrix_Type::ROWS != "
                "MEASUREMENT_JACOBIAN_X_ROWS");
  static_assert(Measurement_Jacobian_X_Matrix_Type_In::COLS ==
                    MEASUREMENT_JACOBIAN_X_COLS,
                "Measurement_Jacobian_X_Matrix_Type::COLS != "
                "MEASUREMENT_JACOBIAN_X_COLS");

  static_assert(State_Hessian_XX_Matrix_Type_In::ROWS == STATE_HESSIAN_XX_ROWS,
                "State_Hessian_XX_Matrix_Type::ROWS != STATE_HESSIAN_XX_ROWS");
  static_assert(State_Hessian_XX_Matrix_Type_In::COLS == STATE_HESSIAN_XX_COLS,
                "State_Hessian_XX_Matrix_Type::COLS != STATE_HESSIAN_XX_COLS");

  static_assert(State_Hessian_XU_Matrix_Type_In::ROWS == STATE_HESSIAN_XU_ROWS,
                "State_Hessian_XU_Matrix_Type::ROWS != STATE_HESSIAN_XU_ROWS");
  static_assert(State_Hessian_XU_Matrix_Type_In::COLS == STATE_HESSIAN_XU_COLS,
                "State_Hessian_XU_Matrix_Type::COLS != STATE_HESSIAN_XU_COLS");

  static_assert(State_Hessian_UX_Matrix_Type_In::ROWS == STATE_HESSIAN_UX_ROWS,
                "State_Hessian_UX_Matrix_Type::ROWS != STATE_HESSIAN_UX_ROWS");
  static_assert(State_Hessian_UX_Matrix_Type_In::COLS == STATE_HESSIAN_UX_COLS,
                "State_Hessian_UX_Matrix_Type::COLS != STATE_HESSIAN_UX_COLS");

  static_assert(State_Hessian_UU_Matrix_Type_In::ROWS == STATE_HESSIAN_UU_ROWS,
                "State_Hessian_UU_Matrix_Type::ROWS != STATE_HESSIAN_UU_ROWS");
  static_assert(State_Hessian_UU_Matrix_Type_In::COLS == STATE_HESSIAN_UU_COLS,
                "State_Hessian_UU_Matrix_Type::COLS != STATE_HESSIAN_UU_COLS");

  static_assert(Measurement_Hessian_XX_Matrix_Type_In::ROWS ==
                    MEASUREMENT_HESSIAN_XX_COLS,
                "Measurement_Hessian_XX_Matrix_Type::ROWS != "
                "MEASUREMENT_HESSIAN_XX_COLS");
  static_assert(Measurement_Hessian_XX_Matrix_Type_In::COLS ==
                    MEASUREMENT_HESSIAN_XX_ROWS,
                "Measurement_Hessian_XX_Matrix_Type::COLS != "
                "MEASUREMENT_HESSIAN_XX_ROWS");

protected:
  /* Type */
  using T_ = T;
  using Parameter_Type_ = Parameter_Type_In;

  using Qx_Type_ = PythonNumpy::DiagMatrix_Type<T_, STATE_SIZE>;
  using R_Type_ = PythonNumpy::DiagMatrix_Type<T_, INPUT_SIZE>;
  using Qy_Type_ = PythonNumpy::DiagMatrix_Type<T_, OUTPUT_SIZE>;

  using U_Min_Type_ = U_Min_Type_In;
  using U_Max_Type_ = U_Max_Type_In;
  using Y_Min_Type_ = Y_Min_Type_In;
  using Y_Max_Type_ = Y_Max_Type_In;

  using U_Min_Matrix_Type_ = PythonNumpy::Tile_Type<1, NP, U_Min_Type_In>;
  using U_Max_Matrix_Type_ = PythonNumpy::Tile_Type<1, NP, U_Max_Type_In>;
  using Y_Min_Matrix_Type_ = PythonNumpy::Tile_Type<1, (NP + 1), Y_Min_Type_In>;
  using Y_Max_Matrix_Type_ = PythonNumpy::Tile_Type<1, (NP + 1), Y_Max_Type_In>;

  using StateEquation_Out_Type_ = X_Type;
  using MeasurementEquation_Out_Type_ = Y_Type;

  using StateEquationJacobian_X_Out_Type_ = State_Jacobian_X_Matrix_Type_In;
  using StateEquationJacobian_U_Out_Type_ = State_Jacobian_U_Matrix_Type_In;
  using MeasurementEquationJacobian_X_Out_Type_ =
      Measurement_Jacobian_X_Matrix_Type_In;

  using StateEquationHessian_XX_Out_Type_ = State_Hessian_XX_Matrix_Type_In;
  using StateEquationHessian_XU_Out_Type_ = State_Hessian_XU_Matrix_Type_In;
  using StateEquationHessian_UX_Out_Type_ = State_Hessian_UX_Matrix_Type_In;
  using StateEquationHessian_UU_Out_Type_ = State_Hessian_UU_Matrix_Type_In;
  using MeasurementEquationHessian_XX_Out_Type_ =
      Measurement_Hessian_XX_Matrix_Type_In;

  using StateEquation_Object_ =
      StateEquation_Object<X_Type, U_Type, Parameter_Type_>;
  using MeasurementEquation_Object_ =
      MeasurementEquation_Object<Y_Type, X_Type, U_Type, Parameter_Type_>;

  using StateEquationJacobian_X_Object_ =
      StateEquationJacobian_X_Object<StateEquationJacobian_X_Out_Type_, X_Type,
                                     U_Type, Parameter_Type_>;
  using StateEquationJacobian_U_Object_ =
      StateEquationJacobian_U_Object<StateEquationJacobian_U_Out_Type_, X_Type,
                                     U_Type, Parameter_Type_>;
  using MeasurementEquationJacobian_X_Object_ =
      MeasurementEquationJacobian_X_Object<
          MeasurementEquationJacobian_X_Out_Type_, X_Type, U_Type,
          Parameter_Type_>;

  using StateEquationHessian_XX_Object_ =
      StateEquationHessian_XX_Object<StateEquationHessian_XX_Out_Type_, X_Type,
                                     U_Type, Parameter_Type_>;
  using StateEquationHessian_XU_Object_ =
      StateEquationHessian_XU_Object<StateEquationHessian_XU_Out_Type_, X_Type,
                                     U_Type, Parameter_Type_>;
  using StateEquationHessian_UX_Object_ =
      StateEquationHessian_UX_Object<StateEquationHessian_UX_Out_Type_, X_Type,
                                     U_Type, Parameter_Type_>;
  using StateEquationHessian_UU_Object_ =
      StateEquationHessian_UU_Object<StateEquationHessian_UU_Out_Type_, X_Type,
                                     U_Type, Parameter_Type_>;
  using MeasurementEquationHessian_XX_Object_ =
      MeasurementEquationHessian_XX_Object<
          MeasurementEquationHessian_XX_Out_Type_, X_Type, U_Type,
          Parameter_Type_>;

  using Reference_Trajectory_Type_ = Y_Horizon_Type;

  using Gradient_Type_ = U_Horizon_Type;
  using V_Horizon_Type_ = U_Horizon_Type;
  using HVP_Type_ = U_Horizon_Type;

public:
  /* Constructor */
  SQP_CostMatrices_NMPC()
      : Y_min_max_rho_(), Y_offset_(), Qx_(), R_(), Qy_(), Px_(), Py_(),
        U_min_matrix_(), U_max_matrix_(), Y_min_matrix_(), Y_max_matrix_(),
        _state_equation(), _measurement_equation(),
        _state_equation_jacobian_x(), _state_equation_jacobian_u(),
        _measurement_equation_jacobian_x(), _state_equation_hessian_xx(),
        _state_equation_hessian_xu(), _state_equation_hessian_ux(),
        _state_equation_hessian_uu(), _measurement_equation_hessian_xx() {}

  SQP_CostMatrices_NMPC(const Qx_Type_ &Qx, const R_Type_ &R,
                        const Qy_Type_ &Qy, const U_Min_Type &U_min,
                        const U_Max_Type &U_max, const Y_Min_Type &Y_min,
                        const Y_Max_Type &Y_max)
      : Y_min_max_rho_(static_cast<T_>(Y_MIN_MAX_RHO_FACTOR_DEFAULT)),
        Y_offset_(), Qx_(Qx), R_(R), Qy_(Qy), Px_(Qx), Py_(Qy), U_min_matrix_(),
        U_max_matrix_(), Y_min_matrix_(), Y_max_matrix_(), _state_equation(),
        _measurement_equation(), _state_equation_jacobian_x(),
        _state_equation_jacobian_u(), _measurement_equation_jacobian_x(),
        _state_equation_hessian_xx(), _state_equation_hessian_xu(),
        _state_equation_hessian_ux(), _state_equation_hessian_uu(),
        _measurement_equation_hessian_xx() {

    this->set_U_min(U_min);
    this->set_U_max(U_max);
    this->set_Y_min(Y_min);
    this->set_Y_max(Y_max);
  }

  /* Copy Constructor */
  SQP_CostMatrices_NMPC(const SQP_CostMatrices_NMPC &input)
      : state_space_parameters(input.state_space_parameters),
        reference_trajectory(input.reference_trajectory),
        Y_min_max_rho_(input.Y_min_max_rho_), Y_offset_(input.Y_offset_),
        Qx_(input.Qx_), R_(input.R_), Qy_(input.Qy_), Px_(input.Px_),
        Py_(input.Py_), U_min_matrix_(input.U_min_matrix_),
        U_max_matrix_(input.U_max_matrix_), Y_min_matrix_(input.Y_min_matrix_),
        Y_max_matrix_(input.Y_max_matrix_),
        _state_equation(input._state_equation),
        _measurement_equation(input._measurement_equation),
        _state_equation_jacobian_x(input._state_equation_jacobian_x),
        _state_equation_jacobian_u(input._state_equation_jacobian_u),
        _measurement_equation_jacobian_x(
            input._measurement_equation_jacobian_x),
        _state_equation_hessian_xx(input._state_equation_hessian_xx),
        _state_equation_hessian_xu(input._state_equation_hessian_xu),
        _state_equation_hessian_ux(input._state_equation_hessian_ux),
        _state_equation_hessian_uu(input._state_equation_hessian_uu),
        _measurement_equation_hessian_xx(
            input._measurement_equation_hessian_xx) {}

  SQP_CostMatrices_NMPC &operator=(const SQP_CostMatrices_NMPC &input) {
    if (this != &input) {
      this->state_space_parameters = input.state_space_parameters;
      this->reference_trajectory = input.reference_trajectory;
      this->Y_min_max_rho_ = input.Y_min_max_rho_;
      this->Y_offset_ = input.Y_offset_;

      this->Qx_ = input.Qx_;
      this->R_ = input.R_;
      this->Qy_ = input.Qy_;
      this->Px_ = input.Px_;
      this->Py_ = input.Py_;

      this->U_min_matrix_ = input.U_min_matrix_;
      this->U_max_matrix_ = input.U_max_matrix_;
      this->Y_min_matrix_ = input.Y_min_matrix_;
      this->Y_max_matrix_ = input.Y_max_matrix_;

      this->_state_equation = input._state_equation;
      this->_measurement_equation = input._measurement_equation;

      this->_state_equation_jacobian_x = input._state_equation_jacobian_x;
      this->_state_equation_jacobian_u = input._state_equation_jacobian_u;
      this->_measurement_equation_jacobian_x =
          input._measurement_equation_jacobian_x;

      this->_state_equation_hessian_xx = input._state_equation_hessian_xx;
      this->_state_equation_hessian_xu = input._state_equation_hessian_xu;
      this->_state_equation_hessian_ux = input._state_equation_hessian_ux;
      this->_state_equation_hessian_uu = input._state_equation_hessian_uu;
      this->_measurement_equation_hessian_xx =
          input._measurement_equation_hessian_xx;
    }
    return *this;
  }

  /* Move Constructor */
  SQP_CostMatrices_NMPC(SQP_CostMatrices_NMPC &&input) noexcept
      : state_space_parameters(std::move(input.state_space_parameters)),
        reference_trajectory(std::move(input.reference_trajectory)),
        Y_min_max_rho_(input.Y_min_max_rho_),
        Y_offset_(std::move(input.Y_offset_)), Qx_(std::move(input.Qx_)),
        R_(std::move(input.R_)), Qy_(std::move(input.Qy_)),
        Px_(std::move(input.Px_)), Py_(std::move(input.Py_)),
        U_min_matrix_(std::move(input.U_min_matrix_)),
        U_max_matrix_(std::move(input.U_max_matrix_)),
        Y_min_matrix_(std::move(input.Y_min_matrix_)),
        Y_max_matrix_(std::move(input.Y_max_matrix_)),
        _state_equation(std::move(input._state_equation)),
        _measurement_equation(std::move(input._measurement_equation)),
        _state_equation_jacobian_x(std::move(input._state_equation_jacobian_x)),
        _state_equation_jacobian_u(std::move(input._state_equation_jacobian_u)),
        _measurement_equation_jacobian_x(
            std::move(input._measurement_equation_jacobian_x)),
        _state_equation_hessian_xx(std::move(input._state_equation_hessian_xx)),
        _state_equation_hessian_xu(std::move(input._state_equation_hessian_xu)),
        _state_equation_hessian_ux(std::move(input._state_equation_hessian_ux)),
        _state_equation_hessian_uu(std::move(input._state_equation_hessian_uu)),
        _measurement_equation_hessian_xx(
            std::move(input._measurement_equation_hessian_xx)) {}

  SQP_CostMatrices_NMPC &operator=(SQP_CostMatrices_NMPC &&input) noexcept {
    if (this != &input) {
      this->state_space_parameters = std::move(input.state_space_parameters);
      this->reference_trajectory = std::move(input.reference_trajectory);
      this->Y_min_max_rho_ = input.Y_min_max_rho_;
      this->Y_offset_ = std::move(input.Y_offset_);

      this->Qx_ = std::move(input.Qx_);
      this->R_ = std::move(input.R_);
      this->Qy_ = std::move(input.Qy_);
      this->Px_ = std::move(input.Px_);
      this->Py_ = std::move(input.Py_);

      this->U_min_matrix_ = std::move(input.U_min_matrix_);
      this->U_max_matrix_ = std::move(input.U_max_matrix_);
      this->Y_min_matrix_ = std::move(input.Y_min_matrix_);
      this->Y_max_matrix_ = std::move(input.Y_max_matrix_);

      this->_state_equation = std::move(input._state_equation);
      this->_measurement_equation = std::move(input._measurement_equation);

      this->_state_equation_jacobian_x =
          std::move(input._state_equation_jacobian_x);
      this->_state_equation_jacobian_u =
          std::move(input._state_equation_jacobian_u);
      this->_measurement_equation_jacobian_x =
          std::move(input._measurement_equation_jacobian_x);

      this->_state_equation_hessian_xx =
          std::move(input._state_equation_hessian_xx);
      this->_state_equation_hessian_xu =
          std::move(input._state_equation_hessian_xu);
      this->_state_equation_hessian_ux =
          std::move(input._state_equation_hessian_ux);
      this->_state_equation_hessian_uu =
          std::move(input._state_equation_hessian_uu);
      this->_measurement_equation_hessian_xx =
          std::move(input._measurement_equation_hessian_xx);
    }
    return *this;
  }

public:
  /* Setters */
  /**
   * @brief Sets the function objects required for optimization computations.
   *
   * This method assigns the provided state and measurement equation objects,
   * along with their respective Jacobians and Hessians, to the internal members
   * of the class. These function objects are used in various optimization
   * routines, such as SQP (Sequential Quadratic Programming).
   *
   * @param state_equation The state transition function object.
   * @param measurement_equation The measurement equation object.
   * @param state_equation_jacobian_x Jacobian of the state equation with
   * respect to state variables.
   * @param state_equation_jacobian_u Jacobian of the state equation with
   * respect to control variables.
   * @param measurement_equation_jacobian_x Jacobian of the measurement equation
   * with respect to state variables.
   * @param state_equation_hessian_xx Hessian of the state equation with respect
   * to state variables (XX).
   * @param state_equation_hessian_xu Hessian of the state equation with respect
   * to state and control variables (XU).
   * @param state_equation_hessian_ux Hessian of the state equation with respect
   * to control and state variables (UX).
   * @param state_equation_hessian_uu Hessian of the state equation with respect
   * to control variables (UU).
   * @param measurement_equation_hessian_xx Hessian of the measurement equation
   * with respect to state variables (XX).
   */
  inline void set_function_objects(
      const StateEquation_Object_ &state_equation,
      const MeasurementEquation_Object_ &measurement_equation,
      const StateEquationJacobian_X_Object_ &state_equation_jacobian_x,
      const StateEquationJacobian_U_Object_ &state_equation_jacobian_u,
      const MeasurementEquationJacobian_X_Object_
          &measurement_equation_jacobian_x,
      const StateEquationHessian_XX_Object_ &state_equation_hessian_xx,
      const StateEquationHessian_XU_Object_ &state_equation_hessian_xu,
      const StateEquationHessian_UX_Object_ &state_equation_hessian_ux,
      const StateEquationHessian_UU_Object_ &state_equation_hessian_uu,
      const MeasurementEquationHessian_XX_Object_
          &measurement_equation_hessian_xx) {

    this->_state_equation = state_equation;
    this->_measurement_equation = measurement_equation;

    this->_state_equation_jacobian_x = state_equation_jacobian_x;
    this->_state_equation_jacobian_u = state_equation_jacobian_u;
    this->_measurement_equation_jacobian_x = measurement_equation_jacobian_x;

    this->_state_equation_hessian_xx = state_equation_hessian_xx;
    this->_state_equation_hessian_xu = state_equation_hessian_xu;
    this->_state_equation_hessian_ux = state_equation_hessian_ux;
    this->_state_equation_hessian_uu = state_equation_hessian_uu;
    this->_measurement_equation_hessian_xx = measurement_equation_hessian_xx;
  }

  inline void set_U_min(const U_Min_Type_ &U_min) {

    PythonNumpy::update_tile_concatenated_matrix<1, NP, U_Min_Type_>(
        this->U_min_matrix_, U_min);
  }

  inline void set_U_max(const U_Max_Type_ &U_max) {

    PythonNumpy::update_tile_concatenated_matrix<1, NP, U_Max_Type_>(
        this->U_max_matrix_, U_max);
  }

  inline void set_Y_min(const Y_Min_Type_ &Y_min) {

    PythonNumpy::update_tile_concatenated_matrix<1, (NP + 1), Y_Min_Type_>(
        this->Y_min_matrix_, Y_min);
  }

  inline void set_Y_max(const Y_Max_Type_ &Y_max) {

    PythonNumpy::update_tile_concatenated_matrix<1, (NP + 1), Y_Max_Type_>(
        this->Y_max_matrix_, Y_max);
  }

  inline void set_Y_offset(Y_Type Y_offset) { this->Y_offset_ = Y_offset; }

  inline void set_Qx(const Qx_Type_ &Qx) {
    this->Qx_ = Qx;
    this->Px_ = Qx;
  }

  inline void set_R(const R_Type_ &R) { this->R_ = R; }

  inline void set_Qy(const Qy_Type_ &Qy) {
    this->Qy_ = Qy;
    this->Py_ = Qy;
  }

  inline void set_Y_min_max_rho(const T_ &Y_min_max_rho) {
    this->Y_min_max_rho_ = Y_min_max_rho;
  }

  /* Getters */
  inline U_Min_Matrix_Type_ get_U_min_matrix(void) const {
    return this->U_min_matrix_;
  }

  inline U_Max_Matrix_Type_ get_U_max_matrix(void) const {
    return this->U_max_matrix_;
  }

  inline Y_Min_Matrix_Type_ get_Y_min_matrix(void) const {
    return this->Y_min_matrix_;
  }

  inline Y_Max_Matrix_Type_ get_Y_max_matrix(void) const {
    return this->Y_max_matrix_;
  }

  inline Qx_Type_ get_Qx(void) const { return this->Qx_; }

  inline R_Type_ get_R(void) const { return this->R_; }

  inline Qy_Type_ get_Qy(void) const { return this->Qy_; }

  /* Function */
  /**
   * @brief Computes the second derivative of the objective function with
   * respect to the state variables.
   *
   * This function returns twice the value of the internal state weighting
   * matrix (Qx_). The input parameters X and U are not used in the computation.
   *
   * @param X State variables (unused).
   * @param U Control variables (unused).
   * @return Qx_Type_ The result of 2 * Qx_.
   */
  inline auto l_xx(const X_Type &X, const U_Type &U) -> Qx_Type_ {
    static_cast<void>(X);
    static_cast<void>(U);

    return static_cast<T_>(2) * this->Qx_;
  }

  /**
   * @brief Computes the second derivative of the Lagrangian with respect to the
   * control variable U.
   *
   * This function ignores its input parameters X and U, and returns twice the
   * value of the member variable R_, cast to type T_. It is typically used in
   * optimization routines where the Hessian with respect to U is constant.
   *
   * @tparam X_Type Type of the state variable X.
   * @tparam U_Type Type of the control variable U.
   * @tparam R_Type_ Return type of the function.
   * @tparam T_ Type used for casting the result.
   *
   * @param X State variable (unused).
   * @param U Control variable (unused).
   * @return R_Type_ Twice the value of R_, cast to type T_.
   */
  inline auto l_uu(const X_Type &X, const U_Type &U) -> R_Type_ {
    static_cast<void>(X);
    static_cast<void>(U);

    return static_cast<T_>(2) * this->R_;
  }

  /**
   * @brief Creates and returns an empty sparse matrix of type
   * PythonNumpy::SparseMatrixEmpty_Type.
   *
   * This function generates an empty sparse matrix with the specified template
   * parameters. The input arguments X and U are provided for interface
   * consistency but are not used in the function.
   *
   * @tparam T_         The data type of the matrix elements.
   * @tparam STATE_SIZE Number of columns in the matrix.
   * @tparam INPUT_SIZE Number of rows in the matrix.
   * @param X           State vector (unused).
   * @param U           Input vector (unused).
   * @return PythonNumpy::SparseMatrixEmpty_Type<T_, STATE_SIZE, INPUT_SIZE> An
   * empty sparse matrix.
   */
  inline auto l_xu(const X_Type &X, const U_Type &U)
      -> PythonNumpy::SparseMatrixEmpty_Type<T_, STATE_SIZE, INPUT_SIZE> {
    static_cast<void>(X);
    static_cast<void>(U);

    return PythonNumpy::make_SparseMatrixEmpty<T_, STATE_SIZE, INPUT_SIZE>();
  }

  /**
   * @brief Creates an empty sparse matrix representing the partial derivative
   * of the cost function with respect to the state (X) and input (U) variables.
   *
   * This function returns a sparse matrix of type
   * PythonNumpy::SparseMatrixEmpty_Type with template parameters T_,
   * INPUT_SIZE, and STATE_SIZE. The input arguments X and U are not used in the
   * computation and are only present to match the expected function signature.
   *
   * @tparam T_         The data type of the matrix elements.
   * @tparam INPUT_SIZE The number of input variables (rows).
   * @tparam STATE_SIZE The number of state variables (cols).
   * @param X           The state vector (unused).
   * @param U           The input vector (unused).
   * @return PythonNumpy::SparseMatrixEmpty_Type<T_, INPUT_SIZE, STATE_SIZE>
   *         An empty sparse matrix of the specified type and dimensions.
   */
  inline auto l_ux(const X_Type &X, const U_Type &U)
      -> PythonNumpy::SparseMatrixEmpty_Type<T_, INPUT_SIZE, STATE_SIZE> {
    static_cast<void>(X);
    static_cast<void>(U);

    return PythonNumpy::make_SparseMatrixEmpty<T_, INPUT_SIZE, STATE_SIZE>();
  }

  /**
   * @brief Calculates the state equation using the provided inputs.
   *
   * This function evaluates the internal state equation with the given state
   * vector `X`, control vector `U`, and parameter set `parameter`, returning
   * the computed state.
   *
   * @param X The current state vector.
   * @param U The control input vector.
   * @param parameter The set of parameters required for the state equation.
   * @return StateEquation_Out_Type_ The result of the state equation
   * evaluation.
   */
  inline auto calculate_state_equation(const X_Type &X, const U_Type &U,
                                       const Parameter_Type_ &parameter)
      -> StateEquation_Out_Type_ {

    return this->_state_equation(X, U, parameter);
  }

  /**
   * @brief Calculates the measurement equation using the provided state, input,
   * and parameters.
   *
   * This function invokes the internal measurement equation with the given
   * arguments to compute the output of the measurement model.
   *
   * @param X The current state vector.
   * @param U The current input/control vector.
   * @param parameter The parameter set required for the measurement equation.
   * @return The output of the measurement equation.
   */
  inline auto calculate_measurement_equation(const X_Type &X, const U_Type &U,
                                             const Parameter_Type_ &parameter)
      -> MeasurementEquation_Out_Type_ {

    return this->_measurement_equation(X, U, parameter);
  }

  /**
   * @brief Calculates the Jacobian of the state equation with respect to the
   * state vector X.
   *
   * This function computes the partial derivatives of the state equation with
   * respect to the state variables, given the current state vector X, control
   * input vector U, and system parameters.
   *
   * @param X The current state vector.
   * @param U The current control input vector.
   * @param parameter The system parameters required for the computation.
   * @return The Jacobian matrix of the state equation with respect to X.
   */
  inline auto calculate_state_jacobian_x(const X_Type &X, const U_Type &U,
                                         const Parameter_Type_ &parameter)
      -> StateEquationJacobian_X_Out_Type_ {

    return this->_state_equation_jacobian_x(X, U, parameter);
  }

  /**
   * @brief Calculates the Jacobian of the state equation with respect to the
   * control input.
   *
   * This function computes the partial derivatives of the state equation with
   * respect to the control input `U`, given the current state `X`, control
   * input `U`, and system parameters `parameter`.
   *
   * @param X The current state vector.
   * @param U The current control input vector.
   * @param parameter The system parameters required for the Jacobian
   * calculation.
   * @return The Jacobian matrix of the state equation with respect to the
   * control input.
   */
  inline auto calculate_state_jacobian_u(const X_Type &X, const U_Type &U,
                                         const Parameter_Type_ &parameter)
      -> StateEquationJacobian_U_Out_Type_ {

    return this->_state_equation_jacobian_u(X, U, parameter);
  }

  /**
   * @brief Calculates the Jacobian of the measurement equation with respect to
   * the state vector X.
   *
   * This function computes the partial derivatives of the measurement equation
   * with respect to the state variables, given the current state X, input U,
   * and model parameters. It delegates the actual computation to the
   * _measurement_equation_jacobian_x member function.
   *
   * @param X The current state vector.
   * @param U The current input/control vector.
   * @param parameter The model parameters required for the Jacobian
   * calculation.
   * @return The Jacobian matrix of the measurement equation with respect to X.
   */
  inline auto calculate_measurement_jacobian_x(const X_Type &X, const U_Type &U,
                                               const Parameter_Type_ &parameter)
      -> MeasurementEquationJacobian_X_Out_Type_ {

    return this->_measurement_equation_jacobian_x(X, U, parameter);
  }

  /**
   * @brief Contracts the Hessian of the state equation with the given direction
   * and lambda vector.
   *
   * This function computes the contraction of the Hessian of the state equation
   * with respect to the state variables (X), using the provided direction
   * vector (dX) and the lambda vector (lam_next). The contraction is performed
   * by first obtaining the Hessian matrix via `_state_equation_hessian_xx`, and
   * then applying the contraction operation using
   * `MatrixOperation::compute_fxx_lambda_contract`.
   *
   * @tparam Lambda_Vector_Type Type of the lambda vector, must be a row vector
   * with STATE_SIZE rows.
   * @param X Current state vector.
   * @param U Current control vector.
   * @param parameter Additional parameters required for the state equation.
   * @param lam_next Lambda vector for the next time step (row vector of size
   * STATE_SIZE).
   * @param dX Direction vector for contraction.
   * @return X_Type Result of the contraction operation.
   *
   * @note Lambda_Vector_Type must have ROWS == STATE_SIZE and COLS == 1.
   */
  template <typename Lambda_Vector_Type>
  inline auto fx_xx_lambda_contract(const X_Type &X, const U_Type &U,
                                    const Parameter_Type_ &parameter,
                                    const Lambda_Vector_Type &lam_next,
                                    const X_Type &dX) -> X_Type {

    static_assert(Lambda_Vector_Type::ROWS == STATE_SIZE,
                  "Lambda_Vector_Type::ROWS != STATE_SIZE");
    static_assert(Lambda_Vector_Type::COLS == 1,
                  "Lambda_Vector_Type::COLS != 1");

    auto Hf_xx = this->_state_equation_hessian_xx(X, U, parameter);

    X_Type out;

    MatrixOperation::compute_fxx_lambda_contract(Hf_xx, dX, lam_next, out);

    return out;
  }

  /**
   * @brief Contracts the Hessian of the state equation with the input direction
   * and lambda vector.
   *
   * This function computes the contraction of the Hessian of the state equation
   * with respect to state and input variables (`Hf_xu`), the input direction
   * (`dU`), and the lambda vector (`lam_next`). The result is stored in an
   * output state vector and returned.
   *
   * @tparam Lambda_Vector_Type Type of the lambda vector, must have ROWS ==
   * STATE_SIZE and COLS == 1.
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
                                    const Parameter_Type_ &parameter,
                                    const Lambda_Vector_Type &lam_next,
                                    const U_Type &dU) -> X_Type {

    static_assert(Lambda_Vector_Type::ROWS == STATE_SIZE,
                  "Lambda_Vector_Type::ROWS != STATE_SIZE");
    static_assert(Lambda_Vector_Type::COLS == 1,
                  "Lambda_Vector_Type::COLS != 1");

    auto Hf_xu = this->_state_equation_hessian_xu(X, U, parameter);

    X_Type out;

    MatrixOperation::compute_fx_xu_lambda_contract(Hf_xu, dU, lam_next, out);

    return out;
  }

  /**
   * @brief Contracts the Hessian of the state equation with the provided
   * direction and lambda vector.
   *
   * This function computes the contraction of the Hessian of the state equation
   * with respect to state and control variables (`Hf_ux`), the direction vector
   * `dX`, and the next lambda vector `lam_next`. The result is stored in an
   * output variable of type `U_Type`.
   *
   * @tparam Lambda_Vector_Type Type of the lambda vector, must be a row vector
   * with STATE_SIZE rows.
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
                                    const Parameter_Type_ &parameter,
                                    const Lambda_Vector_Type &lam_next,
                                    const X_Type &dX) -> U_Type {

    static_assert(Lambda_Vector_Type::ROWS == STATE_SIZE,
                  "Lambda_Vector_Type::ROWS != STATE_SIZE");
    static_assert(Lambda_Vector_Type::COLS == 1,
                  "Lambda_Vector_Type::COLS != 1");

    auto Hf_ux = this->_state_equation_hessian_ux(X, U, parameter);

    U_Type out;

    MatrixOperation::compute_fu_xx_lambda_contract(Hf_ux, dX, lam_next, out);

    return out;
  }

  /**
   * @brief Contracts the Hessian of the state equation with the input direction
   * and lambda vector.
   *
   * This function computes the contraction of the Hessian of the state equation
   * with respect to the input variables (`U`) using the provided state (`X`),
   * input (`U`), parameters (`parameter`), next-step lambda vector
   * (`lam_next`), and input direction (`dU`). The contraction is performed by
   * calling `MatrixOperation::compute_fu_uu_lambda_contract`, which combines
   * the Hessian, input direction, and lambda vector to produce the output.
   *
   * @tparam Lambda_Vector_Type Type of the lambda vector, must be a row vector
   * with `STATE_SIZE` rows.
   * @param X The current state vector.
   * @param U The current input vector.
   * @param parameter The parameter set for the state equation.
   * @param lam_next The lambda vector for the next time step (row vector of
   * size `STATE_SIZE`).
   * @param dU The direction vector for the input.
   * @return U_Type The result of the contraction operation.
   *
   * @note Compile-time assertions ensure that `lam_next` is a row vector with
   * the correct number of rows.
   */
  template <typename Lambda_Vector_Type>
  inline auto fu_uu_lambda_contract(const X_Type &X, const U_Type &U,
                                    const Parameter_Type_ &parameter,
                                    const Lambda_Vector_Type &lam_next,
                                    const U_Type &dU) -> U_Type {

    static_assert(Lambda_Vector_Type::ROWS == STATE_SIZE,
                  "Lambda_Vector_Type::ROWS != STATE_SIZE");
    static_assert(Lambda_Vector_Type::COLS == 1,
                  "Lambda_Vector_Type::COLS != 1");

    auto Hf_uu = this->_state_equation_hessian_uu(X, U, parameter);

    U_Type out;

    MatrixOperation::compute_fu_uu_lambda_contract(Hf_uu, dU, lam_next, out);

    return out;
  }

  /**
   * @brief Contracts the Hessian of the measurement equation with a direction
   * vector and a weight vector.
   *
   * This function computes the contraction of the Hessian of the measurement
   * function (with respect to X) using the provided direction vector `dX` and
   * the weight vector `weight`. The result is stored in `out` and returned. The
   * function asserts that the weight vector has the correct dimensions.
   *
   * @tparam Weight_Vector_Type Type of the weight vector, must have ROWS ==
   * OUTPUT_SIZE and COLS == 1.
   * @param X The input variable for which the Hessian is computed.
   * @param parameter Additional parameters required for the measurement
   * function.
   * @param weight The weight vector used in the contraction.
   * @param dX The direction vector for contraction.
   * @return X_Type The contracted result.
   */
  template <typename Weight_Vector_Type>
  inline auto hxx_lambda_contract(const X_Type &X,
                                  const Parameter_Type_ &parameter,
                                  const Weight_Vector_Type &weight,
                                  const X_Type &dX) -> X_Type {

    static_assert(Weight_Vector_Type::ROWS == OUTPUT_SIZE,
                  "Weight_Vector_Type::ROWS != OUTPUT_SIZE");
    static_assert(Weight_Vector_Type::COLS == 1,
                  "Weight_Vector_Type::COLS != 1");

    U_Type U;
    auto Hh_xx = this->_measurement_equation_hessian_xx(X, U, parameter);

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
   * is updated using the `calculate_state_equation`, which models the system
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
                                  const Parameter_Type_ &parameter)
      -> X_Horizon_Type {

    X_Horizon_Type X_horizon;
    X_Type X = X_initial;

    MatrixOperation::set_row(X_horizon, X, 0);

    for (std::size_t k = 0; k < NP; k++) {
      auto U = MatrixOperation::get_row(U_horizon, k);
      X = this->calculate_state_equation(X, U, parameter);

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
        Y_horizon, this->Y_min_matrix_, this->Y_max_matrix_, Y_limit_penalty);

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
  inline auto
  calculate_Y_limit_penalty_and_active(const Y_Horizon_Type &Y_horizon)
      -> std::tuple<Y_Horizon_Type, Y_Horizon_Type> {

    Y_Horizon_Type Y_limit_penalty;
    Y_Horizon_Type Y_limit_active;

    MatrixOperation::calculate_Y_limit_penalty_and_active(
        Y_horizon, this->Y_min_matrix_, this->Y_max_matrix_, Y_limit_penalty,
        Y_limit_active);

    return std::make_tuple(Y_limit_penalty, Y_limit_active);
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
   * @return T_ The computed cost value.
   */
  inline auto compute_cost(const X_Type X_initial,
                           const U_Horizon_Type &U_horizon) -> T_ {

    auto X_horizon = this->simulate_trajectory(X_initial, U_horizon,
                                               this->state_space_parameters);

    Y_Horizon_Type Y_horizon;
    PythonNumpy::update_tile_concatenated_matrix<1, (NP + 1), Y_Type>(
        Y_horizon, this->Y_offset_);
    U_Type U_dummy;

    for (std::size_t k = 0; k < (NP + 1); k++) {
      auto Y_k = this->calculate_measurement_equation(
          MatrixOperation::get_row(X_horizon, k), U_dummy,
          this->state_space_parameters);

      MatrixOperation::set_row(Y_horizon, Y_k, k);
    }

    auto Y_limit_penalty = this->calculate_Y_limit_penalty(Y_horizon);

    T_ J = static_cast<T_>(0);
    for (std::size_t k = 0; k < NP; k++) {
      auto e_y_r = MatrixOperation::get_row(Y_horizon, k) -
                   MatrixOperation::get_row(this->reference_trajectory, k);

      auto X_T_Qx_X = MatrixOperation::calculate_quadratic_form(
          MatrixOperation::get_row(X_horizon, k), this->Qx_);
      auto e_y_r_T_Qy_e_y_r =
          MatrixOperation::calculate_quadratic_form(e_y_r, this->Qy_);

      auto U_T_R_U = MatrixOperation::calculate_quadratic_form(
          MatrixOperation::get_row(U_horizon, k), this->R_);

      auto Y_limit_penalty_T_Y_limit_penalty =
          MatrixOperation::calculate_quadratic_no_weighted(
              MatrixOperation::get_row(Y_limit_penalty, k));

      J += X_T_Qx_X + e_y_r_T_Qy_e_y_r + U_T_R_U +
           this->Y_min_max_rho_ * Y_limit_penalty_T_Y_limit_penalty;
    }

    auto eN_y_r = MatrixOperation::get_row(Y_horizon, NP) -
                  MatrixOperation::get_row(this->reference_trajectory, NP);

    auto XN_T_Qx_XN = MatrixOperation::calculate_quadratic_form(
        MatrixOperation::get_row(X_horizon, NP), this->Qx_);
    auto eN_y_r_T_Qy_eN_y_r =
        MatrixOperation::calculate_quadratic_form(eN_y_r, this->Qy_);
    auto YN_limit_penalty_T_YN_limit_penalty =
        MatrixOperation::calculate_quadratic_no_weighted(
            MatrixOperation::get_row(Y_limit_penalty, NP));

    J += XN_T_Qx_XN + eN_y_r_T_Qy_eN_y_r +
         this->Y_min_max_rho_ * YN_limit_penalty_T_YN_limit_penalty;

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
   * @tparam T_ Scalar type used for cost and gradient calculations.
   * @tparam Gradient_Type_ Type representing the gradient vector/matrix.
   *
   * @param X_initial Initial state vector.
   * @param U_horizon Control input horizon matrix.
   *
   * @return std::tuple of (J, gradient), where J is the computed cost value
   * and gradient is the gradient of the cost with respect to control inputs.
   *
   * @note
   * - Assumes that all required system parameters and matrices (e.g., Qx, Qy,
   * R, Px, Py) are initialized.
   * - The function is intended for use in sequential quadratic programming
   * (SQP) or similar optimization routines.
   */
  inline auto compute_cost_and_gradient(const X_Type X_initial,
                                        const U_Horizon_Type &U_horizon)
      -> std::tuple<T_, Gradient_Type_> {

    T_ J;
    Gradient_Type_ gradient;

    U_Type U_dummy;

    auto X_horizon = this->simulate_trajectory(X_initial, U_horizon,
                                               this->state_space_parameters);

    Y_Horizon_Type Y_horizon;
    PythonNumpy::update_tile_concatenated_matrix<1, (NP + 1), Y_Type>(
        Y_horizon, this->Y_offset_);

    for (std::size_t k = 0; k < (NP + 1); k++) {
      auto Y_k = this->calculate_measurement_equation(
          MatrixOperation::get_row(X_horizon, k), U_dummy,
          this->state_space_parameters);

      MatrixOperation::set_row(Y_horizon, Y_k, k);
    }

    auto Y_limit_penalty = this->calculate_Y_limit_penalty(Y_horizon);

    J = static_cast<T_>(0);
    for (std::size_t k = 0; k < NP; k++) {
      auto e_y_r = MatrixOperation::get_row(Y_horizon, k) -
                   MatrixOperation::get_row(this->reference_trajectory, k);

      auto X_T_Qx_X = MatrixOperation::calculate_quadratic_form(
          MatrixOperation::get_row(X_horizon, k), this->Qx_);
      auto e_y_r_T_Qy_e_y_r =
          MatrixOperation::calculate_quadratic_form(e_y_r, this->Qy_);

      auto U_T_R_U = MatrixOperation::calculate_quadratic_form(
          MatrixOperation::get_row(U_horizon, k), this->R_);

      auto Y_limit_penalty_T_Y_limit_penalty =
          MatrixOperation::calculate_quadratic_no_weighted(
              MatrixOperation::get_row(Y_limit_penalty, k));

      J += X_T_Qx_X + e_y_r_T_Qy_e_y_r + U_T_R_U +
           this->Y_min_max_rho_ * Y_limit_penalty_T_Y_limit_penalty;
    }

    auto eN_y_r = MatrixOperation::get_row(Y_horizon, NP) -
                  MatrixOperation::get_row(this->reference_trajectory, NP);

    auto XN_T_Qx_XN = MatrixOperation::calculate_quadratic_form(
        MatrixOperation::get_row(X_horizon, NP), this->Qx_);
    auto eN_y_r_T_Qy_eN_y_r =
        MatrixOperation::calculate_quadratic_form(eN_y_r, this->Qy_);
    auto YN_limit_penalty_T_YN_limit_penalty =
        MatrixOperation::calculate_quadratic_no_weighted(
            MatrixOperation::get_row(Y_limit_penalty, NP));

    J += XN_T_Qx_XN + eN_y_r_T_Qy_eN_y_r +
         this->Y_min_max_rho_ * YN_limit_penalty_T_YN_limit_penalty;

    // terminal adjoint
    auto C_N = this->calculate_measurement_jacobian_x(
        MatrixOperation::get_row(X_horizon, NP), U_dummy,
        this->state_space_parameters);

    auto Px_X = this->Px_ * MatrixOperation::get_row(X_horizon, NP);
    auto Py_eN_y_r = this->Py_ * eN_y_r;
    auto Y_min_max_rho_YN_limit_penalty =
        this->Y_min_max_rho_ * MatrixOperation::get_row(Y_limit_penalty, NP);

    auto lam_next =
        static_cast<T_>(2) *
        (Px_X + PythonNumpy::ATranspose_mul_B(
                    C_N, (Py_eN_y_r + Y_min_max_rho_YN_limit_penalty)));

    gradient = Gradient_Type_();

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

      auto _2_R_U = static_cast<T_>(2) * this->R_ *
                    MatrixOperation::get_row(U_horizon, k);
      auto B_k_T_lam_next = PythonNumpy::ATranspose_mul_B(B_k, lam_next);

      MatrixOperation::set_row(gradient, _2_R_U + B_k_T_lam_next, k);

      auto Qx_X = this->Qx_ * MatrixOperation::get_row(X_horizon, k);
      auto Qy_ek_y = this->Qy_ * ek_y;
      auto Y_min_max_rho_Yk_limit_penalty =
          static_cast<T_>(2) * this->Y_min_max_rho_ *
          MatrixOperation::get_row(Y_limit_penalty, k);

      auto A_k_T_lam_next = PythonNumpy::ATranspose_mul_B(A_k, lam_next);

      lam_next =
          static_cast<T_>(2) *
              (Qx_X + PythonNumpy::ATranspose_mul_B(
                          Cx_k, (Qy_ek_y + Y_min_max_rho_Yk_limit_penalty))) +
          A_k_T_lam_next;
    }

    return std::make_tuple(J, gradient);
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
                           const V_Horizon_Type_ &V_horizon) -> HVP_Type_ {

    U_Type U_dummy;

    /* --- 1) forward states */
    auto X_horizon = this->simulate_trajectory(X_initial, U_horizon,
                                               this->state_space_parameters);

    Y_Horizon_Type Y_horizon;
    for (std::size_t k = 0; k < (NP + 1); k++) {
      auto Y_k = this->calculate_measurement_equation(
          MatrixOperation::get_row(X_horizon, k), U_dummy,
          this->state_space_parameters);

      MatrixOperation::set_row(Y_horizon, Y_k, k);
    }

    auto yN = this->calculate_measurement_equation(
        MatrixOperation::get_row(X_horizon, NP), U_dummy,
        this->state_space_parameters);

    auto eN_y = yN - MatrixOperation::get_row(this->reference_trajectory, NP);

    Y_Horizon_Type Y_limit_penalty;
    Y_Horizon_Type Y_limit_active;
    std::tie(Y_limit_penalty, Y_limit_active) =
        this->calculate_Y_limit_penalty_and_active(Y_horizon);

    // /* --- 2) first-order adjoint (costate lambda) with output terms */
    X_Horizon_Type lam;
    auto Cx_N = this->calculate_measurement_jacobian_x(
        MatrixOperation::get_row(X_horizon, NP), U_dummy,
        this->state_space_parameters);

    auto Px_XN = this->Px_ * MatrixOperation::get_row(X_horizon, NP);
    auto Py_eN_y = this->Py_ * eN_y;
    auto Y_min_max_rho_YN_limit_penalty =
        this->Y_min_max_rho_ * MatrixOperation::get_row(Y_limit_penalty, NP);

    auto lam_input =
        static_cast<T_>(2) *
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

      auto Qx_X = this->Qx_ * MatrixOperation::get_row(X_horizon, k);
      auto Qy_ek_y = this->Qy_ * ek_y;
      auto Y_min_max_rho_Yk_limit_penalty =
          this->Y_min_max_rho_ * MatrixOperation::get_row(Y_limit_penalty, k);
      auto A_k_T_lam = PythonNumpy::ATranspose_mul_B(
          A_k, MatrixOperation::get_row(lam, k + 1));

      auto lam_input_ =
          static_cast<T_>(2) *
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
        Cx_N, (static_cast<T_>(2) * this->Py_ * Cx_N_dx));

    Y_Type YN_limit_active_CX_N_dx;
    PythonNumpy::element_wise_multiply(
        YN_limit_active_CX_N_dx, MatrixOperation::get_row(Y_limit_active, NP),
        Cx_N_dx);

    auto Y_min_max_rho_YN_limit_active_CX_N_dx =
        static_cast<T_>(2) * this->Y_min_max_rho_ * YN_limit_active_CX_N_dx;

    auto CX_N_T_penalty_CX_N_dx = PythonNumpy::ATranspose_mul_B(
        Cx_N, Y_min_max_rho_YN_limit_active_CX_N_dx);

    Y_min_max_rho_YN_limit_penalty =
        static_cast<T_>(2) * this->Y_min_max_rho_ *
        MatrixOperation::get_row(Y_limit_penalty, NP);

    auto Hxx_penalty_term_N = this->hxx_lambda_contract(
        MatrixOperation::get_row(X_horizon, NP), this->state_space_parameters,
        Y_min_max_rho_YN_limit_penalty, MatrixOperation::get_row(dx, NP));

    auto d_lambda_input = l_xx_dx + CX_N_T_Py_Cx_N_dx + Hxx_penalty_term_N +
                          CX_N_T_penalty_CX_N_dx;
    MatrixOperation::set_row(d_lambda, d_lambda_input, NP);

    HVP_Type_ Hu;

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
          Cx_k, (static_cast<T_>(2) * this->Py_ * Cx_dx_k));

      auto term_Qy_hxx = this->hxx_lambda_contract(
          MatrixOperation::get_row(X_horizon, k), this->state_space_parameters,
          static_cast<T_>(2) * this->Py_ * ek_y,
          MatrixOperation::get_row(dx, k));

      Y_Type YN_limit_active_CX_k_dx;
      PythonNumpy::element_wise_multiply(
          YN_limit_active_CX_k_dx, MatrixOperation::get_row(Y_limit_active, k),
          Cx_dx_k);

      auto Y_min_max_rho_Yk_limit_active_Cx_dx_k =
          static_cast<T_>(2) * this->Y_min_max_rho_ * YN_limit_active_CX_k_dx;

      auto term_penalty_GN = PythonNumpy::ATranspose_mul_B(
          Cx_k, Y_min_max_rho_Yk_limit_active_Cx_dx_k);

      auto Y_min_max_rho_Yk_limit_penalty =
          static_cast<T_>(2) * this->Y_min_max_rho_ *
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
  Parameter_Type_ state_space_parameters;
  Reference_Trajectory_Type_ reference_trajectory;

protected:
  /* Variable */
  T_ Y_min_max_rho_;
  Y_Type Y_offset_;

  Qx_Type_ Qx_;
  R_Type_ R_;
  Qy_Type_ Qy_;

  Qx_Type_ Px_;
  Qy_Type_ Py_;

  U_Min_Matrix_Type_ U_min_matrix_;
  U_Max_Matrix_Type_ U_max_matrix_;
  Y_Min_Matrix_Type_ Y_min_matrix_;
  Y_Max_Matrix_Type_ Y_max_matrix_;

  StateEquation_Object_ _state_equation;
  MeasurementEquation_Object_ _measurement_equation;

  StateEquationJacobian_X_Object_ _state_equation_jacobian_x;
  StateEquationJacobian_U_Object_ _state_equation_jacobian_u;
  MeasurementEquationJacobian_X_Object_ _measurement_equation_jacobian_x;

  StateEquationHessian_XX_Object_ _state_equation_hessian_xx;
  StateEquationHessian_XU_Object_ _state_equation_hessian_xu;
  StateEquationHessian_UX_Object_ _state_equation_hessian_ux;
  StateEquationHessian_UU_Object_ _state_equation_hessian_uu;
  MeasurementEquationHessian_XX_Object_ _measurement_equation_hessian_xx;
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
    const PythonNumpy::DiagMatrix_Type<T, State_Jacobian_X_Matrix_Type::COLS>
        &Qx,
    const PythonNumpy::DiagMatrix_Type<T, State_Jacobian_U_Matrix_Type::COLS>
        &R,
    const PythonNumpy::DiagMatrix_Type<
        T, Measurement_Jacobian_X_Matrix_Type::ROWS> &Qy,
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

#endif // PYTHON_OPTIMIZATION_SQP_MATRIX_UTILITY_HPP_
