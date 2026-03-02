/**
 * @file python_optimization_optimization_engine_matrix_utility.hpp
 *
 * @brief Utility definitions for PANOC/ALM cost matrices in Nonlinear MPC
 * (NMPC) optimization.
 *
 * This header provides type definitions, function objects, and the main class
 * for handling cost matrices and related operations in PANOC/ALM-based
 * optimization for Nonlinear Model Predictive Control (NMPC) problems.
 *
 * Unlike the SQP variant (python_optimization_sqp_matrix_utility.hpp), this
 * class does not require Hessian computations, as the PANOC algorithm relies
 * only on first-order gradient information. Additionally, it provides output
 * constraint mapping and its Jacobian transpose for the Augmented Lagrangian
 * Method (ALM).
 */
#ifndef __PYTHON_OPTIMIZATION_OPTIMIZATION_ENGINE_MATRIX_UTILITY_HPP__
#define __PYTHON_OPTIMIZATION_OPTIMIZATION_ENGINE_MATRIX_UTILITY_HPP__

#include "python_optimization_common_matrix_operation.hpp"
#include "python_optimization_matrix_utility_common.hpp"

#include "python_control.hpp"
#include "python_numpy.hpp"

#include <array>
#include <functional>
#include <type_traits>

namespace PythonOptimization {

/* Optimization Engine cost matrices for Nonlinear MPC (PANOC/ALM) */

/**
 * @brief PANOC/ALM Cost Matrices for Nonlinear Model Predictive Control (NMPC)
 *
 * This class encapsulates the cost matrices and related operations for
 * PANOC/ALM-based NMPC problems. It provides interfaces for setting up cost
 * weights, constraints, system and measurement functions, and their Jacobians.
 * The class supports simulation of system trajectories, cost and gradient
 * computation (via adjoint method), and output constraint mapping with its
 * Jacobian transpose for ALM.
 *
 * Unlike SQP_CostMatrices_NMPC, this class does not require second-order
 * derivative information (Hessians).
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
 */
template <typename T, std::size_t Np_In, typename Parameter_Type_In,
          typename U_Min_Type_In, typename U_Max_Type_In,
          typename Y_Min_Type_In, typename Y_Max_Type_In,
          typename State_Jacobian_X_Matrix_Type_In,
          typename State_Jacobian_U_Matrix_Type_In,
          typename Measurement_Jacobian_X_Matrix_Type_In>
class OptimizationEngine_CostMatrices {
public:
  /* Constant */
  static constexpr std::size_t STATE_SIZE =
      State_Jacobian_X_Matrix_Type_In::ROWS;
  static constexpr std::size_t INPUT_SIZE =
      State_Jacobian_U_Matrix_Type_In::ROWS;
  static constexpr std::size_t OUTPUT_SIZE =
      Measurement_Jacobian_X_Matrix_Type_In::COLS;

  // To avoid ODR violation in C++11/14, use enum hack.
  enum : std::size_t { NP = Np_In };

  static constexpr std::size_t STATE_JACOBIAN_X_COLS = STATE_SIZE;
  static constexpr std::size_t STATE_JACOBIAN_X_ROWS = STATE_SIZE;

  static constexpr std::size_t STATE_JACOBIAN_U_COLS = STATE_SIZE;
  static constexpr std::size_t STATE_JACOBIAN_U_ROWS = INPUT_SIZE;

  static constexpr std::size_t MEASUREMENT_JACOBIAN_X_COLS = OUTPUT_SIZE;
  static constexpr std::size_t MEASUREMENT_JACOBIAN_X_ROWS = STATE_SIZE;

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

  using _Reference_Trajectory_Type = Y_Horizon_Type;

  using _Gradient_Type = U_Horizon_Type;

  /* Output mapping types for ALM */
  using _Output_Mapping_Type = Y_Horizon_Type;
  using _Dual_Type = Y_Horizon_Type;

public:
  /* Constructor */
  OptimizationEngine_CostMatrices()
      : _Y_offset(), _Qx(), _R(), _Qy(), _Px(), _Py(), _U_min_matrix(),
        _U_max_matrix(), _Y_min_matrix(), _Y_max_matrix(), _state_function(),
        _measurement_function(), _state_function_jacobian_x(),
        _state_function_jacobian_u(), _measurement_function_jacobian_x() {}

  OptimizationEngine_CostMatrices(const _Qx_Type &Qx, const _R_Type &R,
                                  const _Qy_Type &Qy, const U_Min_Type &U_min,
                                  const U_Max_Type &U_max,
                                  const Y_Min_Type &Y_min,
                                  const Y_Max_Type &Y_max)
      : _Y_offset(), _Qx(Qx), _R(R), _Qy(Qy), _Px(Qx), _Py(Qy), _U_min_matrix(),
        _U_max_matrix(), _Y_min_matrix(), _Y_max_matrix(), _state_function(),
        _measurement_function(), _state_function_jacobian_x(),
        _state_function_jacobian_u(), _measurement_function_jacobian_x() {

    this->set_U_min(U_min);
    this->set_U_max(U_max);
    this->set_Y_min(Y_min);
    this->set_Y_max(Y_max);
  }

  /* Copy Constructor */
  OptimizationEngine_CostMatrices(const OptimizationEngine_CostMatrices &input)
      : X_initial(input.X_initial),
        state_space_parameters(input.state_space_parameters),
        reference_trajectory(input.reference_trajectory),
        _Y_offset(input._Y_offset), _Qx(input._Qx), _R(input._R),
        _Qy(input._Qy), _Px(input._Px), _Py(input._Py),
        _U_min_matrix(input._U_min_matrix), _U_max_matrix(input._U_max_matrix),
        _Y_min_matrix(input._Y_min_matrix), _Y_max_matrix(input._Y_max_matrix),
        _state_function(input._state_function),
        _measurement_function(input._measurement_function),
        _state_function_jacobian_x(input._state_function_jacobian_x),
        _state_function_jacobian_u(input._state_function_jacobian_u),
        _measurement_function_jacobian_x(
            input._measurement_function_jacobian_x) {}

  OptimizationEngine_CostMatrices &
  operator=(const OptimizationEngine_CostMatrices &input) {
    if (this != &input) {
      this->X_initial = input.X_initial;
      this->state_space_parameters = input.state_space_parameters;
      this->reference_trajectory = input.reference_trajectory;
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
    }
    return *this;
  }

  /* Move Constructor */
  OptimizationEngine_CostMatrices(
      OptimizationEngine_CostMatrices &&input) noexcept
      : X_initial(std::move(input.X_initial)),
        state_space_parameters(std::move(input.state_space_parameters)),
        reference_trajectory(std::move(input.reference_trajectory)),
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
            std::move(input._measurement_function_jacobian_x)) {}

  OptimizationEngine_CostMatrices &
  operator=(OptimizationEngine_CostMatrices &&input) noexcept {
    if (this != &input) {
      this->X_initial = std::move(input.X_initial);
      this->state_space_parameters = std::move(input.state_space_parameters);
      this->reference_trajectory = std::move(input.reference_trajectory);
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
    }
    return *this;
  }

public:
  /* Setters */
  /**
   * @brief Sets the function objects required for optimization computations.
   *
   * This method assigns the provided state and measurement function objects,
   * along with their respective Jacobians, to the internal members of the
   * class. These function objects are used in PANOC/ALM optimization routines.
   * No Hessian function objects are needed.
   *
   * @param state_function The state transition function object.
   * @param measurement_function The measurement function object.
   * @param state_function_jacobian_x Jacobian of the state function with
   * respect to state variables.
   * @param state_function_jacobian_u Jacobian of the state function with
   * respect to control variables.
   * @param measurement_function_jacobian_x Jacobian of the measurement function
   * with respect to state variables.
   */
  inline void set_function_objects(
      const _StateFunction_Object &state_function,
      const _MeasurementFunction_Object &measurement_function,
      const _StateFunctionJacobian_X_Object &state_function_jacobian_x,
      const _StateFunctionJacobian_U_Object &state_function_jacobian_u,
      const _MeasurementFunctionJacobian_X_Object
          &measurement_function_jacobian_x) {

    this->_state_function = state_function;
    this->_measurement_function = measurement_function;

    this->_state_function_jacobian_x = state_function_jacobian_x;
    this->_state_function_jacobian_u = state_function_jacobian_u;
    this->_measurement_function_jacobian_x = measurement_function_jacobian_x;
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
   * @brief Calculates the state function using the provided inputs.
   *
   * This function evaluates the internal state function with the given state
   * vector `X`, control vector `U`, and parameter set `parameter`, returning
   * the computed next state.
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
   * control input U.
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
   * @brief Simulates the state trajectory over a time horizon given initial
   * state and control inputs.
   *
   * This function computes the evolution of the system state starting from an
   * initial state (`X_initial_in`) and applying a sequence of control inputs
   * (`U_horizon`) over a fixed number of steps (`NP`). At each step, the state
   * is updated using the `calculate_state_function`, which models the system
   * dynamics. The resulting trajectory of states is stored in `X_horizon`.
   *
   * @param X_initial_in The initial state vector.
   * @param U_horizon The matrix of control inputs for each time step in the
   * horizon.
   * @param parameter Additional parameters required for the state update
   * function.
   * @return X_Horizon_Type The matrix containing the state trajectory over the
   * horizon.
   */
  inline auto simulate_trajectory(const X_Type &X_initial_in,
                                  const U_Horizon_Type &U_horizon,
                                  const _Parameter_Type &parameter)
      -> X_Horizon_Type {

    X_Horizon_Type X_horizon;
    X_Type X = X_initial_in;

    MatrixOperation::set_row(X_horizon, X, 0);

    for (std::size_t k = 0; k < NP; k++) {
      auto U = MatrixOperation::get_row(U_horizon, k);
      X = this->calculate_state_function(X, U, parameter);

      MatrixOperation::set_row(X_horizon, X, k + 1);
    }

    return X_horizon;
  }

  // ----------------------------------------------------------------
  // PANOC interface methods
  // ----------------------------------------------------------------

  /**
   * @brief Computes the cost function value for PANOC/ALM.
   *
   * The cost includes state tracking, output tracking, and input penalty terms.
   * Output constraint penalties are NOT included here; they are handled
   * by ALM when output constraints are present.
   *
   * Uses this->X_initial as the current initial state (must be set
   * before calling this method).
   *
   * @param U_horizon Control input sequence over the prediction horizon.
   * @return _T Cost function value.
   */
  inline auto compute_cost(const U_Horizon_Type &U_horizon) -> _T {

    auto X_horizon = this->simulate_trajectory(this->X_initial, U_horizon,
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

      J += X_T_Qx_X + e_y_r_T_Qy_e_y_r + U_T_R_U;
    }

    auto eN_y_r = MatrixOperation::get_row(Y_horizon, NP) -
                  MatrixOperation::get_row(this->reference_trajectory, NP);

    auto XN_T_Px_XN = MatrixOperation::calculate_quadratic_form(
        MatrixOperation::get_row(X_horizon, NP), this->_Px);
    auto eN_y_r_T_Py_eN_y_r =
        MatrixOperation::calculate_quadratic_form(eN_y_r, this->_Py);

    J += XN_T_Px_XN + eN_y_r_T_Py_eN_y_r;

    return J;
  }

  /**
   * @brief Computes the gradient of the cost function via adjoint method for
   * PANOC/ALM.
   *
   * The gradient does NOT include output constraint penalty gradient;
   * those terms are handled by ALM when output constraints are present.
   *
   * Uses this->X_initial as the current initial state (must be set
   * before calling this method).
   *
   * @param U_horizon Control input sequence over the prediction horizon.
   * @return _Gradient_Type Gradient of the cost with respect to control inputs.
   */
  inline auto compute_gradient(const U_Horizon_Type &U_horizon)
      -> _Gradient_Type {

    auto X_horizon = this->simulate_trajectory(this->X_initial, U_horizon,
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

    // Terminal adjoint
    auto C_N = this->calculate_measurement_jacobian_x(
        MatrixOperation::get_row(X_horizon, NP), U_dummy,
        this->state_space_parameters);

    auto eN_y_r = MatrixOperation::get_row(Y_horizon, NP) -
                  MatrixOperation::get_row(this->reference_trajectory, NP);

    auto Px_XN = this->_Px * MatrixOperation::get_row(X_horizon, NP);
    auto Py_eN_y_r = this->_Py * eN_y_r;

    auto lam_next = static_cast<_T>(2) *
                    (Px_XN + PythonNumpy::ATranspose_mul_B(C_N, Py_eN_y_r));

    _Gradient_Type gradient;

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

      auto A_k_T_lam_next = PythonNumpy::ATranspose_mul_B(A_k, lam_next);

      lam_next = static_cast<_T>(2) *
                     (Qx_X + PythonNumpy::ATranspose_mul_B(Cx_k, Qy_ek_y)) +
                 A_k_T_lam_next;
    }

    return gradient;
  }

  // ----------------------------------------------------------------
  // ALM output constraint mapping methods
  // ----------------------------------------------------------------

  /**
   * @brief Computes the output constraint mapping F1(u) for ALM.
   *
   * F1(u) returns the predicted output trajectory.
   *
   * Uses this->X_initial as the current initial state (must be set
   * before calling this method).
   *
   * @param U_horizon Control input sequence over the prediction horizon.
   * @return Y_Horizon_Type Output trajectory over the prediction horizon.
   */
  inline auto compute_output_mapping(const U_Horizon_Type &U_horizon)
      -> Y_Horizon_Type {

    auto X_horizon = this->simulate_trajectory(this->X_initial, U_horizon,
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

    return Y_horizon;
  }

  /**
   * @brief Computes JF1(u)^T @ d for ALM via adjoint method.
   *
   * JF1 is the Jacobian of the output trajectory mapping F1(u).
   * This function computes the transpose-Jacobian product efficiently
   * using a backward adjoint pass:
   *   mu_Np = C_Np^T @ D[:, Np]
   *   For k = Np-1, ..., 0:
   *     result[:, k] = B_k^T @ mu_{k+1}
   *     mu_k = C_k^T @ D[:, k] + A_k^T @ mu_{k+1}
   *
   * Uses this->X_initial as the current initial state (must be set
   * before calling this method).
   *
   * @param U_horizon Control input sequence over the prediction horizon.
   * @param D Dual vector reshaped as (ny, Np+1) tile matrix.
   * @return _Gradient_Type JF1(u)^T @ d result.
   */
  inline auto compute_output_jacobian_trans(const U_Horizon_Type &U_horizon,
                                            const _Dual_Type &D)
      -> _Gradient_Type {

    auto X_horizon = this->simulate_trajectory(this->X_initial, U_horizon,
                                               this->state_space_parameters);

    U_Type U_dummy;

    // Backward adjoint pass
    auto C_N = this->calculate_measurement_jacobian_x(
        MatrixOperation::get_row(X_horizon, NP), U_dummy,
        this->state_space_parameters);

    auto mu =
        PythonNumpy::ATranspose_mul_B(C_N, MatrixOperation::get_row(D, NP));

    _Gradient_Type result;
    for (std::size_t k = NP; k-- > 0;) {
      auto A_k = this->calculate_state_jacobian_x(
          MatrixOperation::get_row(X_horizon, k),
          MatrixOperation::get_row(U_horizon, k), this->state_space_parameters);
      auto B_k = this->calculate_state_jacobian_u(
          MatrixOperation::get_row(X_horizon, k),
          MatrixOperation::get_row(U_horizon, k), this->state_space_parameters);

      auto B_k_T_mu = PythonNumpy::ATranspose_mul_B(B_k, mu);
      MatrixOperation::set_row(result, B_k_T_mu, k);

      auto C_k = this->calculate_measurement_jacobian_x(
          MatrixOperation::get_row(X_horizon, k), U_dummy,
          this->state_space_parameters);

      mu = PythonNumpy::ATranspose_mul_B(C_k, MatrixOperation::get_row(D, k)) +
           PythonNumpy::ATranspose_mul_B(A_k, mu);
    }

    return result;
  }

public:
  /* Variable */
  X_Type X_initial;
  _Parameter_Type state_space_parameters;
  _Reference_Trajectory_Type reference_trajectory;

protected:
  /* Variable */
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
};

/* make OptimizationEngine_CostMatrices */
/**
 * @brief Constructs an OptimizationEngine_CostMatrices object for Nonlinear
 * Model Predictive Control (NMPC).
 *
 * This function creates and returns an instance of
 * OptimizationEngine_CostMatrices, which encapsulates the cost matrices and
 * constraints required for PANOC/ALM-based NMPC.
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
 *
 * @param Qx Diagonal weighting matrix for state error.
 * @param R Diagonal weighting matrix for control input.
 * @param Qy Diagonal weighting matrix for measurement error.
 * @param U_min Minimum control input constraints.
 * @param U_max Maximum control input constraints.
 * @param Y_min Minimum output constraints.
 * @param Y_max Maximum output constraints.
 *
 * @return OptimizationEngine_CostMatrices object initialized with the provided
 * matrices and constraints.
 */
template <typename T, std::size_t Np, typename Parameter_Type,
          typename U_Min_Type, typename U_Max_Type, typename Y_Min_Type,
          typename Y_Max_Type, typename State_Jacobian_X_Matrix_Type,
          typename State_Jacobian_U_Matrix_Type,
          typename Measurement_Jacobian_X_Matrix_Type>
inline auto make_OptimizationEngine_CostMatrices(
    const PythonNumpy::DiagMatrix_Type<T, State_Jacobian_X_Matrix_Type::ROWS>
        &Qx,
    const PythonNumpy::DiagMatrix_Type<T, State_Jacobian_U_Matrix_Type::ROWS>
        &R,
    const PythonNumpy::DiagMatrix_Type<
        T, Measurement_Jacobian_X_Matrix_Type::COLS> &Qy,
    U_Min_Type U_min, U_Max_Type U_max, Y_Min_Type Y_min, Y_Max_Type Y_max)
    -> OptimizationEngine_CostMatrices<
        T, Np, Parameter_Type, U_Min_Type, U_Max_Type, Y_Min_Type, Y_Max_Type,
        State_Jacobian_X_Matrix_Type, State_Jacobian_U_Matrix_Type,
        Measurement_Jacobian_X_Matrix_Type> {

  return OptimizationEngine_CostMatrices<
      T, Np, Parameter_Type, U_Min_Type, U_Max_Type, Y_Min_Type, Y_Max_Type,
      State_Jacobian_X_Matrix_Type, State_Jacobian_U_Matrix_Type,
      Measurement_Jacobian_X_Matrix_Type>(Qx, R, Qy, U_min, U_max, Y_min,
                                          Y_max);
}

/* OptimizationEngine_CostMatrices type */
/**
 * @brief Alias template for OptimizationEngine_CostMatrices with multiple type
 * parameters.
 *
 * This alias simplifies the usage of the OptimizationEngine_CostMatrices
 * template by providing a more concise name.
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
 */
template <typename T, std::size_t Np, typename Parameter_Type,
          typename U_Min_Type, typename U_Max_Type, typename Y_Min_Type,
          typename Y_Max_Type, typename State_Jacobian_X_Matrix_Type,
          typename State_Jacobian_U_Matrix_Type,
          typename Measurement_Jacobian_X_Matrix_Type>
using OptimizationEngine_CostMatrices_Type = OptimizationEngine_CostMatrices<
    T, Np, Parameter_Type, U_Min_Type, U_Max_Type, Y_Min_Type, Y_Max_Type,
    State_Jacobian_X_Matrix_Type, State_Jacobian_U_Matrix_Type,
    Measurement_Jacobian_X_Matrix_Type>;

} // namespace PythonOptimization

#endif // __PYTHON_OPTIMIZATION_OPTIMIZATION_ENGINE_MATRIX_UTILITY_HPP__
