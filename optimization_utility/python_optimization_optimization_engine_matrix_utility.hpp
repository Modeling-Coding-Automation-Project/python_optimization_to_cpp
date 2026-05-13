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
#ifndef PYTHON_OPTIMIZATION_OPTIMIZATION_ENGINE_MATRIX_UTILITY_HPP_
#define PYTHON_OPTIMIZATION_OPTIMIZATION_ENGINE_MATRIX_UTILITY_HPP_

#include "python_optimization_matrix_utility_common.hpp"
#include "python_optimization_utility_matrix_operation.hpp"

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
      State_Jacobian_X_Matrix_Type_In::COLS;
  static constexpr std::size_t INPUT_SIZE =
      State_Jacobian_U_Matrix_Type_In::COLS;
  static constexpr std::size_t OUTPUT_SIZE =
      Measurement_Jacobian_X_Matrix_Type_In::ROWS;

  // To avoid ODR violation in C++11/14, use enum hack.
  enum : std::size_t { NP = Np_In };

  static constexpr std::size_t STATE_JACOBIAN_X_ROWS = STATE_SIZE;
  static constexpr std::size_t STATE_JACOBIAN_X_COLS = STATE_SIZE;

  static constexpr std::size_t STATE_JACOBIAN_U_ROWS = STATE_SIZE;
  static constexpr std::size_t STATE_JACOBIAN_U_COLS = INPUT_SIZE;

  static constexpr std::size_t MEASUREMENT_JACOBIAN_X_ROWS = OUTPUT_SIZE;
  static constexpr std::size_t MEASUREMENT_JACOBIAN_X_COLS = STATE_SIZE;

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

  using StateFunction_Out_Type_ = X_Type;
  using MeasurementFunction_Out_Type_ = Y_Type;

  using StateFunctionJacobian_X_Out_Type_ = State_Jacobian_X_Matrix_Type_In;
  using StateFunctionJacobian_U_Out_Type_ = State_Jacobian_U_Matrix_Type_In;
  using MeasurementFunctionJacobian_X_Out_Type_ =
      Measurement_Jacobian_X_Matrix_Type_In;

  using StateFunction_Object_ =
      StateFunction_Object<X_Type, U_Type, Parameter_Type_>;
  using MeasurementFunction_Object_ =
      MeasurementFunction_Object<Y_Type, X_Type, U_Type, Parameter_Type_>;

  using StateFunctionJacobian_X_Object_ =
      StateFunctionJacobian_X_Object<StateFunctionJacobian_X_Out_Type_, X_Type,
                                     U_Type, Parameter_Type_>;
  using StateFunctionJacobian_U_Object_ =
      StateFunctionJacobian_U_Object<StateFunctionJacobian_U_Out_Type_, X_Type,
                                     U_Type, Parameter_Type_>;
  using MeasurementFunctionJacobian_X_Object_ =
      MeasurementFunctionJacobian_X_Object<
          MeasurementFunctionJacobian_X_Out_Type_, X_Type, U_Type,
          Parameter_Type_>;

  using Reference_Trajectory_Type_ = Y_Horizon_Type;

  using Gradient_Type_ = U_Horizon_Type;

  /* Output mapping types for ALM */
  using Output_Mapping_Type_ = Y_Horizon_Type;
  using Dual_Type_ = Y_Horizon_Type;

public:
  /* Constructor */
  OptimizationEngine_CostMatrices()
      : Y_offset_(), Qx_(), R_(), Qy_(), Px_(), Py_(), U_min_matrix_(),
        U_max_matrix_(), Y_min_matrix_(), Y_max_matrix_(), _state_function(),
        _measurement_function(), _state_function_jacobian_x(),
        _state_function_jacobian_u(), _measurement_function_jacobian_x() {}

  OptimizationEngine_CostMatrices(const Qx_Type_ &Qx, const R_Type_ &R,
                                  const Qy_Type_ &Qy, const U_Min_Type &U_min,
                                  const U_Max_Type &U_max,
                                  const Y_Min_Type &Y_min,
                                  const Y_Max_Type &Y_max)
      : Y_offset_(), Qx_(Qx), R_(R), Qy_(Qy), Px_(Qx), Py_(Qy), U_min_matrix_(),
        U_max_matrix_(), Y_min_matrix_(), Y_max_matrix_(), _state_function(),
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
        Y_offset_(input.Y_offset_), Qx_(input.Qx_), R_(input.R_),
        Qy_(input.Qy_), Px_(input.Px_), Py_(input.Py_),
        U_min_matrix_(input.U_min_matrix_), U_max_matrix_(input.U_max_matrix_),
        Y_min_matrix_(input.Y_min_matrix_), Y_max_matrix_(input.Y_max_matrix_),
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
        Y_offset_(std::move(input.Y_offset_)), Qx_(std::move(input.Qx_)),
        R_(std::move(input.R_)), Qy_(std::move(input.Qy_)),
        Px_(std::move(input.Px_)), Py_(std::move(input.Py_)),
        U_min_matrix_(std::move(input.U_min_matrix_)),
        U_max_matrix_(std::move(input.U_max_matrix_)),
        Y_min_matrix_(std::move(input.Y_min_matrix_)),
        Y_max_matrix_(std::move(input.Y_max_matrix_)),
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
      const StateFunction_Object_ &state_function,
      const MeasurementFunction_Object_ &measurement_function,
      const StateFunctionJacobian_X_Object_ &state_function_jacobian_x,
      const StateFunctionJacobian_U_Object_ &state_function_jacobian_u,
      const MeasurementFunctionJacobian_X_Object_
          &measurement_function_jacobian_x) {

    this->_state_function = state_function;
    this->_measurement_function = measurement_function;

    this->_state_function_jacobian_x = state_function_jacobian_x;
    this->_state_function_jacobian_u = state_function_jacobian_u;
    this->_measurement_function_jacobian_x = measurement_function_jacobian_x;
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
   * @brief Calculates the state function using the provided inputs.
   *
   * This function evaluates the internal state function with the given state
   * vector `X`, control vector `U`, and parameter set `parameter`, returning
   * the computed next state.
   *
   * @param X The current state vector.
   * @param U The control input vector.
   * @param parameter The set of parameters required for the state function.
   * @return StateFunction_Out_Type_ The result of the state function
   * evaluation.
   */
  inline auto calculate_state_function(const X_Type &X, const U_Type &U,
                                       const Parameter_Type_ &parameter)
      -> StateFunction_Out_Type_ {

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
                                             const Parameter_Type_ &parameter)
      -> MeasurementFunction_Out_Type_ {

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
                                         const Parameter_Type_ &parameter)
      -> StateFunctionJacobian_X_Out_Type_ {

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
                                         const Parameter_Type_ &parameter)
      -> StateFunctionJacobian_U_Out_Type_ {

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
                                               const Parameter_Type_ &parameter)
      -> MeasurementFunctionJacobian_X_Out_Type_ {

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
                                  const Parameter_Type_ &parameter)
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
   * @return T_ Cost function value.
   */
  inline auto compute_cost(const U_Horizon_Type &U_horizon) -> T_ {

    auto X_horizon = this->simulate_trajectory(this->X_initial, U_horizon,
                                               this->state_space_parameters);

    Y_Horizon_Type Y_horizon;
    PythonNumpy::update_tile_concatenated_matrix<1, (NP + 1), Y_Type>(
        Y_horizon, this->Y_offset_);
    U_Type U_dummy;

    for (std::size_t k = 0; k < (NP + 1); k++) {
      auto Y_k = this->calculate_measurement_function(
          MatrixOperation::get_row(X_horizon, k), U_dummy,
          this->state_space_parameters);

      MatrixOperation::set_row(Y_horizon, Y_k, k);
    }

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

      J += X_T_Qx_X + e_y_r_T_Qy_e_y_r + U_T_R_U;
    }

    auto eN_y_r = MatrixOperation::get_row(Y_horizon, NP) -
                  MatrixOperation::get_row(this->reference_trajectory, NP);

    auto XN_T_Px_XN = MatrixOperation::calculate_quadratic_form(
        MatrixOperation::get_row(X_horizon, NP), this->Px_);
    auto eN_y_r_T_Py_eN_y_r =
        MatrixOperation::calculate_quadratic_form(eN_y_r, this->Py_);

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
   * @return Gradient_Type_ Gradient of the cost with respect to control inputs.
   */
  inline auto compute_gradient(const U_Horizon_Type &U_horizon)
      -> Gradient_Type_ {

    auto X_horizon = this->simulate_trajectory(this->X_initial, U_horizon,
                                               this->state_space_parameters);

    Y_Horizon_Type Y_horizon;
    PythonNumpy::update_tile_concatenated_matrix<1, (NP + 1), Y_Type>(
        Y_horizon, this->Y_offset_);
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

    auto Px_XN = this->Px_ * MatrixOperation::get_row(X_horizon, NP);
    auto Py_eN_y_r = this->Py_ * eN_y_r;

    auto lam_next = static_cast<T_>(2) *
                    (Px_XN + PythonNumpy::ATranspose_mul_B(C_N, Py_eN_y_r));

    Gradient_Type_ gradient;

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

      auto A_k_T_lam_next = PythonNumpy::ATranspose_mul_B(A_k, lam_next);

      lam_next = static_cast<T_>(2) *
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
        Y_horizon, this->Y_offset_);
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
   * @return Gradient_Type_ JF1(u)^T @ d result.
   */
  inline auto compute_output_jacobian_trans(const U_Horizon_Type &U_horizon,
                                            const Dual_Type_ &D)
      -> Gradient_Type_ {

    auto X_horizon = this->simulate_trajectory(this->X_initial, U_horizon,
                                               this->state_space_parameters);

    U_Type U_dummy;

    // Backward adjoint pass
    auto C_N = this->calculate_measurement_jacobian_x(
        MatrixOperation::get_row(X_horizon, NP), U_dummy,
        this->state_space_parameters);

    auto mu =
        PythonNumpy::ATranspose_mul_B(C_N, MatrixOperation::get_row(D, NP));

    Gradient_Type_ result;
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
  Parameter_Type_ state_space_parameters;
  Reference_Trajectory_Type_ reference_trajectory;

protected:
  /* Variable */
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

  StateFunction_Object_ _state_function;
  MeasurementFunction_Object_ _measurement_function;

  StateFunctionJacobian_X_Object_ _state_function_jacobian_x;
  StateFunctionJacobian_U_Object_ _state_function_jacobian_u;
  MeasurementFunctionJacobian_X_Object_ _measurement_function_jacobian_x;
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
    const PythonNumpy::DiagMatrix_Type<T, State_Jacobian_X_Matrix_Type::COLS>
        &Qx,
    const PythonNumpy::DiagMatrix_Type<T, State_Jacobian_U_Matrix_Type::COLS>
        &R,
    const PythonNumpy::DiagMatrix_Type<
        T, Measurement_Jacobian_X_Matrix_Type::ROWS> &Qy,
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

#endif // PYTHON_OPTIMIZATION_OPTIMIZATION_ENGINE_MATRIX_UTILITY_HPP_
