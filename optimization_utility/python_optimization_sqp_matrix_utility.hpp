#ifndef __PYTHON_OPTIMIZATION_SQP_MATRIX_UTILITY_HPP__
#define __PYTHON_OPTIMIZATION_SQP_MATRIX_UTILITY_HPP__

#include "python_control.hpp"
#include "python_numpy.hpp"

#include <array>
#include <functional>
#include <type_traits>

namespace PythonOptimization {

static constexpr double Y_MIN_MAX_RHO_FACTOR_DEFAULT = 1.0e2;

/* State Space Function Objects */

template <typename State_Type, typename Input_Type, typename Parameter_Type>
using StateFunction_Object = std::function<State_Type(
    const State_Type &, const Input_Type &, const Parameter_Type &)>;

template <typename Output_Type, typename State_Type, typename Parameter_Type>
using MeasurementFunction_Object =
    std::function<Output_Type(const State_Type &, const Parameter_Type &)>;

template <typename State_Jacobian_X_Matrix_Type, typename State_Type,
          typename Input_Type, typename Parameter_Type>
using StateFunctionJacobian_X_Object =
    std::function<State_Jacobian_X_Matrix_Type(
        const State_Type &, const Input_Type &, const Parameter_Type &)>;

template <typename State_Jacobian_U_Matrix_Type, typename State_Type,
          typename Input_Type, typename Parameter_Type>
using StateFunctionJacobian_U_Object =
    std::function<State_Jacobian_U_Matrix_Type(
        const State_Type &, const Input_Type &, const Parameter_Type &)>;

template <typename Measurement_Jacobian_X_Matrix_Type, typename State_Type,
          typename Input_Type, typename Parameter_Type>
using MeasurementFunctionJacobian_X_Object =
    std::function<Measurement_Jacobian_X_Matrix_Type(
        const State_Type &, const Input_Type &, const Parameter_Type &)>;

template <typename State_Hessian_XX_Matrix_Type, typename State_Type,
          typename Input_Type, typename Parameter_Type>
using StateFunctionHessian_XX_Object =
    std::function<State_Hessian_XX_Matrix_Type(
        const State_Type &, const Input_Type &, const Parameter_Type &)>;

template <typename State_Hessian_XU_Matrix_Type, typename State_Type,
          typename Input_Type, typename Parameter_Type>
using StateFunctionHessian_XU_Object =
    std::function<State_Hessian_XU_Matrix_Type(
        const State_Type &, const Input_Type &, const Parameter_Type &)>;

template <typename State_Hessian_UX_Matrix_Type, typename State_Type,
          typename Input_Type, typename Parameter_Type>
using StateFunctionHessian_UX_Object =
    std::function<State_Hessian_UX_Matrix_Type(
        const State_Type &, const Input_Type &, const Parameter_Type &)>;

template <typename State_Hessian_UU_Matrix_Type, typename State_Type,
          typename Input_Type, typename Parameter_Type>
using StateFunctionHessian_UU_Object =
    std::function<State_Hessian_UU_Matrix_Type(
        const State_Type &, const Input_Type &, const Parameter_Type &)>;

template <typename Measurement_Hessian_XX_Matrix_Type, typename State_Type,
          typename Input_Type, typename Parameter_Type>
using MeasurementFunctionHessian_XX_Object =
    std::function<Measurement_Hessian_XX_Matrix_Type(
        const State_Type &, const Input_Type &, const Parameter_Type &)>;

/* SQP cost matrices for Nonlinear MPC */

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

  static constexpr std::size_t NP = Np_In;

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

  /* Check Compatibility */
  static_assert(std::is_same<U_Min_Type_In::Value_Type, T>::value,
                "U_Min_Type_In::Value_Type != T");
  static_assert(std::is_same<U_Max_Type_In::Value_Type, T>::value,
                "U_Max_Type_In::Value_Type != T");
  static_assert(std::is_same<Y_Min_Type_In::Value_Type, T>::value,
                "Y_Min_Type_In::Value_Type != T");
  static_assert(std::is_same<Y_Max_Type_In::Value_Type, T>::value,
                "Y_Max_Type_In::Value_Type != T");

  static_assert(
      std::is_same<State_Jacobian_X_Matrix_Type_In::Value_Type, T>::value,
      "State_Jacobian_X_Matrix_Type::Value_Type != T");
  static_assert(
      std::is_same<State_Jacobian_U_Matrix_Type_In::Value_Type, T>::value,
      "State_Jacobian_U_Matrix_Type::Value_Type != T");
  static_assert(
      std::is_same<Measurement_Jacobian_X_Matrix_Type_In::Value_Type, T>::value,
      "Measurement_Jacobian_X_Matrix_Type::Value_Type != T");

  static_assert(
      std::is_same<State_Hessian_XX_Matrix_Type_In::Value_Type, T>::value,
      "State_Hessian_XX_Matrix_Type::Value_Type != T");
  static_assert(
      std::is_same<State_Hessian_XU_Matrix_Type_In::Value_Type, T>::value,
      "State_Hessian_XU_Matrix_Type::Value_Type != T");
  static_assert(
      std::is_same<State_Hessian_UX_Matrix_Type_In::Value_Type, T>::value,
      "State_Hessian_UX_Matrix_Type::Value_Type != T");
  static_assert(
      std::is_same<State_Hessian_UU_Matrix_Type_In::Value_Type, T>::value,
      "State_Hessian_UU_Matrix_Type::Value_Type != T");
  static_assert(
      std::is_same<Measurement_Hessian_XX_Matrix_Type_In::Value_Type, T>::value,
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
      MeasurementFunction_Object<Y_Type, X_Type, _Parameter_Type>;

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

  using _X_horizon_Type = PythonNumpy::Tile_Type<1, (NP + 1), X_Type>;
  using _U_horizon_Type = PythonNumpy::Tile_Type<1, NP, U_Type>;
  using _Y_horizon_Type = PythonNumpy::Tile_Type<1, (NP + 1), Y_Type>;

public:
  /* Constructor */
  SQP_CostMatrices_NMPC() {}

public:
  /* Setters */
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

  /* Function */
  inline auto l_xx(const X_Type &X, const U_Type &U) -> _Qx_Type & {
    static_cast<void>(X);
    static_cast<void>(U);

    return 2.0 * this->_Qx;
  }

  inline auto l_uu(const X_Type &X, const U_Type &U) -> _R_Type & {
    static_cast<void>(X);
    static_cast<void>(U);

    return 2.0 * this->_R;
  }

  inline auto l_xu(const X_Type &X, const U_Type &U)
      -> PythonNumpy::SparseMatrixEmpty_Type<_T, STATE_SIZE, INPUT_SIZE> {
    static_cast<void>(X);
    static_cast<void>(U);

    return PythonNumpy::make_SparseMatrixEmpty<_T, STATE_SIZE, INPUT_SIZE>();
  }

  inline auto l_ux(const X_Type &X, const U_Type &U)
      -> PythonNumpy::SparseMatrixEmpty_Type<_T, INPUT_SIZE, STATE_SIZE> {
    static_cast<void>(X);
    static_cast<void>(U);

    return PythonNumpy::make_SparseMatrixEmpty<_T, INPUT_SIZE, STATE_SIZE>();
  }

  inline auto calculate_state_function(const X_Type &X, const U_Type &U,
                                       const _Parameter_Type &parameter)
      -> _StateFunction_Out_Type {

    return this->_state_function(X, U, parameter);
  }

  inline auto calculate_measurement_function(const X_Type &X,
                                             const _Parameter_Type &parameter)
      -> _MeasurementFunction_Out_Type {

    return this->_measurement_function(X, parameter);
  }

  inline auto calculate_state_jacobian_x(const X_Type &X, const U_Type &U,
                                         const _Parameter_Type &parameter)
      -> _StateFunctionJacobian_X_Out_Type {

    return this->_state_function_jacobian_x(X, U, parameter);
  }

  inline auto calculate_state_jacobian_u(const X_Type &X, const U_Type &U,
                                         const _Parameter_Type &parameter)
      -> _StateFunctionJacobian_U_Out_Type {

    return this->_state_function_jacobian_u(X, U, parameter);
  }

  inline auto calculate_measurement_jacobian_x(const X_Type &X, const U_Type &U,
                                               const _Parameter_Type &parameter)
      -> _MeasurementFunctionJacobian_X_Out_Type {

    return this->_measurement_function_jacobian_x(X, U, parameter);
  }

  template <typename Lambda_Vector_Type>
  inline auto fx_xx_lambda_contract(const X_Type &X, const U_Type &U,
                                    const _Parameter_Type &parameter,
                                    const Lambda_Vector_Type &lam_next,
                                    const X_Type &dX)
      -> _StateFunctionHessian_XX_Out_Type {

    static_assert(Lambda_Vector_Type::COLS == STATE_SIZE,
                  "Lambda_Vector_Type::COLS != STATE_SIZE");
    static_assert(Lambda_Vector_Type::ROWS == 1,
                  "Lambda_Vector_Type::ROWS != 1");

    auto Hf_xx = this->_state_function_hessian_xx(X, U, parameter);

    _StateFunctionHessian_XX_Out_Type out;

    for (std::size_t i = 0; i < STATE_SIZE; i++) {

      for (std::size_t j = 0; j < STATE_SIZE; j++) {
        _T acc = static_cast<_T>(0);

        for (std::size_t k = 0; k < STATE_SIZE; k++) {
          acc += Hf_xx(i * STATE_SIZE + j, k) * dX(k, 0);
        }
        out(j, 0) += lam_next(i, 0) * acc;
      }
    }

    return out;
  }

  template <typename Lambda_Vector_Type>
  inline auto fx_xu_lambda_contract(const X_Type &X, const U_Type &U,
                                    const _Parameter_Type &parameter,
                                    const Lambda_Vector_Type &lam_next,
                                    const U_Type &dU)
      -> _StateFunctionHessian_XU_Out_Type {
    static_assert(Lambda_Vector_Type::COLS == STATE_SIZE,
                  "Lambda_Vector_Type::COLS != STATE_SIZE");
    static_assert(Lambda_Vector_Type::ROWS == 1,
                  "Lambda_Vector_Type::ROWS != 1");

    auto Hf_xu = this->_state_function_hessian_xu(X, U, parameter);

    _StateFunctionHessian_XU_Out_Type out;

    if (0 == INPUT_SIZE) {
      // Do Nothing.
    } else {
      for (std::size_t i = 0; i < STATE_SIZE; i++) {

        for (std::size_t j = 0; j < STATE_SIZE; j++) {
          _T acc = static_cast<_T>(0);

          for (std::size_t k = 0; k < INPUT_SIZE; k++) {
            acc += Hf_xu(i * STATE_SIZE + j, k) * dU(k, 0);
          }
          out(j, 0) += lam_next(i, 0) * acc;
        }
      }
    }

    return out;
  }

  template <typename Lambda_Vector_Type>
  inline auto fu_xx_lambda_contract(const X_Type &X, const U_Type &U,
                                    const _Parameter_Type &parameter,
                                    const Lambda_Vector_Type &lam_next,
                                    const X_Type &dX)
      -> _StateFunctionHessian_UX_Out_Type {

    static_assert(Lambda_Vector_Type::COLS == STATE_SIZE,
                  "Lambda_Vector_Type::COLS != STATE_SIZE");
    static_assert(Lambda_Vector_Type::ROWS == 1,
                  "Lambda_Vector_Type::ROWS != 1");

    auto Hf_ux = this->_state_function_hessian_ux(X, U, parameter);

    _StateFunctionHessian_UX_Out_Type out;

    for (std::size_t i = 0; i < STATE_SIZE; i++) {

      for (std::size_t k = 0; k < INPUT_SIZE; k++) {
        _T acc = static_cast<_T>(0);

        for (std::size_t j = 0; j < STATE_SIZE; j++) {
          acc += Hf_ux(i * INPUT_SIZE + k, j) * dX(j, 0);
        }
        out(k, 0) += lam_next(i, 0) * acc;
      }
    }

    return out;
  }

  template <typename Lambda_Vector_Type>
  inline auto fu_uu_lambda_contract(const X_Type &X, const U_Type &U,
                                    const _Parameter_Type &parameter,
                                    const Lambda_Vector_Type &lam_next,
                                    const U_Type &dU)
      -> _StateFunctionHessian_UU_Out_Type {

    static_assert(Lambda_Vector_Type::COLS == STATE_SIZE,
                  "Lambda_Vector_Type::COLS != STATE_SIZE");
    static_assert(Lambda_Vector_Type::ROWS == 1,
                  "Lambda_Vector_Type::ROWS != 1");

    auto Hf_uu = this->_state_function_hessian_uu(X, U, parameter);

    _StateFunctionHessian_UU_Out_Type out;

    if (0 == INPUT_SIZE) {
      // Do Nothing.
    } else {
      for (std::size_t i = 0; i < STATE_SIZE; i++) {

        for (std::size_t j = 0; j < INPUT_SIZE; j++) {
          _T acc = static_cast<_T>(0);

          for (std::size_t k = 0; k < INPUT_SIZE; k++) {
            acc += Hf_uu(i * INPUT_SIZE + j, k) * dU(k, 0);
          }
          out(j, 0) += lam_next(i, 0) * acc;
        }
      }
    }

    return out;
  }

  template <typename Weight_Vector_Type>
  inline auto
  hxx_lambda_contract(const X_Type &X, const _Parameter_Type &parameter,
                      const Weight_Vector_Type &weight, const X_Type &dX)
      -> _MeasurementFunctionHessian_XX_Out_Type {

    static_assert(Weight_Vector_Type::COLS == OUTPUT_SIZE,
                  "Weight_Vector_Type::COLS != OUTPUT_SIZE");
    static_assert(Weight_Vector_Type::ROWS == 1,
                  "Weight_Vector_Type::ROWS != 1");

    U_Type U;
    auto Hh_xx = this->_measurement_function_hessian_xx(X, U, parameter);

    _MeasurementFunctionHessian_XX_Out_Type out;

    for (std::size_t i = 0; i < OUTPUT_SIZE; i++) {

      for (std::size_t j = 0; j < STATE_SIZE; j++) {
        _T acc = static_cast<_T>(0);

        for (std::size_t k = 0; k < STATE_SIZE; k++) {
          acc += Hh_xx(i * STATE_SIZE + j, k) * dX(k, 0);
        }
        out(j, 0) += weight(i, 0) * acc;
      }
    }

    return out;
  }

  inline auto simulate_trajectory(const X_Type &X_initial,
                                  const _U_horizon_Type &U_horizon,
                                  const _Parameter_Type &parameter)
      -> _X_horizon_Type {

    _X_horizon_Type X_horizon;
    X_Type X = X_initial;

    for (std::size_t k = 0; k < NP; k++) {
      X = this->calculate_state_function(X, U_horizon, parameter);

      for (std::size_t i = 0; i < STATE_SIZE; i++) {
        X_horizon(i, k + 1) = X(i, 0);
      }
    }

    return X_horizon;
  }

  /*
      def calculate_Y_limit_penalty(self, Y: np.ndarray):
        """
        Calculates the penalty matrix Y_limit_penalty for the given output
     matrix Y based on minimum and maximum constraints. For each element in Y,
     the penalty is computed as follows:
            - If Y[i, j] is less than the corresponding minimum constraint
              (self.Y_min_matrix[i, j]),
              the penalty is Y[i, j] - self.Y_min_matrix[i, j].
            - If Y[i, j] is greater than the corresponding maximum constraint
              (self.Y_max_matrix[i, j]),
              the penalty is Y[i, j] - self.Y_max_matrix[i, j].
            - Otherwise, the penalty is 0.
        Args:
            Y (np.ndarray): Output matrix of shape (self.ny, self.Np + 1)
              to be checked against constraints.
        Returns:
            np.ndarray: Penalty matrix of the same shape as Y,
              containing the calculated penalties.
        """
        Y_limit_penalty = np.zeros((self.ny, self.Np + 1))
        for i in range(self.ny):
            for j in range(self.Np + 1):
                if Y[i, j] < self.Y_min_matrix[i, j]:
                    Y_limit_penalty[i, j] = Y[i, j] - self.Y_min_matrix[i, j]
                elif Y[i, j] > self.Y_max_matrix[i, j]:
                    Y_limit_penalty[i, j] = Y[i, j] - self.Y_max_matrix[i, j]

        return Y_limit_penalty

  */

  inline auto calculate_Y_limit_penalty(const _Y_horizon_Type &Y_horizon)
      -> _Y_horizon_Type {
    _Y_horizon_Type Y_limit_penalty;

    for (std::size_t i = 0; i < OUTPUT_SIZE; i++) {
      for (std::size_t j = 0; j < (NP + 1); j++) {
        if (Y_horizon(i, j) < this->_Y_min_matrix(i, j)) {
          Y_limit_penalty(i, j) = Y_horizon(i, j) - this->_Y_min_matrix(i, j);
        } else if (Y_horizon(i, j) > this->_Y_max_matrix(i, j)) {
          Y_limit_penalty(i, j) = Y_horizon(i, j) - this->_Y_max_matrix(i, j);
        } else {
          // Do Nothing.
        }
      }
    }

    return Y_limit_penalty;
  }

public:
  /* Variable */
  _Parameter_Type state_space_parameters;

protected:
  /* Variable */
  _T _Y_min_max_rho;
  Y_Type _Y_offset;

  _Qx_Type _Qx;
  _R_Type _R;
  _Qy_Type _Qy;

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

} // namespace PythonOptimization

#endif // __PYTHON_OPTIMIZATION_SQP_MATRIX_UTILITY_HPP__
