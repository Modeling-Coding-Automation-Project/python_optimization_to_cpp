#ifndef __PYTHON_OPTIMIZATION_SQP_MATRIX_UTILITY_HPP__
#define __PYTHON_OPTIMIZATION_SQP_MATRIX_UTILITY_HPP__

#include "python_control.hpp"
#include "python_numpy.hpp"

#include <array>
#include <functional>
#include <type_traits>

namespace PythonOptimization {

static constexpr double Y_MIN_MAX_RHO_FACTOR_DEFAULT = 1.0e2;

namespace MatrixOperation {

template <typename Matrix_Out_Type, typename Matrix_In_Type>
inline void set_row(Matrix_Out_Type &out_matrix,
                    const Matrix_In_Type &in_matrix,
                    const std::size_t &row_index) {

  static_assert(Matrix_Out_Type::COLS == Matrix_In_Type::COLS,
                "Matrix_Out_Type::COLS != Matrix_In_Type::COLS");
  static_assert(Matrix_Out_Type::ROWS == 1, "Matrix_Out_Type::ROWS != 1");

  for (std::size_t i = 0; i < Matrix_In_Type::ROWS; i++) {
    out_matrix(i, row_index) = in_matrix(i, 0);
  }
}

template <typename Matrix_In_Type>
inline auto get_row(const Matrix_In_Type &in_matrix,
                    const std::size_t &row_index)
    -> PythonNumpy::DenseMatrix_Type<typename Matrix_In_Type::Value_Type,
                                     Matrix_In_Type::COLS, 1> {

  PythonNumpy::DenseMatrix_Type<typename Matrix_In_Type::Value_Type,
                                Matrix_In_Type::COLS, 1>
      out;

  for (std::size_t i = 0; i < Matrix_In_Type::COLS; i++) {
    out(i, 0) = in_matrix(i, row_index);
  }

  return out;
}

template <typename X_Type, typename W_Type>
inline auto calculate_quadratic_form(const X_Type &X, const W_Type &W) ->
    typename X_Type::Value_Type {

  static_assert(W_Type::ROWS == X_Type::COLS, "W_Type::ROWS != X_Type::COLS");
  static_assert(X_Type::ROWS == 1, "X_Type::ROWS != 1");

  auto result = PythonNumpy::ATranspose_mul_B(X, W) * X;

  return result.template get<0, 0>();
}

template <typename X_Type>
inline auto calculate_quadratic_no_weighted(const X_Type &X) ->
    typename X_Type::Value_Type {

  static_assert(X_Type::ROWS == 1, "X_Type::ROWS != 1");

  auto result = PythonNumpy::ATranspose_mul_B(X, X);

  return result.template get<0, 0>();
}

} // namespace MatrixOperation

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

  using X_Horizon_Type = PythonNumpy::Tile_Type<1, (NP + 1), X_Type>;
  using U_Horizon_Type = PythonNumpy::Tile_Type<1, NP, U_Type>;
  using Y_Horizon_Type = PythonNumpy::Tile_Type<1, (NP + 1), Y_Type>;

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

  using _Reference_Trajectory_Type = Y_Horizon_Type;

  using _Gradient_Type = U_Horizon_Type;
  using _V_horizon_Type = U_Horizon_Type;
  using _HVP_Type = U_Horizon_Type;

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

    return static_cast<_T>(2) * this->_Qx;
  }

  inline auto l_uu(const X_Type &X, const U_Type &U) -> _R_Type & {
    static_cast<void>(X);
    static_cast<void>(U);

    return static_cast<_T>(2) * this->_R;
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
      /* Do Nothing. */
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
      /* Do Nothing. */
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
                                  const U_Horizon_Type &U_horizon,
                                  const _Parameter_Type &parameter)
      -> X_Horizon_Type {

    X_Horizon_Type X_horizon;
    X_Type X = X_initial;

    for (std::size_t k = 0; k < NP; k++) {
      X = this->calculate_state_function(X, U_horizon, parameter);

      MatrixOperation::set_row(X_horizon, X, k + 1);
    }

    return X_horizon;
  }

  inline auto calculate_Y_limit_penalty(const Y_Horizon_Type &Y_horizon)
      -> Y_Horizon_Type {
    Y_Horizon_Type Y_limit_penalty;

    for (std::size_t i = 0; i < OUTPUT_SIZE; i++) {
      for (std::size_t j = 0; j < (NP + 1); j++) {
        if (Y_horizon(i, j) < this->_Y_min_matrix(i, j)) {
          Y_limit_penalty(i, j) = Y_horizon(i, j) - this->_Y_min_matrix(i, j);
        } else if (Y_horizon(i, j) > this->_Y_max_matrix(i, j)) {
          Y_limit_penalty(i, j) = Y_horizon(i, j) - this->_Y_max_matrix(i, j);
        } else {
          /* Do Nothing. */
        }
      }
    }

    return Y_limit_penalty;
  }

  inline void
  calculate_Y_limit_penalty_and_active(const Y_Horizon_Type &Y_horizon,
                                       Y_Horizon_Type &Y_limit_penalty,
                                       Y_Horizon_Type &Y_limit_active) {
    Y_limit_penalty = Y_Horizon_Type();
    Y_limit_active = Y_Horizon_Type();

    for (std::size_t i = 0; i < OUTPUT_SIZE; i++) {
      for (std::size_t j = 0; j < (NP + 1); j++) {

        if (Y_horizon(i, j) < this->_Y_min_matrix(i, j)) {
          Y_limit_penalty(i, j) = Y_horizon(i, j) - this->_Y_min_matrix(i, j);
          Y_limit_active(i, j) = static_cast<_T>(1);

        } else if (Y_horizon(i, j) > this->_Y_max_matrix(i, j)) {
          Y_limit_penalty(i, j) = Y_horizon(i, j) - this->_Y_max_matrix(i, j);
          Y_limit_active(i, j) = static_cast<_T>(1);
        } else {
          /* Do Nothing. */
        }
      }
    }
  }

  inline auto compute_cost(const X_Type X_initial,
                           const U_Horizon_Type &U_horizon) -> _T {

    auto X_horizon = this->simulate_trajectory(X_initial, U_horizon,
                                               this->state_space_parameters);

    Y_Horizon_Type Y_horizon;
    PythonNumpy::update_tile_concatenated_matrix<1, (NP + 1), Y_Type>(
        Y_horizon, this->_Y_offset);

    for (std::size_t k = 0; k < (NP + 1); k++) {
      auto Y_k = this->calculate_measurement_function(
          MatrixOperation::get_row(X_horizon, k), this->state_space_parameters);

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
          MatrixOperation::get_row(X_horizon, k), this->state_space_parameters);

      MatrixOperation::set_row(Y_horizon, Y_k, k);
    }

    auto Y_limit_penalty = this->calculate_Y_limit_penalty(Y_horizon);

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

  inline auto hvp_analytic(const X_Type &X_initial,
                           const U_Horizon_Type &U_horizon,
                           const _V_horizon_Type &V_horizon) -> _HVP_Type {

    /*
            # --- 1) forward states
            X = self.simulate_trajectory(
                X_initial, U_horizon, self.state_space_parameters)
            Y = np.zeros((self.ny, self.Np + 1))
            for k in range(self.Np + 1):
                Y[:, k] = self.calculate_measurement_function(
                    X[:, k], self.state_space_parameters).flatten()
            yN = self.calculate_measurement_function(
                X[:, self.Np], self.state_space_parameters)

            eN_y = yN - self.reference_trajectory[:, self.Np].reshape(-1, 1)

            Y_limit_penalty, Y_limit_active = \
                self.calculate_Y_limit_penalty_and_active(Y)

            # --- 2) first-order adjoint (costate lambda) with output terms
            lam = np.zeros((self.nx, self.Np + 1))
            Cx_N = self.calculate_measurement_jacobian_x(
                X[:, self.Np], self.state_space_parameters)

            lam[:, self.Np] = 2.0 * self.Px @ X[:, self.Np] + \
                (Cx_N.T @ ((2.0 * self.Py @ eN_y).flatten() + 2.0 *
                           self.Y_min_max_rho * Y_limit_penalty[:, self.Np])
                 ).flatten()

            for k in range(self.Np - 1, -1, -1):
                A_k = self.calculate_state_jacobian_x(
                    X[:, k], U_horizon[:, k], self.state_space_parameters)
                Cx_k = self.calculate_measurement_jacobian_x(
                    X[:, k], self.state_space_parameters)
                ek_y = Y[:, k] - self.reference_trajectory[:, k]

                lam[:, k] = 2.0 * self.Qx @ X[:, k] + \
                    Cx_k.T @ (2.0 * self.Qy @ ek_y +
                              2.0 * self.Y_min_max_rho * Y_limit_penalty[:, k])
       + \ A_k.T @ lam[:, k + 1]

            # --- 3) forward directional state: delta_x ---
            dx = np.zeros((self.nx, self.Np + 1))
            for k in range(self.Np):
                A_k = self.calculate_state_jacobian_x(
                    X[:, k], U_horizon[:, k], self.state_space_parameters)
                B_k = self.calculate_state_jacobian_u(
                    X[:, k], U_horizon[:, k], self.state_space_parameters)
                dx[:, k + 1] = A_k @ dx[:, k] + B_k @ V_horizon[:, k]

            # --- 4) backward second-order adjoint ---
            d_lambda = np.zeros((self.nx, self.Np + 1))

            # Match the treatment of the terminal term phi_xx = l_xx(X_N,·)
       (currently 2P) # Additionally, contributions from pure second-order
       output and second derivatives of output l_xx_dx = self.l_xx(X[:,
       self.Np], None) @ dx[:, self.Np] CX_N_dx = Cx_N @ dx[:, self.Np]

            CX_N_T_Py_Cx_N_dx = Cx_N.T @ (2.0 * self.Py @ CX_N_dx)
            CX_N_T_penalty_CX_N_dx = Cx_N.T @ ((2.0 * self.Y_min_max_rho)
                                               * (Y_limit_active[:, self.Np] *
       CX_N_dx))

            Hxx_penalty_term_N = self.hxx_lambda_contract(
                X[:, self.Np], self.state_space_parameters,
                2.0 * self.Y_min_max_rho *
                Y_limit_penalty[:, self.Np], dx[:, self.Np]
            )

            d_lambda[:, self.Np] += \
                l_xx_dx.flatten() + \
                CX_N_T_Py_Cx_N_dx.flatten() + \
                Hxx_penalty_term_N.flatten()

            d_lambda[:, self.Np] += CX_N_T_penalty_CX_N_dx.flatten()

            Hu = np.zeros_like(U_horizon)
            for k in range(self.Np - 1, -1, -1):
                A_k = self.calculate_state_jacobian_x(
                    X[:, k], U_horizon[:, k], self.state_space_parameters)
                B_k = self.calculate_state_jacobian_u(
                    X[:, k], U_horizon[:, k], self.state_space_parameters)
                Cx_k = self.calculate_measurement_jacobian_x(
                    X[:, k], self.state_space_parameters)
                ek_y = Y[:, k] - self.reference_trajectory[:, k]

                Cx_dx_k = Cx_k @ dx[:, k]
                term_Qy_GN = Cx_k.T @ (2.0 * self.Qy @ Cx_dx_k)
                term_Qy_hxx = self.hxx_lambda_contract(
                    X[:, k], self.state_space_parameters,
                    2.0 * self.Qy @ ek_y, dx[:, k]
                )

                term_penalty_GN = Cx_k.T @ ((2.0 * self.Y_min_max_rho)
                                            * (Y_limit_active[:, k] * Cx_dx_k))
                term_penalty_hxx = self.hxx_lambda_contract(
                    X[:, k], self.state_space_parameters,
                    2.0 * self.Y_min_max_rho * Y_limit_penalty[:, k], dx[:, k]
                )

                term_xx = self.fx_xx_lambda_contract(
                    X[:, k], U_horizon[:, k], self.state_space_parameters,
       lam[:, k + 1], dx[:, k]) term_xu = self.fx_xu_lambda_contract( X[:, k],
       U_horizon[:, k], self.state_space_parameters, lam[:, k + 1], V_horizon[:,
       k])

                d_lambda[:, k] = \
                    (self.l_xx(X[:, k], U_horizon[:, k]) @ dx[:, k]).flatten() +
       \
                    (self.l_xu(X[:, k], U_horizon[:, k]) @ V_horizon[:,
       k]).flatten() + \
                    (A_k.T @ d_lambda[:, k + 1]).flatten() + \
                    term_Qy_GN.flatten() + \
                    term_Qy_hxx.flatten() + \
                    term_penalty_GN.flatten() + \
                    term_penalty_hxx.flatten() + \
                    term_xx.flatten() + \
                    term_xu.flatten()

                # (HV)_k:
                #   2R V + B^T dlambda_{k+1} + second-order terms from dynamics
                #   (Cu=0 -> no direct contribution from output terms)
                term_ux = self.fu_xx_lambda_contract(
                    X[:, k], U_horizon[:, k], self.state_space_parameters,
       lam[:, k + 1], dx[:, k]) term_uu = self.fu_uu_lambda_contract( X[:, k],
       U_horizon[:, k], self.state_space_parameters, lam[:, k + 1], V_horizon[:,
       k])

                Hu[:, k] = \
                    (self.l_uu(X[:, k], U_horizon[:, k]) @ V_horizon[:,
       k]).flatten() + \
                    (self.l_ux(X[:, k], U_horizon[:, k]) @ dx[:, k]).flatten() +
       \
                    (B_k.T @ d_lambda[:, k + 1]).flatten() + \
                    term_ux.flatten() + term_uu.flatten()

            return Hu
    */

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

    /* --- 2) first-order adjoint (costate lambda) with output terms */
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
      auto A_k_T_lam =
          PythonNumpy::ATranspose_mul_B(A_k, MatrixOperation::get_row(lam, k));

      auto lam_input =
          static_cast<_T>(2) *
              (Qx_X + PythonNumpy::ATranspose_mul_B(
                          Cx_k, (Qy_ek_y + Y_min_max_rho_Yk_limit_penalty))) +
          A_k_T_lam;
      MatrixOperation::set_row(lam, lam_input, k);
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
     * Match the treatment of the terminal term phi_xx = l_xx(X_N,·) (currently
     * 2P) Additionally, contributions from pure second-order output and second
     * derivatives of output
     */
    auto l_xx_dx =
        this->l_xx(MatrixOperation::get_row(X_horizon, NP), U_dummy) *
        MatrixOperation::get_row(dx, NP);
    auto Cx_N_dx = Cx_N * MatrixOperation::get_row(dx, NP);

    auto CX_N_T_Py_Cx_N_dx = PythonNumpy::ATranspose_mul_B(
        Cx_N, (static_cast<_T>(2) * this->_Py * Cx_N_dx));

    auto Y_min_max_rho_YN_limit_active_CX_N_dx =
        static_cast<_T>(2) * this->_Y_min_max_rho *
        MatrixOperation::get_row(Y_limit_active, NP) * Cx_N_dx;

    auto CX_N_T_penalty_CX_N_dx = PythonNumpy::ATranspose_mul_B(
        Cx_N, Y_min_max_rho_YN_limit_active_CX_N_dx);

    auto Y_min_max_rho_YN_limit_penalty =
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

      auto Y_min_max_rho_Yk_limit_active_Cx_dx_k =
          static_cast<_T>(2) * this->_Y_min_max_rho *
          MatrixOperation::get_row(Y_limit_active, k) * Cx_dx_k;

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

      auto l_xx_dx = this->l_xx(MatrixOperation::get_row(X_horizon, k),
                                MatrixOperation::get_row(U_horizon, k)) *
                     MatrixOperation::get_row(dx, k);

      auto l_xu_V = this->l_xu(MatrixOperation::get_row(X_horizon, k),
                               MatrixOperation::get_row(U_horizon, k)) *
                    MatrixOperation::get_row(V_horizon, k);

      auto A_k_T_d_lambda = PythonNumpy::ATranspose_mul_B(
          A_k, MatrixOperation::get_row(d_lambda, k + 1));

      auto d_lambda_input = l_xx_dx + l_xu_V + A_k_T_d_lambda + term_Qy_GN +
                            term_Qy_hxx + term_penalty_GN + term_penalty_hxx +
                            term_xx + term_xu;

      MatrixOperation::set_row(d_lambda, d_lambda_input, k);

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
