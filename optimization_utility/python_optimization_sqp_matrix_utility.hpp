#ifndef __PYTHON_OPTIMIZATION_SQP_MATRIX_UTILITY_HPP__
#define __PYTHON_OPTIMIZATION_SQP_MATRIX_UTILITY_HPP__

#include "python_numpy.hpp"

#include <array>
#include <functional>
#include <type_traits>

namespace PythonOptimization {

static constexpr double Y_MIN_MAX_RHO_FACTOR_DEFAULT = 1.0e2;

template <typename T, std::size_t Np_In, typename Parameter_Type_In,
          typename U_Min_Type_In, typename U_Max_Type_In,
          typename Y_Min_Type_In, typename Y_Max_Type_In,
          typename State_Jacobian_X_Type_In, typename State_Jacobian_U_Type_In,
          typename Measurement_Jacobian_X_Type_In,
          typename State_Hessian_XX_Type_In, typename State_Hessian_XU_Type_In,
          typename State_Hessian_UX_Type_In, typename State_Hessian_UU_Type_In,
          typename Measurement_Hessian_XX_Type_In>
class SQP_CostMatrices_NMPC {
public:
  /* Constant */
  static constexpr std::size_t STATE_SIZE = State_Jacobian_X_Type_In::COLS;
  static constexpr std::size_t INPUT_SIZE = State_Jacobian_U_Type_In::COLS;
  static constexpr std::size_t OUTPUT_SIZE =
      Measurement_Jacobian_X_Type_In::COLS;
  static constexpr std::size_t NP = Np_In;

public:
  /* Type */
  using Value_Type = T;

  /* Check Compatibility */
  static_assert(std::is_same<U_Min_Type_In::Value_Type, T>::value,
                "U_Min_Type_In::Value_Type != T");
  static_assert(std::is_same<U_Max_Type_In::Value_Type, T>::value,
                "U_Max_Type_In::Value_Type != T");
  static_assert(std::is_same<Y_Min_Type_In::Value_Type, T>::value,
                "Y_Min_Type_In::Value_Type != T");
  static_assert(std::is_same<Y_Max_Type_In::Value_Type, T>::value,
                "Y_Max_Type_In::Value_Type != T");

  static_assert(std::is_same<State_Jacobian_X_Type_In::Value_Type, T>::value,
                "State_Jacobian_X_Type_In::Value_Type != T");
  static_assert(std::is_same<State_Jacobian_U_Type_In::Value_Type, T>::value,
                "State_Jacobian_U_Type_In::Value_Type != T");
  static_assert(
      std::is_same<Measurement_Jacobian_X_Type_In::Value_Type, T>::value,
      "Measurement_Jacobian_X_Type_In::Value_Type != T");

  static_assert(std::is_same<State_Hessian_XX_Type_In::Value_Type, T>::value,
                "State_Hessian_XX_Type_In::Value_Type != T");
  static_assert(std::is_same<State_Hessian_XU_Type_In::Value_Type, T>::value,
                "State_Hessian_XU_Type_In::Value_Type != T");
  static_assert(std::is_same<State_Hessian_UX_Type_In::Value_Type, T>::value,
                "State_Hessian_UX_Type_In::Value_Type != T");
  static_assert(std::is_same<State_Hessian_UU_Type_In::Value_Type, T>::value,
                "State_Hessian_UU_Type_In::Value_Type != T");
  static_assert(
      std::is_same<Measurement_Hessian_XX_Type_In::Value_Type, T>::value,
      "Measurement_Hessian_XX_Type_In::Value_Type != T");

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

  static_assert(State_Jacobian_X_Type_In::COLS == STATE_SIZE,
                "State_Jacobian_X_Type_In::COLS != STATE_SIZE");
  static_assert(State_Jacobian_X_Type_In::ROWS == STATE_SIZE,
                "State_Jacobian_X_Type_In::ROWS != STATE_SIZE");

  static_assert(State_Jacobian_U_Type_In::COLS == STATE_SIZE,
                "State_Jacobian_U_Type_In::COLS != STATE_SIZE");
  static_assert(State_Jacobian_U_Type_In::ROWS == INPUT_SIZE,
                "State_Jacobian_U_Type_In::ROWS != INPUT_SIZE");

  static_assert(Measurement_Jacobian_X_Type_In::COLS == OUTPUT_SIZE,
                "Measurement_Jacobian_X_Type_In::COLS != OUTPUT_SIZE");
  static_assert(Measurement_Jacobian_X_Type_In::ROWS == STATE_SIZE,
                "Measurement_Jacobian_X_Type_In::ROWS != STATE_SIZE");

  static_assert(State_Hessian_XX_Type_In::COLS == STATE_SIZE * STATE_SIZE,
                "State_Hessian_XX_Type_In::COLS != STATE_SIZE * STATE_SIZE");
  static_assert(State_Hessian_XX_Type_In::ROWS == STATE_SIZE,
                "State_Hessian_XX_Type_In::ROWS != STATE_SIZE");

  static_assert(State_Hessian_XU_Type_In::COLS == STATE_SIZE * STATE_SIZE,
                "State_Hessian_XU_Type_In::COLS != STATE_SIZE * INPUT_SIZE");
  static_assert(State_Hessian_XU_Type_In::ROWS == INPUT_SIZE,
                "State_Hessian_XU_Type_In::ROWS != INPUT_SIZE");

  static_assert(State_Hessian_UX_Type_In::COLS == INPUT_SIZE * STATE_SIZE,
                "State_Hessian_UX_Type_In::COLS != INPUT_SIZE * STATE_SIZE");
  static_assert(State_Hessian_UX_Type_In::ROWS == STATE_SIZE,
                "State_Hessian_UX_Type_In::ROWS != STATE_SIZE");

  static_assert(State_Hessian_UU_Type_In::COLS == INPUT_SIZE * STATE_SIZE,
                "State_Hessian_UU_Type_In::COLS != INPUT_SIZE * STATE_SIZE");
  static_assert(State_Hessian_UU_Type_In::ROWS == INPUT_SIZE,
                "State_Hessian_UU_Type_In::ROWS != INPUT_SIZE");

  static_assert(
      Measurement_Hessian_XX_Type_In::COLS == OUTPUT_SIZE * STATE_SIZE,
      "Measurement_Hessian_XX_Type_In::COLS != OUTPUT_SIZE * STATE_SIZE");
  static_assert(Measurement_Hessian_XX_Type_In::ROWS == STATE_SIZE,
                "Measurement_Hessian_XX_Type_In::ROWS != STATE_SIZE");

protected:
  /* Type */
  using _T = T;
  using _Parameter_Type = Parameter_Type_In;

  using _Qx_Type = PythonNumpy::DiagMatrix_Type<_T, STATE_SIZE>;
  using _R_Type = PythonNumpy::DiagMatrix_Type<_T, INPUT_SIZE>;
  using _Qy_Type = PythonNumpy::DiagMatrix_Type<_T, OUTPUT_SIZE>;

  using _U_Min_Matrix_Type = PythonNumpy::Tile_Type<1, NP, U_Min_Type_In>;

  using _U_Max_Matrix_Type = PythonNumpy::Tile_Type<1, NP, U_Max_Type_In>;
  using _Y_Min_Matrix_Type = PythonNumpy::Tile_Type<1, (NP + 1), Y_Min_Type_In>;
  using _Y_Max_Matrix_Type = PythonNumpy::Tile_Type<1, (NP + 1), Y_Max_Type_In>;

  using _State_Jacobian_X_Type = State_Jacobian_X_Type_In;
  using _State_Jacobian_U_Type = State_Jacobian_U_Type_In;
  using _Measurement_Jacobian_X_Type = Measurement_Jacobian_X_Type_In;

  using _State_Hessian_XX_Type = State_Hessian_XX_Type_In;
  using _State_Hessian_XU_Type = State_Hessian_XU_Type_In;
  using _State_Hessian_UX_Type = State_Hessian_UX_Type_In;
  using _State_Hessian_UU_Type = State_Hessian_UU_Type_In;
  using _Measurement_Hessian_XX_Type = Measurement_Hessian_XX_Type_In;

protected:
  /* Variable */
  _T Y_min_max_rho;
};

} // namespace PythonOptimization

#endif // __PYTHON_OPTIMIZATION_SQP_MATRIX_UTILITY_HPP__
