#ifndef __PYTHON_OPTIMIZATION_SQP_MATRIX_UTILITY_HPP__
#define __PYTHON_OPTIMIZATION_SQP_MATRIX_UTILITY_HPP__

#include "python_control.hpp"
#include "python_numpy.hpp"

#include <array>
#include <functional>
#include <type_traits>

namespace PythonOptimization {

static constexpr double Y_MIN_MAX_RHO_FACTOR_DEFAULT = 1.0e2;

template <typename T, std::size_t Np_In, std::size_t State_Size_In,
          std::size_t Input_Size_In, std::size_t Output_Size_In,
          typename Parameter_Type_In, typename U_Min_Type_In,
          typename U_Max_Type_In, typename Y_Min_Type_In,
          typename Y_Max_Type_In>
class SQP_CostMatrices_NMPC {
public:
  /* Constant */
  static constexpr std::size_t STATE_SIZE = State_Size_In;
  static constexpr std::size_t INPUT_SIZE = Input_Size_In;
  static constexpr std::size_t OUTPUT_SIZE = Output_Size_In;

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
};

} // namespace PythonOptimization

#endif // __PYTHON_OPTIMIZATION_SQP_MATRIX_UTILITY_HPP__
