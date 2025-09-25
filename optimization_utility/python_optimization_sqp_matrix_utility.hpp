#ifndef __PYTHON_OPTIMIZATION_SQP_MATRIX_UTILITY_HPP__
#define __PYTHON_OPTIMIZATION_SQP_MATRIX_UTILITY_HPP__

#include "python_numpy.hpp"

#include <array>
#include <functional>
#include <type_traits>

namespace PythonOptimization {

static constexpr double Y_MIN_MAX_RHO_FACTOR_DEFAULT = 1.0e2;

template <typename T, std::size_t State_Size_In, std::size_t Input_Size_In,
          std::size_t Output_Size_In, std::size_t Np_In,
          typename Parameter_Type_In>
class SQP_CostMatrices_NMPC {
public:
  /* Constant */
  static constexpr std::size_t STATE_SIZE = State_Size_In;
  static constexpr std::size_t INPUT_SIZE = Input_Size_In;
  static constexpr std::size_t OUTPUT_SIZE = Output_Size_In;
  static constexpr std::size_t NP = Np_In;

public:
  /* Type */
  using Value_Type = T;

protected:
  /* Type */
  using _T = T;
  using _Parameter_Type = Parameter_Type_In;

  using Qx_Type = PythonNumpy::DiagMatrix_Type<_T, STATE_SIZE>;
  using R_Type = PythonNumpy::DiagMatrix_Type<_T, INPUT_SIZE>;
  using Qy_Type = PythonNumpy::DiagMatrix_Type<_T, OUTPUT_SIZE>;
};

} // namespace PythonOptimization

#endif // __PYTHON_OPTIMIZATION_SQP_MATRIX_UTILITY_HPP__
