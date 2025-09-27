#ifndef __PYTHON_OPTIMIZATION_SQP_ACTIVE_SET_PCG_PLS_HPP__
#define __PYTHON_OPTIMIZATION_SQP_ACTIVE_SET_PCG_PLS_HPP__

#include "python_optimization_utility.hpp"

#include "python_control.hpp"
#include "python_numpy.hpp"

#include <array>
#include <functional>
#include <type_traits>

namespace PythonOptimization {

static constexpr double RHS_NORM_ZERO_LIMIT_DEFAULT = 1e-12;

static constexpr double GRADIENT_NORM_ZERO_LIMIT_DEFAULT = 1e-6;

static constexpr double U_NEAR_LIMIT_DEFAULT = 1e-12;
static constexpr double GRADIENT_ZERO_LIMIT_DEFAULT = 1e-12;

static constexpr double PCG_TOL_DEFAULT = 1e-4;
static constexpr std::size_t PCG_MAX_ITERATION_DEFAULT = 30;
static constexpr double PCG_PHP_MINUS_LIMIT_DEFAULT = 1e-14;

static constexpr std::size_t LINE_SEARCH_MAX_ITERATION_DEFAULT = 20;

static constexpr double ALPHA_SMALL_LIMIT_DEFAULT = 1e-6;
static constexpr double ALPHA_DECAY_RATE_DEFAULT = 0.5;

static constexpr std::size_t SOLVER_MAX_ITERATION_DEFAULT = 100;

static constexpr double LAMBDA_FACTOR_DEFAULT = 1e-6;

/* SQP Active Set with PCG and PLS */

template <typename CostMatrices_Type_In> class SQP_ActiveSet_PCG_PLS {
public:
  /* Constant */
  static constexpr std::size_t STATE_SIZE = CostMatrices_Type_In::STATE_SIZE;
  static constexpr std::size_t INPUT_SIZE = CostMatrices_Type_In::INPUT_SIZE;
  static constexpr std::size_t OUTPUT_SIZE = CostMatrices_Type_In::OUTPUT_SIZE;

  static constexpr std::size_t NP = CostMatrices_Type_In::NP;

public:
  /* Type */
  using Value_Type = typename CostMatrices_Type_In::Value_Type;

  using X_Horizon_Type = typename CostMatrices_Type_In::X_Horizon_Type;
  using U_Horizon_Type = typename CostMatrices_Type_In::U_Horizon_Type;
  using Y_Horizon_Type = typename CostMatrices_Type_In::Y_Horizon_Type;

  /* Check Compatibility */

protected:
  /* Type */
  using _CostMatrices_Type = CostMatrices_Type_In;

  using _T = Value_Type;

  using _R_Full_Size =
      PythonNumpy::DenseMatrix_Type<_T, INPUT_SIZE, INPUT_SIZE>;

  using Mask_Type = U_Horizon_Type;

  using ActiveSet_Type = ActiveSet2D_Type<INPUT_SIZE, NP>;

public:
  /* Constructor */
  SQP_ActiveSet_PCG_PLS();

public:
  /* Function */

public:
  /* Variable */

protected:
  /* Variable */
  _T _gradient_norm_zero_limit;
  _T _alpha_small_limit;
  _T _alpha_decay_rate;
  _T _pcg_php_minus_limit;

  std::size_t _solver_max_iteration;
  std::size_t _pcg_max_iteration;
  std::size_t _line_search_max_iteration;

  _T _pcg_tol;
  _T _lambda_factor;

  _R_Full_Size _R_full_size;
};

} // namespace PythonOptimization

#endif // __PYTHON_OPTIMIZATION_SQP_ACTIVE_SET_PCG_PLS_HPP__
