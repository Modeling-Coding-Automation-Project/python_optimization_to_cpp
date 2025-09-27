#ifndef __PYTHON_OPTIMIZATION_SQP_ACTIVE_SET_PCG_PLS_HPP__
#define __PYTHON_OPTIMIZATION_SQP_ACTIVE_SET_PCG_PLS_HPP__

#include "python_optimization_utility.hpp"

#include "python_control.hpp"
#include "python_numpy.hpp"

#include <array>
#include <functional>
#include <type_traits>
#include <utility>

namespace PythonOptimization {

static constexpr double AVOID_ZERO_DIVIDE_LIMIT = 1e-12;

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

namespace MatrixOperation {

template <typename Matrix_Type>
inline auto add_scalar_to_matrix(Matrix_Type &matrix,
                                 typename Matrix_Type::Value_Type scalar)
    -> Matrix_Type {

  Matrix_Type out = matrix;

  for (std::size_t i = 0; i < Matrix_Type::COLS; i++) {
    for (std::size_t j = 0; j < Matrix_Type::ROWS; j++) {
      out(i, j) += scalar;
    }
  }

  return out;
}

} // namespace MatrixOperation

/* Cost Function Objects */
template <typename X_Type, typename U_Horizon_Type>
using CostFunction_Object = std::function<typename X_Type::Value_Type(
    const X_Type &, const U_Horizon_Type &)>;

template <typename X_Type, typename U_Horizon_Type, typename Gradient_Type>
using CostAndGradientFunction_Object =
    std::function<void(const X_Type &, const U_Horizon_Type &,
                       typename X_Type::Value_Type &, Gradient_Type &)>;

template <typename X_Type, typename U_Horizon_Type, typename V_Horizon_Type,
          typename HVP_Type>
using HVP_Function_Object = std::function<HVP_Type(
    const X_Type &, const U_Horizon_Type &, const V_Horizon_Type &)>;

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

  using X_Type = typename CostMatrices_Type_In::X_Type;
  using U_Type = typename CostMatrices_Type_In::U_Type;
  using Y_Type = typename CostMatrices_Type_In::Y_Type;

  using X_Horizon_Type = typename CostMatrices_Type_In::X_Horizon_Type;
  using U_Horizon_Type = typename CostMatrices_Type_In::U_Horizon_Type;
  using Y_Horizon_Type = typename CostMatrices_Type_In::Y_Horizon_Type;

protected:
  /* Type */
  using _CostMatrices_Type = CostMatrices_Type_In;

  using _T = Value_Type;

  using _R_Full_Type =
      PythonNumpy::DenseMatrix_Type<_T, INPUT_SIZE, INPUT_SIZE>;

  using _Mask_Type = U_Horizon_Type;
  using _Gradient_Type = U_Horizon_Type;
  using _V_Horizon_Type = U_Horizon_Type;
  using _HVP_Type = U_Horizon_Type;
  using _RHS_Type = U_Horizon_Type;
  using _M_Inv_Type = U_Horizon_Type;

  using _ActiveSet_Type = ActiveSet2D_Type<INPUT_SIZE, NP>;

  using _Cost_Function_Object_Type =
      CostFunction_Object<X_Type, U_Horizon_Type>;

  using _Cost_And_Gradient_Function_Object_Type =
      CostAndGradientFunction_Object<X_Type, U_Horizon_Type, _Gradient_Type>;

  using _HVP_Function_Object_Type =
      HVP_Function_Object<X_Type, U_Horizon_Type, _V_Horizon_Type, _HVP_Type>;

  using _U_min_Type = typename _CostMatrices_Type::U_Min_Type;
  using _U_max_Type = typename _CostMatrices_Type::U_Max_Type;
  using _Y_min_Type = typename _CostMatrices_Type::Y_Min_Type;
  using _Y_max_Type = typename _CostMatrices_Type::Y_Max_Type;

  using _U_Min_Matrix_Type = PythonNumpy::Tile_Type<1, NP, _U_min_Type>;
  using _U_Max_Matrix_Type = PythonNumpy::Tile_Type<1, NP, _U_max_Type>;
  using _Y_Min_Matrix_Type = PythonNumpy::Tile_Type<1, (NP + 1), _Y_min_Type>;
  using _Y_Max_Matrix_Type = PythonNumpy::Tile_Type<1, (NP + 1), _Y_max_Type>;

  using _At_Lower_Upper_Type =
      PythonNumpy::DenseMatrix_Type<bool, INPUT_SIZE, NP>;

public:
  /* Constructor */
  SQP_ActiveSet_PCG_PLS()
      : U_horizon(), X_initial(), hvp_function(nullptr),
        _gradient_norm_zero_limit(
            static_cast<_T>(GRADIENT_NORM_ZERO_LIMIT_DEFAULT)),
        _alpha_small_limit(static_cast<_T>(ALPHA_SMALL_LIMIT_DEFAULT)),
        _alpha_decay_rate(static_cast<_T>(ALPHA_DECAY_RATE_DEFAULT)),
        _pcg_php_minus_limit(static_cast<_T>(PCG_PHP_MINUS_LIMIT_DEFAULT)),
        _solver_max_iteration(SOLVER_MAX_ITERATION_DEFAULT),
        _pcg_max_iteration(PCG_MAX_ITERATION_DEFAULT),
        _line_search_max_iteration(LINE_SEARCH_MAX_ITERATION_DEFAULT),
        _pcg_tol(static_cast<_T>(PCG_TOL_DEFAULT)),
        _lambda_factor(static_cast<_T>(LAMBDA_FACTOR_DEFAULT)),
        _diag_R_full(_R_Full_Type::ones()), _mask(), _active_set(),
        _solver_step_iterated_number(0), _pcg_step_iterated_number(0),
        _line_search_step_iterated_number(0), _J_optimal(static_cast<_T>(0)) {}

  /* Copy constructor */
  SQP_ActiveSet_PCG_PLS(const SQP_ActiveSet_PCG_PLS &input)
      : U_horizon(input.U_horizon), X_initial(input.X_initial),
        hvp_function(input.hvp_function),
        _gradient_norm_zero_limit(input._gradient_norm_zero_limit),
        _alpha_small_limit(input._alpha_small_limit),
        _alpha_decay_rate(input._alpha_decay_rate),
        _pcg_php_minus_limit(input._pcg_php_minus_limit),
        _solver_max_iteration(input._solver_max_iteration),
        _pcg_max_iteration(input._pcg_max_iteration),
        _line_search_max_iteration(input._line_search_max_iteration),
        _pcg_tol(input._pcg_tol), _lambda_factor(input._lambda_factor),
        _diag_R_full(input._diag_R_full), _mask(input._mask),
        _active_set(input._active_set),
        _solver_step_iterated_number(input._solver_step_iterated_number),
        _pcg_step_iterated_number(input._pcg_step_iterated_number),
        _line_search_step_iterated_number(
            input._line_search_step_iterated_number),
        _J_optimal(input._J_optimal) {}

  /* Copy assignment */
  SQP_ActiveSet_PCG_PLS &operator=(const SQP_ActiveSet_PCG_PLS &input) {
    if (this != &input) {
      this->U_horizon = input.U_horizon;
      this->X_initial = input.X_initial;
      this->hvp_function = input.hvp_function;

      this->_gradient_norm_zero_limit = input._gradient_norm_zero_limit;
      this->_alpha_small_limit = input._alpha_small_limit;
      this->_alpha_decay_rate = input._alpha_decay_rate;
      this->_pcg_php_minus_limit = input._pcg_php_minus_limit;

      this->_solver_max_iteration = input._solver_max_iteration;
      this->_pcg_max_iteration = input._pcg_max_iteration;
      this->_line_search_max_iteration = input._line_search_max_iteration;

      this->_pcg_tol = input._pcg_tol;
      this->_lambda_factor = input._lambda_factor;

      this->_diag_R_full = input._diag_R_full;

      this->_mask = input._mask;
      this->_active_set = input._active_set;

      this->_solver_step_iterated_number = input._solver_step_iterated_number;
      this->_pcg_step_iterated_number = input._pcg_step_iterated_number;
      this->_line_search_step_iterated_number =
          input._line_search_step_iterated_number;

      this->_J_optimal = input._J_optimal;
    }
    return *this;
  }

  /* Move constructor */
  SQP_ActiveSet_PCG_PLS(SQP_ActiveSet_PCG_PLS &&input) noexcept
      : U_horizon(std::move(input.U_horizon)),
        X_initial(std::move(input.X_initial)),
        hvp_function(std::move(input.hvp_function)),
        _gradient_norm_zero_limit(input._gradient_norm_zero_limit),
        _alpha_small_limit(input._alpha_small_limit),
        _alpha_decay_rate(input._alpha_decay_rate),
        _pcg_php_minus_limit(input._pcg_php_minus_limit),
        _solver_max_iteration(input._solver_max_iteration),
        _pcg_max_iteration(input._pcg_max_iteration),
        _line_search_max_iteration(input._line_search_max_iteration),
        _pcg_tol(input._pcg_tol), _lambda_factor(input._lambda_factor),
        _diag_R_full(std::move(input._diag_R_full)),
        _mask(std::move(input._mask)),
        _active_set(std::move(input._active_set)),
        _solver_step_iterated_number(input._solver_step_iterated_number),
        _pcg_step_iterated_number(input._pcg_step_iterated_number),
        _line_search_step_iterated_number(
            input._line_search_step_iterated_number),
        _J_optimal(input._J_optimal) {}

  /* Move assignment */
  SQP_ActiveSet_PCG_PLS &operator=(SQP_ActiveSet_PCG_PLS &&input) noexcept {
    if (this != &input) {
      this->U_horizon = std::move(input.U_horizon);
      this->X_initial = std::move(input.X_initial);
      this->hvp_function = std::move(input.hvp_function);

      this->_gradient_norm_zero_limit = input._gradient_norm_zero_limit;
      this->_alpha_small_limit = input._alpha_small_limit;
      this->_alpha_decay_rate = input._alpha_decay_rate;
      this->_pcg_php_minus_limit = input._pcg_php_minus_limit;

      this->_solver_max_iteration = input._solver_max_iteration;
      this->_pcg_max_iteration = input._pcg_max_iteration;
      this->_line_search_max_iteration = input._line_search_max_iteration;

      this->_pcg_tol = input._pcg_tol;
      this->_lambda_factor = input._lambda_factor;

      this->_diag_R_full = std::move(input._diag_R_full);

      this->_mask = std::move(input._mask);
      this->_active_set = std::move(input._active_set);

      this->_solver_step_iterated_number = input._solver_step_iterated_number;
      this->_pcg_step_iterated_number = input._pcg_step_iterated_number;
      this->_line_search_step_iterated_number =
          input._line_search_step_iterated_number;

      this->_J_optimal = input._J_optimal;
    }
    return *this;
  }

public:
  /* Setter */
  inline void set_gradient_norm_zero_limit(const _T &limit) {
    this->_gradient_norm_zero_limit = limit;
  }

  inline void set_alpha_small_limit(const _T &limit) {
    this->_alpha_small_limit = limit;
  }

  inline void set_alpha_decay_rate(const _T &rate) {
    this->_alpha_decay_rate = rate;
  }

  inline void set_pcg_php_minus_limit(const _T &limit) {
    this->_pcg_php_minus_limit = limit;
  }

  inline void set_solver_max_iteration(const std::size_t &max_iteration) {
    this->_solver_max_iteration = max_iteration;
  }

  inline void set_pcg_max_iteration(const std::size_t &max_iteration) {
    this->_pcg_max_iteration = max_iteration;
  }

  inline void set_line_search_max_iteration(const std::size_t &max_iteration) {
    this->_line_search_max_iteration = max_iteration;
  }

  inline void set_pcg_tol(const _T &tol) { this->_pcg_tol = tol; }

  inline void set_lambda_factor(const _T &factor) {
    this->_lambda_factor = factor;
  }

  inline void set_diag_R_full(const _R_Full_Type &diag_R_full) {
    this->_diag_R_full = diag_R_full;
  }

  /* Getter */
  inline auto get_solver_step_iterated_number(void) const
      -> const std::size_t & {
    return this->_solver_step_iterated_number;
  }

  inline auto get_pcg_step_iterated_number(void) const -> const std::size_t & {
    return this->_pcg_step_iterated_number;
  }

  inline auto get_line_search_step_iterated_number(void) const
      -> const std::size_t & {
    return this->_line_search_step_iterated_number;
  }

  /* Function */
  inline auto preconditioned_conjugate_gradient(const _RHS_Type &rhs,
                                                const _M_Inv_Type &M_inv)
      -> _RHS_Type {

    _RHS_Type d;

    auto rhs_norm = ActiveSet2D_MatrixOperator::norm(rhs, this->_active_set);
    if (rhs_norm < RHS_NORM_ZERO_LIMIT_DEFAULT) {
      /* Do Nothing. */
    } else {
      _RHS_Type r = rhs;

      /* Preconditioning */
      auto z = ActiveSet2D_MatrixOperator::element_wise_product<
          _T, INPUT_SIZE, NP, INPUT_SIZE, NP>(r, M_inv, this->_active_set);

      _RHS_Type p = z;

      auto rz = ActiveSet2D_MatrixOperator::vdot(r, z, this->_active_set);

      for (std::size_t pcg_iteration = 0;
           pcg_iteration < this->_pcg_max_iteration; ++pcg_iteration) {

        auto Hp = this->hvp_function(this->X_initial, this->U_horizon, p);
        Hp = Hp + this->_lambda_factor * p;

        _T denominator =
            ActiveSet2D_MatrixOperator::vdot(p, Hp, this->_active_set);

        /* Simple handling of negative curvature and semi-definiteness */
        this->_pcg_step_iterated_number = pcg_iteration + 1;
        if (denominator <= this->_pcg_php_minus_limit) {
          break;
        }

        _T alpha = rz / denominator;

        d = d + ActiveSet2D_MatrixOperator::matrix_multiply_scalar(
                    p, alpha, this->_active_set);

        r = r - ActiveSet2D_MatrixOperator::matrix_multiply_scalar(
                    Hp, alpha, this->_active_set);

        if (ActiveSet2D_MatrixOperator::norm(r, this->_active_set) <=
            this->_pcg_tol * rhs_norm) {
          break;
        }

        z = ActiveSet2D_MatrixOperator::element_wise_product(r, M_inv,
                                                             this->_active_set);

        _T rz_new = ActiveSet2D_MatrixOperator::vdot(r, z, this->_active_set);

        _T beta = rz_new / rz;

        p = z + ActiveSet2D_MatrixOperator::matrix_multiply_scalar(
                    p, beta, this->_active_set);

        rz = rz_new;
      }
    }

    return d;
  }

  inline auto free_mask(U_Horizon_Type &U_horizon_in, _Gradient_Type &gradient,
                        const _U_Min_Matrix_Type &U_min_matrix,
                        const _U_Max_Matrix_Type &U_max_matrix, const _T &atol,
                        const _T &gtol) -> _Mask_Type {

    auto m = _Mask_Type::ones();

    this->_active_set.clear();
    _At_Lower_Upper_Type at_lower;
    _At_Lower_Upper_Type at_upper;

    for (std::size_t i = 0; i < INPUT_SIZE; ++i) {
      for (std::size_t j = 0; j < NP; ++j) {

        if ((U_horizon_in(i, j) >= (U_min_matrix(i, j) - atol)) &&
            (U_horizon_in(i, j) <= (U_min_matrix(i, j) + atol))) {
          at_lower(i, j) = true;
        }

        if ((U_horizon_in(i, j) >= (U_max_matrix(i, j) - atol)) &&
            (U_horizon_in(i, j) <= (U_max_matrix(i, j) + atol))) {
          at_upper(i, j) = true;
        }
      }
    }

    for (std::size_t i = 0; i < INPUT_SIZE; ++i) {
      for (std::size_t j = 0; j < NP; ++j) {
        if ((at_lower(i, j) && (gradient(i, j) > gtol)) ||
            (at_upper(i, j) && (gradient(i, j) < -gtol))) {
          m(i, j) = false;
        } else {
          this->_active_set.push_active(i, j);
        }
      }
    }

    return m;
  }

  inline auto solve(
      const U_Horizon_Type &U_horizon_initial,
      const _Cost_And_Gradient_Function_Object_Type &cost_and_gradient_function,
      const _Cost_Function_Object_Type &cost_function,
      const _HVP_Function_Object_Type &hvp_function_in,
      const X_Type &X_initial_in, const _U_Min_Matrix_Type &U_min_matrix,
      const _U_Max_Matrix_Type &U_max_matrix) -> U_Horizon_Type {

    this->X_initial = X_initial_in;
    U_Horizon_Type U_horizon_store = U_horizon_initial;

    _T J = static_cast<_T>(0);

    for (std::size_t solver_iteration = 0;
         solver_iteration < this->_solver_max_iteration; ++solver_iteration) {
      /* Calculate cost and gradient */
      _Gradient_Type gradient;
      cost_and_gradient_function(X_initial, U_horizon_store, J, gradient);

      this->_solver_step_iterated_number = solver_iteration + 1;
      if (PythonNumpy::norm(gradient) < this->_gradient_norm_zero_limit) {
        break;
      }

      this->_mask =
          this->free_mask(U_horizon_store, gradient, U_min_matrix, U_max_matrix,
                          static_cast<_T>(U_NEAR_LIMIT_DEFAULT),
                          static_cast<_T>(GRADIENT_ZERO_LIMIT_DEFAULT));

      _RHS_Type rhs = -gradient;

      auto diag_R_full_lambda_factor = MatrixOperation::add_scalar_to_matrix(
          this->_diag_R_full, this->_lambda_factor);

      _M_Inv_Type M_inv;
      for (std::size_t i = 0; i < INPUT_SIZE; ++i) {
        for (std::size_t j = 0; j < NP; ++j) {
          M_inv(i, j) = static_cast<_T>(1) /
                        Base::Utility::avoid_zero_divide(
                            diag_R_full_lambda_factor(i, j),
                            static_cast<_T>(AVOID_ZERO_DIVIDE_LIMIT));
        }
      }

      this->U_horizon = U_horizon_store;
      this->hvp_function = hvp_function_in;

      auto d = this->preconditioned_conjugate_gradient(rhs, M_inv);

      /*
       * line search and projection
       * (No distinction between fixed/free is needed here,
       *  project the whole)
       */
      _T alpha = static_cast<_T>(1);
      U_Horizon_Type U_horizon_new = U_horizon;

      bool U_updated_flag = false;
      for (std::size_t line_search_iteration = 0;
           line_search_iteration < this->_line_search_max_iteration;
           ++line_search_iteration) {
        auto U_candidate = U_horizon + alpha * d;

        for (std::size_t i = 0; i < INPUT_SIZE; ++i) {
          for (std::size_t j = 0; j < NP; ++j) {
            if (U_candidate(i, j) < U_min_matrix(i, j)) {
              U_candidate(i, j) = U_min_matrix(i, j);
            } else if (U_candidate(i, j) > U_max_matrix(i, j)) {
              U_candidate(i, j) = U_max_matrix(i, j);
            }
          }
        }

        auto J_candidate = cost_function(X_initial, U_candidate);

        this->_line_search_step_iterated_number = line_search_iteration + 1;
        if (J_candidate <= J || alpha < this->_alpha_small_limit) {
          U_horizon_new = U_candidate;
          J = J_candidate;
          U_updated_flag = true;
          break;
        }

        alpha *= this->_alpha_decay_rate;
      }

      if (true == U_updated_flag) {
        U_horizon = U_horizon_new;
      } else {
        break;
      }
    }

    this->_J_optimal = J;

    return U_horizon;
  }

public:
  /* Variable */
  U_Horizon_Type U_horizon;
  X_Type X_initial;
  _HVP_Function_Object_Type hvp_function;

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

  _R_Full_Type _diag_R_full;

  _Mask_Type _mask;
  _ActiveSet_Type _active_set;

  std::size_t _solver_step_iterated_number;
  std::size_t _pcg_step_iterated_number;
  std::size_t _line_search_step_iterated_number;

  _T _J_optimal;
};

/* make SQP_ActiveSetPCG_PLS */
template <typename CostMatrices_Type>
inline auto make_SQP_ActiveSet_PCG_PLS(void)
    -> SQP_ActiveSet_PCG_PLS<CostMatrices_Type> {
  return SQP_ActiveSet_PCG_PLS<CostMatrices_Type>();
}

/* SQP_ActiveSetPCG_PLS Type */
template <typename CostMatrices_Type>
using SQP_ActiveSet_PCG_PLS_Type = SQP_ActiveSet_PCG_PLS<CostMatrices_Type>;

} // namespace PythonOptimization

#endif // __PYTHON_OPTIMIZATION_SQP_ACTIVE_SET_PCG_PLS_HPP__
