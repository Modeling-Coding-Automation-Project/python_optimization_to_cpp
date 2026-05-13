/**
 * @file python_optimization_sqp_active_set_pcg_pls.hpp
 *
 * @brief Sequential Quadratic Programming (SQP) Active Set solver with
 * Preconditioned Conjugate Gradient (PCG) and Projected Line Search (PLS).
 *
 * This header defines the SQP_ActiveSet_PCG_PLS class template and related
 * utilities for solving constrained optimization problems, typically arising in
 * optimal control and model predictive control (MPC) applications. The solver
 * uses an active set strategy to handle box constraints and employs a
 * preconditioned conjugate gradient method for solving the quadratic
 * subproblems, with a projected line search for robust step size selection.
 *
 * Key Features:
 * - Active set management for box constraints on control inputs.
 * - Preconditioned conjugate gradient (PCG) for efficient solution of large,
 * sparse quadratic problems.
 * - Projected line search (PLS) for step size selection and constraint
 * satisfaction.
 * - Flexible cost, gradient, and Hessian-vector product function interfaces via
 * std::function.
 * - Customizable solver parameters (tolerances, iteration limits,
 * regularization factors).
 */
#ifndef PYTHON_OPTIMIZATION_SQP_ACTIVE_SET_PCG_PLS_HPP_
#define PYTHON_OPTIMIZATION_SQP_ACTIVE_SET_PCG_PLS_HPP_

#include "python_optimization_common.hpp"
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
static constexpr double STEP_NORM_ZERO_LIMIT_DEFAULT = 1e-12;

template <typename X_Type, typename U_Horizon_Type, typename V_Horizon_Type,
          typename HVP_Type>
using HVP_Function_Object = std::function<HVP_Type(
    const X_Type &, const U_Horizon_Type &, const V_Horizon_Type &)>;

/* SQP Active Set with PCG and PLS */

/**
 * @brief Sequential Quadratic Programming (SQP) solver using Active Set,
 * Preconditioned Conjugate Gradient (PCG), and Projected Line Search (PLS).
 *
 * This class implements an SQP-based optimization algorithm for solving
 * constrained nonlinear problems, typically arising in optimal control and
 * trajectory planning. It leverages an active set strategy for handling
 * constraints, a preconditioned conjugate gradient method for solving the
 * quadratic subproblems, and a projected line search for step size selection.
 */
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
  using CostMatrices_Type_ = CostMatrices_Type_In;

  using T_ = Value_Type;

  using Mask_Type_ = U_Horizon_Type;
  using Gradient_Type_ = U_Horizon_Type;
  using V_Horizon_Type_ = U_Horizon_Type;
  using HVP_Type_ = U_Horizon_Type;
  using RHS_Type_ = U_Horizon_Type;
  using R_Full_Type_ = U_Horizon_Type;
  using M_Inv_Type_ = U_Horizon_Type;

  using ActiveSet_Type_ = ActiveSet2D_Type<INPUT_SIZE, NP>;

  using Cost_Function_Object_Type_ =
      CostFunction_Object<X_Type, U_Horizon_Type>;

  using Cost_And_Gradient_Function_Object_Type_ =
      CostAndGradientFunction_Object<X_Type, U_Horizon_Type, Gradient_Type_>;

  using HVP_Function_Object_Type_ =
      HVP_Function_Object<X_Type, U_Horizon_Type, V_Horizon_Type_, HVP_Type_>;

  using U_min_Type_ = typename CostMatrices_Type_::U_Min_Type;
  using U_max_Type_ = typename CostMatrices_Type_::U_Max_Type;
  using Y_min_Type_ = typename CostMatrices_Type_::Y_Min_Type;
  using Y_max_Type_ = typename CostMatrices_Type_::Y_Max_Type;

  using U_Min_Matrix_Type_ = PythonNumpy::Tile_Type<1, NP, U_min_Type_>;
  using U_Max_Matrix_Type_ = PythonNumpy::Tile_Type<1, NP, U_max_Type_>;
  using Y_Min_Matrix_Type_ = PythonNumpy::Tile_Type<1, (NP + 1), Y_min_Type_>;
  using Y_Max_Matrix_Type_ = PythonNumpy::Tile_Type<1, (NP + 1), Y_max_Type_>;

  using At_Lower_Upper_Type_ =
      PythonNumpy::DenseMatrix_Type<bool, INPUT_SIZE, NP>;

public:
  /* Constructor */
  SQP_ActiveSet_PCG_PLS()
      : X_initial(), hvp_function(nullptr),
        _gradient_norm_zero_limit(
            static_cast<T_>(GRADIENT_NORM_ZERO_LIMIT_DEFAULT)),
        _alpha_small_limit(static_cast<T_>(ALPHA_SMALL_LIMIT_DEFAULT)),
        _alpha_decay_rate(static_cast<T_>(ALPHA_DECAY_RATE_DEFAULT)),
        _pcg_php_minus_limit(static_cast<T_>(PCG_PHP_MINUS_LIMIT_DEFAULT)),
        _solver_max_iteration(SOLVER_MAX_ITERATION_DEFAULT),
        _pcg_max_iteration(PCG_MAX_ITERATION_DEFAULT),
        _line_search_max_iteration(LINE_SEARCH_MAX_ITERATION_DEFAULT),
        _pcg_tol(static_cast<T_>(PCG_TOL_DEFAULT)),
        _lambda_factor(static_cast<T_>(LAMBDA_FACTOR_DEFAULT)),
        _step_norm_zero_limit(static_cast<T_>(STEP_NORM_ZERO_LIMIT_DEFAULT)),
        _diag_R_full(R_Full_Type_::ones()), _mask(), _active_set(),
        _solver_step_iterated_number(0), _pcg_step_iterated_number(0),
        _line_search_step_iterated_number(0), J_optimal_(static_cast<T_>(0)) {}

  /* Copy constructor */
  SQP_ActiveSet_PCG_PLS(const SQP_ActiveSet_PCG_PLS &input)
      : X_initial(input.X_initial), hvp_function(input.hvp_function),
        _gradient_norm_zero_limit(input._gradient_norm_zero_limit),
        _alpha_small_limit(input._alpha_small_limit),
        _alpha_decay_rate(input._alpha_decay_rate),
        _pcg_php_minus_limit(input._pcg_php_minus_limit),
        _solver_max_iteration(input._solver_max_iteration),
        _pcg_max_iteration(input._pcg_max_iteration),
        _line_search_max_iteration(input._line_search_max_iteration),
        _pcg_tol(input._pcg_tol), _lambda_factor(input._lambda_factor),
        _step_norm_zero_limit(input._step_norm_zero_limit),
        _diag_R_full(input._diag_R_full), _mask(input._mask),
        _active_set(input._active_set),
        _solver_step_iterated_number(input._solver_step_iterated_number),
        _pcg_step_iterated_number(input._pcg_step_iterated_number),
        _line_search_step_iterated_number(
            input._line_search_step_iterated_number),
        J_optimal_(input.J_optimal_) {}

  /* Copy assignment */
  SQP_ActiveSet_PCG_PLS &operator=(const SQP_ActiveSet_PCG_PLS &input) {
    if (this != &input) {
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
      this->_step_norm_zero_limit = input._step_norm_zero_limit;

      this->_diag_R_full = input._diag_R_full;

      this->_mask = input._mask;
      this->_active_set = input._active_set;

      this->_solver_step_iterated_number = input._solver_step_iterated_number;
      this->_pcg_step_iterated_number = input._pcg_step_iterated_number;
      this->_line_search_step_iterated_number =
          input._line_search_step_iterated_number;

      this->J_optimal_ = input.J_optimal_;
    }
    return *this;
  }

  /* Move constructor */
  SQP_ActiveSet_PCG_PLS(SQP_ActiveSet_PCG_PLS &&input) noexcept
      : X_initial(std::move(input.X_initial)),
        hvp_function(std::move(input.hvp_function)),
        _gradient_norm_zero_limit(input._gradient_norm_zero_limit),
        _alpha_small_limit(input._alpha_small_limit),
        _alpha_decay_rate(input._alpha_decay_rate),
        _pcg_php_minus_limit(input._pcg_php_minus_limit),
        _solver_max_iteration(input._solver_max_iteration),
        _pcg_max_iteration(input._pcg_max_iteration),
        _line_search_max_iteration(input._line_search_max_iteration),
        _pcg_tol(input._pcg_tol), _lambda_factor(input._lambda_factor),
        _step_norm_zero_limit(input._step_norm_zero_limit),
        _diag_R_full(std::move(input._diag_R_full)),
        _mask(std::move(input._mask)),
        _active_set(std::move(input._active_set)),
        _solver_step_iterated_number(input._solver_step_iterated_number),
        _pcg_step_iterated_number(input._pcg_step_iterated_number),
        _line_search_step_iterated_number(
            input._line_search_step_iterated_number),
        J_optimal_(input.J_optimal_) {}

  /* Move assignment */
  SQP_ActiveSet_PCG_PLS &operator=(SQP_ActiveSet_PCG_PLS &&input) noexcept {
    if (this != &input) {
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
      this->_step_norm_zero_limit = input._step_norm_zero_limit;

      this->_diag_R_full = std::move(input._diag_R_full);

      this->_mask = std::move(input._mask);
      this->_active_set = std::move(input._active_set);

      this->_solver_step_iterated_number = input._solver_step_iterated_number;
      this->_pcg_step_iterated_number = input._pcg_step_iterated_number;
      this->_line_search_step_iterated_number =
          input._line_search_step_iterated_number;

      this->J_optimal_ = input.J_optimal_;
    }
    return *this;
  }

public:
  /* Setter */
  inline void set_gradient_norm_zero_limit(const T_ &limit) {
    this->_gradient_norm_zero_limit = limit;
  }

  inline void set_alpha_small_limit(const T_ &limit) {
    this->_alpha_small_limit = limit;
  }

  inline void set_alpha_decay_rate(const T_ &rate) {
    this->_alpha_decay_rate = rate;
  }

  inline void set_pcg_php_minus_limit(const T_ &limit) {
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

  inline void set_pcg_tol(const T_ &tol) { this->_pcg_tol = tol; }

  inline void set_lambda_factor(const T_ &factor) {
    this->_lambda_factor = factor;
  }

  inline void set_step_norm_zero_limit(const T_ &limit) {
    this->_step_norm_zero_limit = limit;
  }

  inline void set_diag_R_full(const R_Full_Type_ &diag_R_full) {
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

  /**
   * @brief Solves a linear system using the Preconditioned Conjugate Gradient
   * (PCG) method.
   *
   * This function implements the PCG algorithm to solve a system of equations
   * of the form Hx = rhs, where H is implicitly defined by the Hessian-vector
   * product function (`hvp_function`) and regularization. The method uses a
   * preconditioner (`M_inv`) to accelerate convergence, and operates only on
   * the active set.
   *
   * @tparam U_Horizon_Type Type of the input horizon variable.
   * @tparam RHS_Type_ Type of the right-hand side vector.
   * @tparam M_Inv_Type_ Type of the preconditioner matrix/vector.
   *
   * @param U_horizon_in Input horizon variable for the Hessian-vector product.
   * @param rhs Right-hand side vector of the linear system.
   * @param M_inv Preconditioner matrix/vector (inverse of the diagonal or
   * block-diagonal approximation of H).
   * @return Solution vector `d` that approximately solves Hx = rhs.
   */
  inline auto
  preconditioned_conjugate_gradient(const U_Horizon_Type U_horizon_in,
                                    const RHS_Type_ &rhs,
                                    const M_Inv_Type_ &M_inv) -> RHS_Type_ {
    RHS_Type_ d;

    auto rhs_norm = ActiveSet2D_MatrixOperator::norm(rhs, this->_active_set);
    if (rhs_norm < RHS_NORM_ZERO_LIMIT_DEFAULT) {
      /* Do Nothing. */
    } else {
      RHS_Type_ r = rhs;

      /* Preconditioning */
      auto z = ActiveSet2D_MatrixOperator::element_wise_product<
          T_, INPUT_SIZE, NP, INPUT_SIZE, NP>(r, M_inv, this->_active_set);

      RHS_Type_ p = z;

      auto rz = ActiveSet2D_MatrixOperator::vdot(r, z, this->_active_set);

      for (std::size_t pcg_iteration = 0;
           pcg_iteration < this->_pcg_max_iteration; ++pcg_iteration) {

        auto Hp = this->hvp_function(this->X_initial, U_horizon_in, p);
        Hp = Hp + this->_lambda_factor * p;

        T_ denominator =
            ActiveSet2D_MatrixOperator::vdot(p, Hp, this->_active_set);

        /* Simple handling of negative curvature and semi-definiteness */
        this->_pcg_step_iterated_number = pcg_iteration + 1;
        if (denominator <= this->_pcg_php_minus_limit) {
          break;
        }

        T_ alpha = rz / denominator;

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

        T_ rz_new = ActiveSet2D_MatrixOperator::vdot(r, z, this->_active_set);

        T_ beta = rz_new / rz;

        p = z + ActiveSet2D_MatrixOperator::matrix_multiply_scalar(
                    p, beta, this->_active_set);

        rz = rz_new;
      }
    }

    return d;
  }

  /**
   * @brief Computes the mask of free variables in the optimization horizon.
   *
   * This function determines which variables in the input horizon are "free"
   * (not at their bounds) based on the provided minimum and maximum matrices,
   * absolute tolerance, and gradient tolerance. It updates the internal active
   * set with indices of variables at their bounds.
   *
   * @tparam U_Horizon_Type Type of the optimization horizon variable.
   * @tparam Gradient_Type_ Type of the gradient vector.
   * @tparam U_Min_Matrix_Type_ Type of the minimum bound matrix.
   * @tparam U_Max_Matrix_Type_ Type of the maximum bound matrix.
   * @tparam T_ Scalar type for tolerance values.
   * @tparam Mask_Type_ Type of the mask returned.
   *
   * @param U_horizon_in Input optimization horizon variable.
   * @param gradient Gradient vector.
   * @param U_min_matrix Minimum bound matrix.
   * @param U_max_matrix Maximum bound matrix.
   * @param atol Absolute tolerance for bound checking.
   * @param gtol Gradient tolerance for active set determination.
   * @return Mask_Type_ Mask indicating free variables (typically 1 for free, 0
   * for active).
   */
  inline auto free_mask(U_Horizon_Type &U_horizon_in, Gradient_Type_ &gradient,
                        const U_Min_Matrix_Type_ &U_min_matrix,
                        const U_Max_Matrix_Type_ &U_max_matrix, const T_ &atol,
                        const T_ &gtol) -> Mask_Type_ {
    auto m = Mask_Type_::ones();

    this->_active_set.clear();

    At_Lower_Upper_Type_ at_lower;
    At_Lower_Upper_Type_ at_upper;

    std::tie(at_lower, at_upper) =
        MatrixOperation::free_mask_at_check<At_Lower_Upper_Type_,
                                            At_Lower_Upper_Type_>(
            U_horizon_in, U_min_matrix, U_max_matrix, atol);

    MatrixOperation::free_mask_push_active(m, gradient, at_lower, at_upper,
                                           gtol, this->_active_set);

    return m;
  }

  /**
   * @brief Solves an optimization problem using SQP with active set, PCG, and
   * PLS methods.
   *
   * This function iteratively optimizes the control horizon vector
   * `U_horizon_store` given initial values, cost and gradient functions,
   * Hessian-vector product function, and constraints. It performs
   * gradient-based updates, applies a preconditioned conjugate gradient method,
   * and uses line search with projection to ensure constraints are satisfied.
   * The process continues until convergence or maximum iterations are reached.
   *
   * @tparam U_Horizon_Type Type representing the control horizon vector.
   * @tparam Cost_And_Gradient_Function_Object_Type_ Type of the cost and
   * gradient function object.
   * @tparam Cost_Function_Object_Type_ Type of the cost function object.
   * @tparam HVP_Function_Object_Type_ Type of the Hessian-vector product
   * function object.
   * @tparam X_Type Type representing the initial state vector.
   * @tparam U_Min_Matrix_Type_ Type representing the minimum constraint matrix
   * for control horizon.
   * @tparam U_Max_Matrix_Type_ Type representing the maximum constraint matrix
   * for control horizon.
   *
   * @param U_horizon_initial Initial control horizon vector.
   * @param cost_and_gradient_function Function object to compute cost and
   * gradient.
   * @param cost_function Function object to compute cost.
   * @param hvp_function_in Hessian-vector product function object.
   * @param X_initial_in Initial state vector.
   * @param U_min_matrix Minimum constraint matrix for control horizon.
   * @param U_max_matrix Maximum constraint matrix for control horizon.
   * @return Optimized control horizon vector.
   */
  inline auto solve(
      const U_Horizon_Type &U_horizon_initial,
      const Cost_And_Gradient_Function_Object_Type_ &cost_and_gradient_function,
      const Cost_Function_Object_Type_ &cost_function,
      const HVP_Function_Object_Type_ &hvp_function_in,
      const X_Type &X_initial_in, const U_Min_Matrix_Type_ &U_min_matrix,
      const U_Max_Matrix_Type_ &U_max_matrix) -> U_Horizon_Type {
    this->X_initial = X_initial_in;
    U_Horizon_Type U_horizon_store = U_horizon_initial;

    T_ J = static_cast<T_>(0);

    for (std::size_t solver_iteration = 0;
         solver_iteration < this->_solver_max_iteration; ++solver_iteration) {
      /* Calculate cost and gradient */
      Gradient_Type_ gradient;
      std::tie(J, gradient) =
          cost_and_gradient_function(X_initial, U_horizon_store);

      this->_solver_step_iterated_number = solver_iteration + 1;
      if (PythonNumpy::norm(gradient) < this->_gradient_norm_zero_limit) {
        break;
      }

      this->_mask =
          this->free_mask(U_horizon_store, gradient, U_min_matrix, U_max_matrix,
                          static_cast<T_>(U_NEAR_LIMIT_DEFAULT),
                          static_cast<T_>(GRADIENT_ZERO_LIMIT_DEFAULT));

      RHS_Type_ rhs = -gradient;

      auto diag_R_full_lambda_factor = MatrixOperation::add_scalar_to_matrix(
          this->_diag_R_full, this->_lambda_factor);

      M_Inv_Type_ M_inv;

      MatrixOperation::solver_calculate_M_inv(
          M_inv, diag_R_full_lambda_factor,
          static_cast<T_>(AVOID_ZERO_DIVIDE_LIMIT));

      this->hvp_function = hvp_function_in;

      auto d =
          this->preconditioned_conjugate_gradient(U_horizon_store, rhs, M_inv);

      auto norm_d = ActiveSet2D_MatrixOperator::norm(d, this->_active_set);
      if (norm_d < this->_step_norm_zero_limit) {
        break;
      }

      /*
       * line search and projection
       * (No distinction between fixed/free is needed here,
       *  project the whole)
       */
      T_ alpha = static_cast<T_>(1);
      U_Horizon_Type U_horizon_new = U_horizon_store;

      bool U_updated_flag = false;
      for (std::size_t line_search_iteration = 0;
           line_search_iteration < this->_line_search_max_iteration;
           ++line_search_iteration) {
        auto U_candidate = U_horizon_new + alpha * d;

        MatrixOperation::saturate_U_horizon(U_candidate, U_min_matrix,
                                            U_max_matrix);

        auto J_candidate = cost_function(X_initial, U_candidate);

        this->_line_search_step_iterated_number = line_search_iteration + 1;
        if (J_candidate < J || alpha < this->_alpha_small_limit) {
          U_horizon_new = U_candidate;
          J = J_candidate;
          U_updated_flag = true;
          break;
        }

        alpha *= this->_alpha_decay_rate;
      }

      if (true == U_updated_flag) {
        U_horizon_store = U_horizon_new;
      } else {
        break;
      }
    }

    this->J_optimal_ = J;

    return U_horizon_store;
  }

public:
  /* Variable */
  X_Type X_initial;
  HVP_Function_Object_Type_ hvp_function;

protected:
  /* Variable */
  T_ _gradient_norm_zero_limit;
  T_ _alpha_small_limit;
  T_ _alpha_decay_rate;
  T_ _pcg_php_minus_limit;

  std::size_t _solver_max_iteration;
  std::size_t _pcg_max_iteration;
  std::size_t _line_search_max_iteration;

  T_ _pcg_tol;
  T_ _lambda_factor;
  T_ _step_norm_zero_limit;

  R_Full_Type_ _diag_R_full;

  Mask_Type_ _mask;
  ActiveSet_Type_ _active_set;

  std::size_t _solver_step_iterated_number;
  std::size_t _pcg_step_iterated_number;
  std::size_t _line_search_step_iterated_number;

  T_ J_optimal_;
};

/* make SQP_ActiveSetPCG_PLS */
/**
 * @brief Factory function to create an instance of SQP_ActiveSet_PCG_PLS.
 *
 * This function template constructs and returns a default-initialized
 * SQP_ActiveSet_PCG_PLS object parameterized by the specified
 * CostMatrices_Type.
 *
 * @tparam CostMatrices_Type The type representing cost matrices used in the SQP
 * algorithm.
 * @return SQP_ActiveSet_PCG_PLS<CostMatrices_Type> A new instance of
 * SQP_ActiveSet_PCG_PLS.
 */
template <typename CostMatrices_Type>
inline auto make_SQP_ActiveSet_PCG_PLS(void)
    -> SQP_ActiveSet_PCG_PLS<CostMatrices_Type> {
  return SQP_ActiveSet_PCG_PLS<CostMatrices_Type>();
}

/* SQP_ActiveSetPCG_PLS Type */
/**
 * @brief Alias template for SQP_ActiveSet_PCG_PLS with specified cost matrices
 * type.
 *
 * This alias simplifies the usage of the SQP_ActiveSet_PCG_PLS class template
 * by allowing users to refer to it as SQP_ActiveSet_PCG_PLS_Type with a given
 * CostMatrices_Type.
 *
 * @tparam CostMatrices_Type The type representing the cost matrices used in the
 * SQP algorithm.
 */
template <typename CostMatrices_Type>
using SQP_ActiveSet_PCG_PLS_Type = SQP_ActiveSet_PCG_PLS<CostMatrices_Type>;

} // namespace PythonOptimization

#endif // PYTHON_OPTIMIZATION_SQP_ACTIVE_SET_PCG_PLS_HPP_
