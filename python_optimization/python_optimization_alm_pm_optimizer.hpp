/**
 * @file python_optimization_alm_pm_optimizer.hpp
 *
 * @brief ALM/PM (Augmented Lagrangian Method / Penalty Method) optimizer for
 * constrained nonlinear optimization.
 *
 * This header implements the ALM/PM algorithm for solving constrained nonlinear
 * optimization problems, using PANOC as the inner solver. The implementation is
 * a C++ port of alm_pm_optimizer.py and is based on the Rust
 * optimization-engine library.
 *
 * ALM/PM solves problems of the form:
 *
 *     min  f(u)
 *      u
 *     s.t. u element of U           (box constraints on decision variables,
 * handled by PANOC) F1(u) element of C       (ALM-type constraints, e.g.,
 * output constraints) F2(u) = 0                (PM-type equality constraints,
 * optional)
 *
 * For nonlinear MPC applications, the typical constraints are:
 *     - u_min <= u <= u_max  (input box constraints -> set U for PANOC)
 *     - y_min <= Y(u) <= y_max  (output box constraints -> F1(u) = Y(u), C =
 * [y_min, y_max])
 *
 * Algorithm overview (outer loop):
 * 1. y <- Pi_Y(y)                           (project Lagrange multipliers onto
 * set Y)
 * 2. u <- argmin_{u in U} psi(u; xi)        (solve inner problem via PANOC, xi
 * = (c, y))
 * 3. y^+ <- y + c[F1(u) - Pi_C(F1(u) + y/c)] (update Lagrange multipliers)
 * 4. z^+ <- ||y^+ - y||, t^+ <- ||F2(u)||  (compute infeasibility measures)
 * 5. If converged -> return (u, y^+)
 * 6. Else if no sufficient decrease -> c <- rho*c  (increase penalty)
 * 7. epsilon <- max(epsilon, beta*epsilon)   (shrink inner tolerance)
 *
 * The augmented cost function is:
 *     psi(u; xi) = f(u) + (c/2)[dist^2_C(F1(u) + y/c_bar) + ||F2(u)||^2]
 * where c_bar = max(1, c), and its gradient is:
 *     grad psi(u; xi) = grad f(u) + c*JF1(u)^T[t(u) - Pi_C(t(u))] +
 * c*JF2(u)^T*F2(u) where t(u) = F1(u) + y/c.
 *
 * Module structure:
 *     - ALM_Factory:          Builds psi(u; xi) and grad psi(u; xi)
 *     - ALM_Problem:          Bundles all problem data for the ALM optimizer
 *     - ALM_Cache:            Pre-allocated working memory
 *     - ALM_PM_Optimizer:     Main ALM/PM solver (outer loop with PANOC inner
 * solver)
 *     - ALM_SolverStatus:     Result returned by ALM_PM_Optimizer::solve()
 *     - BoxProjectionOperator, BallProjectionOperator: Projection utilities
 *
 * References:
 *     - optimization-engine: https://github.com/alphaville/optimization-engine
 */
#ifndef __PYTHON_OPTIMIZATION_ALM_PM_OPTIMIZER_HPP__
#define __PYTHON_OPTIMIZATION_ALM_PM_OPTIMIZER_HPP__

#include "python_optimization_common.hpp"
#include "python_optimization_panoc.hpp"
#include "python_optimization_utility.hpp"

#include "python_control.hpp"
#include "python_numpy.hpp"

#include <cmath>
#include <functional>
#include <type_traits>

namespace PythonOptimization {

// ============================================================================
// ALM/PM Constants
// ============================================================================
namespace ALM_Constants {

/** @brief Maximum number of outer ALM/PM iterations. */
static constexpr std::size_t DEFAULT_MAX_OUTER_ITERATIONS = 50;

/** @brief Maximum number of inner PANOC iterations per outer iteration. */
static constexpr std::size_t DEFAULT_MAX_INNER_ITERATIONS = 5000;

/** @brief Target tolerance for the inner solver (epsilon). */
static constexpr double DEFAULT_EPSILON_TOLERANCE = 1e-6;

/** @brief Tolerance for ALM/PM infeasibility (delta). */
static constexpr double DEFAULT_DELTA_TOLERANCE = 1e-4;

/** @brief Factor by which the penalty parameter c is multiplied (rho). */
static constexpr double DEFAULT_PENALTY_UPDATE_FACTOR = 5.0;

/** @brief Factor by which the inner tolerance is shrunk each iteration (beta).
 */
static constexpr double DEFAULT_EPSILON_UPDATE_FACTOR = 0.1;

/** @brief Sufficient decrease coefficient (theta) for penalty stall check. */
static constexpr double DEFAULT_INFEASIBLE_SUFFICIENT_DECREASE_FACTOR = 0.1;

/** @brief Initial inner tolerance (epsilon_0). */
static constexpr double DEFAULT_INITIAL_TOLERANCE = 0.1;

/** @brief Initial penalty parameter (c_0). */
static constexpr double DEFAULT_INITIAL_PENALTY = 10.0;

/** @brief Machine epsilon for numerical comparisons. */
static constexpr double SMALL_EPSILON = 1e-30;

} // namespace ALM_Constants

// ============================================================================
// ALM Exit Status (reuses PANOC_ExitStatus)
// ============================================================================

/**
 * @brief Exit status of the ALM/PM solver.
 *
 * Reuses PANOC_ExitStatus to indicate convergence, iteration limit,
 * or numerical failure.
 */
using ALM_ExitStatus = PANOC_ExitStatus;

// ============================================================================
// ALM_SolverStatus
// ============================================================================

/**
 * @brief Result returned by ALM_PM_Optimizer::solve.
 *
 * @tparam T Scalar value type.
 * @tparam N1 Dimension of ALM constraint output (F1).
 */
template <typename T, std::size_t N1> struct ALM_SolverStatus {
  using Lagrange_Type = PythonNumpy::DenseMatrix_Type<T, N1, 1>;

  ALM_ExitStatus exit_status;
  std::size_t num_outer_iterations;
  std::size_t num_inner_iterations;
  T last_problem_norm_fpr;
  Lagrange_Type lagrange_multipliers;
  T penalty;
  T delta_y_norm;
  T f2_norm;
  T cost;

  /**
   * @brief Check whether the solver converged.
   * @return true if converged.
   */
  inline bool has_converged(void) const {
    return exit_status == ALM_ExitStatus::CONVERGED;
  }
};

// ============================================================================
// BoxProjectionOperator
// ============================================================================

/**
 * @brief In-place box projection operator.
 *
 * Projects x in-place onto the box [lower, upper] element-wise.
 *
 * @tparam T Scalar value type.
 * @tparam Size Dimension of the vector to project.
 */
template <typename T, std::size_t Size> class BoxProjectionOperator {
public:
  using Vector_Type = PythonNumpy::DenseMatrix_Type<T, Size, 1>;

  BoxProjectionOperator()
      : _lower(), _upper(), _has_lower(false), _has_upper(false) {}

  BoxProjectionOperator(const Vector_Type &lower, const Vector_Type &upper)
      : _lower(lower), _upper(upper), _has_lower(true), _has_upper(true) {}

  /* Copy / Move: use defaults */
  BoxProjectionOperator(const BoxProjectionOperator &) = default;
  BoxProjectionOperator &operator=(const BoxProjectionOperator &) = default;
  BoxProjectionOperator(BoxProjectionOperator &&) noexcept = default;
  BoxProjectionOperator &operator=(BoxProjectionOperator &&) noexcept = default;

  inline void set_lower(const Vector_Type &lower) {
    this->_lower = lower;
    this->_has_lower = true;
  }

  inline void set_upper(const Vector_Type &upper) {
    this->_upper = upper;
    this->_has_upper = true;
  }

  /**
   * @brief Project x in-place onto the box [lower, upper].
   * @param x Vector to project (modified in-place).
   */
  inline void project(Vector_Type &x) const {
    for (std::size_t i = 0; i < Size; ++i) {
      if (this->_has_lower) {
        if (x(i, 0) < this->_lower(i, 0)) {
          x(i, 0) = this->_lower(i, 0);
        }
      }
      if (this->_has_upper) {
        if (x(i, 0) > this->_upper(i, 0)) {
          x(i, 0) = this->_upper(i, 0);
        }
      }
    }
  }

private:
  Vector_Type _lower;
  Vector_Type _upper;
  bool _has_lower;
  bool _has_upper;
};

// ============================================================================
// BallProjectionOperator
// ============================================================================

/**
 * @brief In-place Euclidean ball projection operator.
 *
 * Projects x in-place onto the ball {x : ||x - center|| <= radius}.
 *
 * @tparam T Scalar value type.
 * @tparam Size Dimension of the vector to project.
 */
template <typename T, std::size_t Size> class BallProjectionOperator {
public:
  using Vector_Type = PythonNumpy::DenseMatrix_Type<T, Size, 1>;

  BallProjectionOperator()
      : _center(), _radius(static_cast<T>(1)), _has_center(false) {}

  BallProjectionOperator(const Vector_Type &center, const T &radius)
      : _center(center), _radius(radius), _has_center(true) {}

  explicit BallProjectionOperator(const T &radius)
      : _center(), _radius(radius), _has_center(false) {}

  /* Copy / Move: use defaults */
  BallProjectionOperator(const BallProjectionOperator &) = default;
  BallProjectionOperator &operator=(const BallProjectionOperator &) = default;
  BallProjectionOperator(BallProjectionOperator &&) noexcept = default;
  BallProjectionOperator &
  operator=(BallProjectionOperator &&) noexcept = default;

  inline void set_center(const Vector_Type &center) {
    this->_center = center;
    this->_has_center = true;
  }

  inline void set_radius(const T &radius) { this->_radius = radius; }

  /**
   * @brief Project x in-place onto the ball.
   * @param x Vector to project (modified in-place).
   */
  inline void project(Vector_Type &x) const {
    Vector_Type d;
    if (this->_has_center) {
      d = x - this->_center;
    } else {
      d = x;
    }

    T norm_d = static_cast<T>(0);
    for (std::size_t i = 0; i < Size; ++i) {
      norm_d += d(i, 0) * d(i, 0);
    }
    norm_d = static_cast<T>(std::sqrt(static_cast<double>(norm_d)));

    if (norm_d > this->_radius) {
      T scale = this->_radius / norm_d;
      if (this->_has_center) {
        x = this->_center + scale * d;
      } else {
        x = scale * d;
      }
    }
  }

private:
  Vector_Type _center;
  T _radius;
  bool _has_center;
};

// ============================================================================
// ALM_Cache
// ============================================================================

/**
 * @brief Pre-allocated working memory for the ALM/PM algorithm.
 *
 * Create once and reuse across multiple solve calls to avoid repeated
 * memory allocation.
 *
 * @tparam T Scalar value type.
 * @tparam ProblemSize Dimension of the decision variable (INPUT_SIZE * NP).
 * @tparam N1 Dimension of F1 output (number of ALM-type constraints).
 * @tparam N2 Dimension of F2 output (number of PM-type equality constraints).
 * @tparam LBFGSMemory L-BFGS memory size for PANOC.
 */
template <typename T, std::size_t ProblemSize, std::size_t N1,
          std::size_t N2 = 0, std::size_t LBFGSMemory = 5>
class ALM_Cache {
public:
  using PANOC_Cache_Type = PANOC_Cache<T, ProblemSize, LBFGSMemory>;
  using Y_Plus_Type = PythonNumpy::DenseMatrix_Type<T, N1, 1>;
  using Xi_Type = PythonNumpy::DenseMatrix_Type<T, (1 + N1), 1>;
  using W_ALM_Aux_Type = PythonNumpy::DenseMatrix_Type<T, N1, 1>;
  using W_PM_Type = PythonNumpy::DenseMatrix_Type<T, (N2 > 0 ? N2 : 1), 1>;

  /* Constructor */
  ALM_Cache()
      : panoc_cache(), y_plus(), xi(), w_alm_aux(), w_pm(),
        delta_y_norm(static_cast<T>(0)),
        delta_y_norm_plus(static_cast<T>(1e30)), f2_norm(static_cast<T>(0)),
        f2_norm_plus(static_cast<T>(1e30)), iteration(0),
        inner_iteration_count(0),
        last_inner_problem_norm_fpr(static_cast<T>(-1)) {
    /* Initialize xi(0, 0) = c (penalty parameter) to default */
    this->xi(0, 0) = static_cast<T>(ALM_Constants::DEFAULT_INITIAL_PENALTY);
  }

  explicit ALM_Cache(const T &panoc_tolerance)
      : panoc_cache(panoc_tolerance), y_plus(), xi(), w_alm_aux(), w_pm(),
        delta_y_norm(static_cast<T>(0)),
        delta_y_norm_plus(static_cast<T>(1e30)), f2_norm(static_cast<T>(0)),
        f2_norm_plus(static_cast<T>(1e30)), iteration(0),
        inner_iteration_count(0),
        last_inner_problem_norm_fpr(static_cast<T>(-1)) {
    this->xi(0, 0) = static_cast<T>(ALM_Constants::DEFAULT_INITIAL_PENALTY);
  }

  /* Copy / Move: use defaults */
  ALM_Cache(const ALM_Cache &) = default;
  ALM_Cache &operator=(const ALM_Cache &) = default;
  ALM_Cache(ALM_Cache &&) noexcept = default;
  ALM_Cache &operator=(ALM_Cache &&) noexcept = default;

  /**
   * @brief Reset the cache to its initial state (called at the start of each
   * solve).
   */
  inline void reset(void) {
    this->panoc_cache.reset();
    this->iteration = 0;
    this->f2_norm = static_cast<T>(0);
    this->f2_norm_plus = static_cast<T>(0);
    this->delta_y_norm = static_cast<T>(0);
    this->delta_y_norm_plus = static_cast<T>(0);
    this->inner_iteration_count = 0;
  }

  /* Public members */
  PANOC_Cache_Type panoc_cache;

  /** @brief Next Lagrange multiplier vector y^+. */
  Y_Plus_Type y_plus;

  /**
   * @brief Parameter vector xi = (c, y) of size (1 + N1) x 1.
   * xi(0,0) = c (penalty parameter), xi(0,1..N1) = y (Lagrange multipliers).
   */
  Xi_Type xi;

  /** @brief Auxiliary working vector for ALM (size N1). */
  W_ALM_Aux_Type w_alm_aux;

  /** @brief Working vector for PM constraints (size N2). */
  W_PM_Type w_pm;

  /** @brief ||y^+ - y|| at current iteration. */
  T delta_y_norm;
  /** @brief ||y^+ - y|| at next iteration. */
  T delta_y_norm_plus;
  /** @brief ||F2(u)|| at current iteration. */
  T f2_norm;
  /** @brief ||F2(u)|| at next iteration. */
  T f2_norm_plus;

  /** @brief Outer iteration counter. */
  std::size_t iteration;
  /** @brief Total inner iteration count. */
  std::size_t inner_iteration_count;
  /** @brief Norm of inner problem FPR at last solve. */
  T last_inner_problem_norm_fpr;
};

// ============================================================================
// ALM_Factory
// ============================================================================

/**
 * @brief Constructs the augmented cost psi(u; xi) and its gradient
 * grad psi(u; xi) from the raw problem data.
 *
 * Given f, grad f, F1, JF1^T*d, C (and optionally F2, JF2^T*d), it builds:
 *
 *     psi(u; xi) = f(u) + (c/2)[dist^2_C(F1(u) + y/c_bar) + ||F2(u)||^2]
 *
 *     grad psi(u; xi) = grad f(u) + c*JF1(u)^T[t(u) - Pi_C(t(u))]
 *                       + c*JF2(u)^T*F2(u)
 *
 * where c_bar = max(1, c), t(u) = F1(u) + y/c_bar for psi and t(u) = F1(u) +
 * y/c for grad psi, and xi = (c, y).
 *
 * @tparam CostMatrices_Type_In Type providing cost matrices and dimensions.
 * @tparam N1 Dimension of F1 output (ALM constraints).
 * @tparam N2 Dimension of F2 output (PM constraints). Default 0.
 */
template <typename CostMatrices_Type_In, std::size_t N1, std::size_t N2 = 0>
class ALM_Factory {
public:
  /* Constants */
  static constexpr std::size_t STATE_SIZE = CostMatrices_Type_In::STATE_SIZE;
  static constexpr std::size_t INPUT_SIZE = CostMatrices_Type_In::INPUT_SIZE;
  static constexpr std::size_t OUTPUT_SIZE = CostMatrices_Type_In::OUTPUT_SIZE;
  enum : std::size_t { NP = CostMatrices_Type_In::NP };
  static constexpr std::size_t PROBLEM_SIZE = INPUT_SIZE * NP;

public:
  /* Types */
  using Value_Type = typename CostMatrices_Type_In::Value_Type;
  using _T = Value_Type;

  using U_Horizon_Type = typename CostMatrices_Type_In::U_Horizon_Type;
  using _Gradient_Type = U_Horizon_Type;

  using F1_Output_Type = PythonNumpy::DenseMatrix_Type<_T, N1, 1>;
  using F2_Output_Type =
      PythonNumpy::DenseMatrix_Type<_T, (N2 > 0 ? N2 : 1), 1>;
  using Xi_Type = PythonNumpy::DenseMatrix_Type<_T, (1 + N1), 1>;

  /** @brief F1(u) -> F1_Output_Type */
  using Mapping_F1_Type = std::function<F1_Output_Type(const U_Horizon_Type &)>;
  /** @brief JF1(u)^T * d -> Gradient_Type */
  using Jacobian_F1_Trans_Type = std::function<_Gradient_Type(
      const U_Horizon_Type &, const F1_Output_Type &)>;
  /** @brief In-place projection onto C: project(x) */
  using Set_C_Project_Type = std::function<void(F1_Output_Type &)>;

  /** @brief F2(u) -> F2_Output_Type */
  using Mapping_F2_Type = std::function<F2_Output_Type(const U_Horizon_Type &)>;
  /** @brief JF2(u)^T * d -> Gradient_Type */
  using Jacobian_F2_Trans_Type = std::function<_Gradient_Type(
      const U_Horizon_Type &, const F2_Output_Type &)>;

  /** @brief Cost function f(u) -> T */
  using Cost_Func_Type = std::function<_T(const U_Horizon_Type &)>;
  /** @brief Gradient function grad f(u) -> Gradient_Type */
  using Gradient_Func_Type =
      std::function<_Gradient_Type(const U_Horizon_Type &)>;

  /* Constructor */
  ALM_Factory()
      : _f(nullptr), _df(nullptr), _mapping_f1(nullptr),
        _jacobian_f1_trans(nullptr), _set_c_project(nullptr),
        _mapping_f2(nullptr), _jacobian_f2_trans(nullptr) {}

  /* Copy / Move: use defaults */
  ALM_Factory(const ALM_Factory &) = default;
  ALM_Factory &operator=(const ALM_Factory &) = default;
  ALM_Factory(ALM_Factory &&) noexcept = default;
  ALM_Factory &operator=(ALM_Factory &&) noexcept = default;

  /* Setters */
  inline void set_cost_function(const Cost_Func_Type &f) { this->_f = f; }
  inline void set_gradient_function(const Gradient_Func_Type &df) {
    this->_df = df;
  }
  inline void set_mapping_f1(const Mapping_F1_Type &mapping_f1) {
    this->_mapping_f1 = mapping_f1;
  }
  inline void
  set_jacobian_f1_trans(const Jacobian_F1_Trans_Type &jacobian_f1_trans) {
    this->_jacobian_f1_trans = jacobian_f1_trans;
  }
  inline void set_c_projection(const Set_C_Project_Type &set_c_project) {
    this->_set_c_project = set_c_project;
  }
  inline void set_mapping_f2(const Mapping_F2_Type &mapping_f2) {
    this->_mapping_f2 = mapping_f2;
  }
  inline void
  set_jacobian_f2_trans(const Jacobian_F2_Trans_Type &jacobian_f2_trans) {
    this->_jacobian_f2_trans = jacobian_f2_trans;
  }

  /**
   * @brief Compute the augmented cost psi(u; xi).
   *
   * @param u Decision variable (U_Horizon_Type).
   * @param xi Parameter vector xi = (c, y).
   * @return Augmented cost value.
   */
  inline auto psi(const U_Horizon_Type &u, const Xi_Type &xi) const -> _T {
    _T cost = this->_f(u);

    /* ALM term: (c/2) * dist^2_C(F1(u) + y/c_bar) */
    if (N1 > 0 && this->_mapping_f1 && this->_set_c_project) {
      _T c = xi(0, 0);
      _T c_bar = (c > static_cast<_T>(1)) ? c : static_cast<_T>(1);

      /* Extract y from xi(0, 1..N1) */
      F1_Output_Type y;
      for (std::size_t i = 0; i < N1; ++i) {
        y(i, 0) = xi(i + 1, 0);
      }

      /* t = F1(u) + y / c_bar */
      F1_Output_Type f1_u = this->_mapping_f1(u);
      F1_Output_Type t = f1_u + (static_cast<_T>(1) / c_bar) * y;

      /* s = Pi_C(t) */
      F1_Output_Type s = t;
      this->_set_c_project(s);

      /* dist^2_C(t) = ||t - s||^2 */
      F1_Output_Type diff = t - s;
      _T dist_sq = static_cast<_T>(0);
      for (std::size_t i = 0; i < N1; ++i) {
        dist_sq += diff(i, 0) * diff(i, 0);
      }
      cost += static_cast<_T>(0.5) * c * dist_sq;
    }

    /* PM term: (c/2) * ||F2(u)||^2 */
    if (N2 > 0 && this->_mapping_f2) {
      _T c = xi(0, 0);
      F2_Output_Type f2_u = this->_mapping_f2(u);
      _T f2_sq = static_cast<_T>(0);
      for (std::size_t i = 0; i < N2; ++i) {
        f2_sq += f2_u(i, 0) * f2_u(i, 0);
      }
      cost += static_cast<_T>(0.5) * c * f2_sq;
    }

    return cost;
  }

  /**
   * @brief Compute the gradient grad psi(u; xi).
   *
   * @param u Decision variable (U_Horizon_Type).
   * @param xi Parameter vector xi = (c, y).
   * @return Gradient of augmented cost.
   */
  inline auto d_psi(const U_Horizon_Type &u, const Xi_Type &xi) const
      -> _Gradient_Type {
    _Gradient_Type grad = this->_df(u);

    /* ALM gradient: c * JF1(u)^T [t(u) - Pi_C(t(u))] */
    if (N1 > 0 && this->_mapping_f1 && this->_jacobian_f1_trans &&
        this->_set_c_project) {
      _T c = xi(0, 0);

      /* Extract y from xi */
      F1_Output_Type y;
      for (std::size_t i = 0; i < N1; ++i) {
        y(i, 0) = xi(i + 1, 0);
      }

      /* t = F1(u) + y/c  (note: uses c, not c_bar) */
      F1_Output_Type f1_u = this->_mapping_f1(u);
      F1_Output_Type t = f1_u + (static_cast<_T>(1) / c) * y;

      /* s = Pi_C(t) */
      F1_Output_Type s = t;
      this->_set_c_project(s);

      /* d = t - Pi_C(t) */
      F1_Output_Type d = t - s;

      /* grad += c * JF1(u)^T * d */
      _Gradient_Type jf1t_d = this->_jacobian_f1_trans(u, d);
      grad = grad + c * jf1t_d;
    }

    /* PM gradient: c * JF2(u)^T * F2(u) */
    if (N2 > 0 && this->_mapping_f2 && this->_jacobian_f2_trans) {
      _T c = xi(0, 0);
      F2_Output_Type f2_u = this->_mapping_f2(u);
      _Gradient_Type jf2t_f2u = this->_jacobian_f2_trans(u, f2_u);
      grad = grad + c * jf2t_f2u;
    }

    return grad;
  }

private:
  Cost_Func_Type _f;
  Gradient_Func_Type _df;
  Mapping_F1_Type _mapping_f1;
  Jacobian_F1_Trans_Type _jacobian_f1_trans;
  Set_C_Project_Type _set_c_project;
  Mapping_F2_Type _mapping_f2;
  Jacobian_F2_Trans_Type _jacobian_f2_trans;
};

// ============================================================================
// ALM_Problem
// ============================================================================

/**
 * @brief Problem definition for ALM/PM optimization.
 *
 * Bundles all data required by ALM_PM_Optimizer: the parametric
 * augmented cost and its gradient (typically built by ALM_Factory),
 * box constraints on the decision variable, constraint mappings, and
 * projection operators.
 *
 * @tparam CostMatrices_Type_In Type providing cost matrices and dimensions.
 * @tparam N1 Dimension of F1 output (ALM constraints).
 * @tparam N2 Dimension of F2 output (PM constraints). Default 0.
 */
template <typename CostMatrices_Type_In, std::size_t N1, std::size_t N2 = 0>
class ALM_Problem {
public:
  /* Constants */
  static constexpr std::size_t STATE_SIZE = CostMatrices_Type_In::STATE_SIZE;
  static constexpr std::size_t INPUT_SIZE = CostMatrices_Type_In::INPUT_SIZE;
  static constexpr std::size_t OUTPUT_SIZE = CostMatrices_Type_In::OUTPUT_SIZE;
  enum : std::size_t { NP = CostMatrices_Type_In::NP };
  static constexpr std::size_t PROBLEM_SIZE = INPUT_SIZE * NP;

public:
  /* Types */
  using Value_Type = typename CostMatrices_Type_In::Value_Type;
  using _T = Value_Type;

  using U_Horizon_Type = typename CostMatrices_Type_In::U_Horizon_Type;
  using _Gradient_Type = U_Horizon_Type;

  using _U_min_Type = typename CostMatrices_Type_In::U_Min_Type;
  using _U_max_Type = typename CostMatrices_Type_In::U_Max_Type;

  using _U_Min_Matrix_Type = PythonNumpy::Tile_Type<1, NP, _U_min_Type>;
  using _U_Max_Matrix_Type = PythonNumpy::Tile_Type<1, NP, _U_max_Type>;

  using F1_Output_Type = PythonNumpy::DenseMatrix_Type<_T, N1, 1>;
  using F2_Output_Type =
      PythonNumpy::DenseMatrix_Type<_T, (N2 > 0 ? N2 : 1), 1>;
  using Xi_Type = PythonNumpy::DenseMatrix_Type<_T, (1 + N1), 1>;

  /** @brief Parametric cost: psi(u, xi) -> T */
  using Parametric_Cost_Type =
      std::function<_T(const U_Horizon_Type &, const Xi_Type &)>;
  /** @brief Parametric gradient: grad psi(u, xi) -> Gradient_Type */
  using Parametric_Gradient_Type =
      std::function<_Gradient_Type(const U_Horizon_Type &, const Xi_Type &)>;

  /** @brief F1(u) -> F1_Output_Type */
  using Mapping_F1_Type = std::function<F1_Output_Type(const U_Horizon_Type &)>;
  /** @brief In-place projection onto C */
  using Set_C_Project_Type = std::function<void(F1_Output_Type &)>;
  /** @brief In-place projection onto Y (for Lagrange multipliers) */
  using Set_Y_Project_Type = std::function<void(F1_Output_Type &)>;
  /** @brief F2(u) -> F2_Output_Type */
  using Mapping_F2_Type = std::function<F2_Output_Type(const U_Horizon_Type &)>;

  /* Constructor */
  ALM_Problem()
      : parametric_cost(nullptr), parametric_gradient(nullptr), u_min_matrix(),
        u_max_matrix(), mapping_f1(nullptr), set_c_project(nullptr),
        set_y_project(nullptr), mapping_f2(nullptr) {}

  /* Copy / Move: use defaults */
  ALM_Problem(const ALM_Problem &) = default;
  ALM_Problem &operator=(const ALM_Problem &) = default;
  ALM_Problem(ALM_Problem &&) noexcept = default;
  ALM_Problem &operator=(ALM_Problem &&) noexcept = default;

  /* Setters */
  inline void set_parametric_cost(const Parametric_Cost_Type &cost) {
    this->parametric_cost = cost;
  }

  inline void
  set_parametric_gradient(const Parametric_Gradient_Type &gradient) {
    this->parametric_gradient = gradient;
  }

  inline void set_u_min_matrix(const _U_Min_Matrix_Type &u_min) {
    this->u_min_matrix = u_min;
  }

  inline void set_u_max_matrix(const _U_Max_Matrix_Type &u_max) {
    this->u_max_matrix = u_max;
  }

  inline void set_mapping_f1(const Mapping_F1_Type &f1) {
    this->mapping_f1 = f1;
  }

  inline void set_c_projection(const Set_C_Project_Type &proj) {
    this->set_c_project = proj;
  }

  inline void set_y_projection(const Set_Y_Project_Type &proj) {
    this->set_y_project = proj;
  }

  inline void set_mapping_f2(const Mapping_F2_Type &f2) {
    this->mapping_f2 = f2;
  }

  /* Public members */
  Parametric_Cost_Type parametric_cost;
  Parametric_Gradient_Type parametric_gradient;
  _U_Min_Matrix_Type u_min_matrix;
  _U_Max_Matrix_Type u_max_matrix;
  Mapping_F1_Type mapping_f1;
  Set_C_Project_Type set_c_project;
  Set_Y_Project_Type set_y_project;
  Mapping_F2_Type mapping_f2;
};

// ============================================================================
// ALM_PM_Optimizer
// ============================================================================

/**
 * @brief ALM/PM solver for constrained nonlinear optimization.
 *
 * Uses PANOC as the inner solver. Solves:
 *
 *     min  f(u)
 *      u
 *     s.t. u in U      (box constraints, handled by PANOC)
 *          F1(u) in C   (ALM-type constraints)
 *          F2(u) = 0    (PM-type constraints, optional)
 *
 * @tparam CostMatrices_Type_In Type providing cost matrices, types, and
 *         dimensions (e.g. OptimizationEngine_CostMatrices).
 * @tparam N1 Dimension of F1 output (ALM constraints).
 * @tparam N2 Dimension of F2 output (PM constraints). Default 0.
 * @tparam LBFGSMemory L-BFGS memory size for inner PANOC solver.
 */
template <typename CostMatrices_Type_In, std::size_t N1, std::size_t N2 = 0,
          std::size_t LBFGSMemory = 5>
class ALM_PM_Optimizer {
public:
  /* Constants */
  static constexpr std::size_t STATE_SIZE = CostMatrices_Type_In::STATE_SIZE;
  static constexpr std::size_t INPUT_SIZE = CostMatrices_Type_In::INPUT_SIZE;
  static constexpr std::size_t OUTPUT_SIZE = CostMatrices_Type_In::OUTPUT_SIZE;
  enum : std::size_t { NP = CostMatrices_Type_In::NP };
  static constexpr std::size_t PROBLEM_SIZE = INPUT_SIZE * NP;

public:
  /* Types */
  using Value_Type = typename CostMatrices_Type_In::Value_Type;

  using X_Type = typename CostMatrices_Type_In::X_Type;
  using U_Type = typename CostMatrices_Type_In::U_Type;
  using Y_Type = typename CostMatrices_Type_In::Y_Type;

  using X_Horizon_Type = typename CostMatrices_Type_In::X_Horizon_Type;
  using U_Horizon_Type = typename CostMatrices_Type_In::U_Horizon_Type;
  using Y_Horizon_Type = typename CostMatrices_Type_In::Y_Horizon_Type;

protected:
  using _CostMatrices_Type = CostMatrices_Type_In;
  using _T = Value_Type;
  using _Gradient_Type = U_Horizon_Type;

  using _U_min_Type = typename _CostMatrices_Type::U_Min_Type;
  using _U_max_Type = typename _CostMatrices_Type::U_Max_Type;

  using _U_Min_Matrix_Type = PythonNumpy::Tile_Type<1, NP, _U_min_Type>;
  using _U_Max_Matrix_Type = PythonNumpy::Tile_Type<1, NP, _U_max_Type>;

  using _FlatVector_Type = PythonNumpy::DenseMatrix_Type<_T, PROBLEM_SIZE, 1>;

  using _F1_Output_Type = PythonNumpy::DenseMatrix_Type<_T, N1, 1>;
  using _F2_Output_Type =
      PythonNumpy::DenseMatrix_Type<_T, (N2 > 0 ? N2 : 1), 1>;
  using _Xi_Type = PythonNumpy::DenseMatrix_Type<_T, (1 + N1), 1>;

  using _Cache_Type = ALM_Cache<_T, PROBLEM_SIZE, N1, N2, LBFGSMemory>;
  using _Problem_Type = ALM_Problem<_CostMatrices_Type, N1, N2>;

  using _PANOC_Type = PANOC_Optimizer<_CostMatrices_Type, LBFGSMemory>;
  using _PANOC_SolverStatus_Type = PANOC_SolverStatus<_T>;

  using _SolverStatus_Type = ALM_SolverStatus<_T, N1>;

  /**
   * @brief Cost function object type for PANOC: f(U_horizon) -> T.
   */
  using _Cost_Function_Type = std::function<_T(const U_Horizon_Type &)>;

  /**
   * @brief Gradient function object type for PANOC:
   *        grad_f(U_horizon) -> Gradient_Type.
   */
  using _Gradient_Function_Type =
      std::function<_Gradient_Type(const U_Horizon_Type &)>;

public:
  /* Constructor */
  ALM_PM_Optimizer()
      : _cache(), _problem(), _panoc(),
        _max_outer_iterations(ALM_Constants::DEFAULT_MAX_OUTER_ITERATIONS),
        _max_inner_iterations(ALM_Constants::DEFAULT_MAX_INNER_ITERATIONS),
        _epsilon_tolerance(
            static_cast<_T>(ALM_Constants::DEFAULT_EPSILON_TOLERANCE)),
        _delta_tolerance(
            static_cast<_T>(ALM_Constants::DEFAULT_DELTA_TOLERANCE)),
        _penalty_update_factor(
            static_cast<_T>(ALM_Constants::DEFAULT_PENALTY_UPDATE_FACTOR)),
        _epsilon_update_factor(
            static_cast<_T>(ALM_Constants::DEFAULT_EPSILON_UPDATE_FACTOR)),
        _sufficient_decrease_coefficient(static_cast<_T>(
            ALM_Constants::DEFAULT_INFEASIBLE_SUFFICIENT_DECREASE_FACTOR)),
        _initial_inner_tolerance(
            static_cast<_T>(ALM_Constants::DEFAULT_INITIAL_TOLERANCE)),
        _solver_status() {}

  /* Copy constructor */
  ALM_PM_Optimizer(const ALM_PM_Optimizer &input)
      : _cache(input._cache), _problem(input._problem), _panoc(input._panoc),
        _max_outer_iterations(input._max_outer_iterations),
        _max_inner_iterations(input._max_inner_iterations),
        _epsilon_tolerance(input._epsilon_tolerance),
        _delta_tolerance(input._delta_tolerance),
        _penalty_update_factor(input._penalty_update_factor),
        _epsilon_update_factor(input._epsilon_update_factor),
        _sufficient_decrease_coefficient(
            input._sufficient_decrease_coefficient),
        _initial_inner_tolerance(input._initial_inner_tolerance),
        _solver_status(input._solver_status) {}

  /* Copy assignment */
  ALM_PM_Optimizer &operator=(const ALM_PM_Optimizer &input) {
    if (this != &input) {
      this->_cache = input._cache;
      this->_problem = input._problem;
      this->_panoc = input._panoc;
      this->_max_outer_iterations = input._max_outer_iterations;
      this->_max_inner_iterations = input._max_inner_iterations;
      this->_epsilon_tolerance = input._epsilon_tolerance;
      this->_delta_tolerance = input._delta_tolerance;
      this->_penalty_update_factor = input._penalty_update_factor;
      this->_epsilon_update_factor = input._epsilon_update_factor;
      this->_sufficient_decrease_coefficient =
          input._sufficient_decrease_coefficient;
      this->_initial_inner_tolerance = input._initial_inner_tolerance;
      this->_solver_status = input._solver_status;
    }
    return *this;
  }

  /* Move constructor */
  ALM_PM_Optimizer(ALM_PM_Optimizer &&input) noexcept
      : _cache(std::move(input._cache)), _problem(std::move(input._problem)),
        _panoc(std::move(input._panoc)),
        _max_outer_iterations(input._max_outer_iterations),
        _max_inner_iterations(input._max_inner_iterations),
        _epsilon_tolerance(input._epsilon_tolerance),
        _delta_tolerance(input._delta_tolerance),
        _penalty_update_factor(input._penalty_update_factor),
        _epsilon_update_factor(input._epsilon_update_factor),
        _sufficient_decrease_coefficient(
            input._sufficient_decrease_coefficient),
        _initial_inner_tolerance(input._initial_inner_tolerance),
        _solver_status(input._solver_status) {}

  /* Move assignment */
  ALM_PM_Optimizer &operator=(ALM_PM_Optimizer &&input) noexcept {
    if (this != &input) {
      this->_cache = std::move(input._cache);
      this->_problem = std::move(input._problem);
      this->_panoc = std::move(input._panoc);
      this->_max_outer_iterations = input._max_outer_iterations;
      this->_max_inner_iterations = input._max_inner_iterations;
      this->_epsilon_tolerance = input._epsilon_tolerance;
      this->_delta_tolerance = input._delta_tolerance;
      this->_penalty_update_factor = input._penalty_update_factor;
      this->_epsilon_update_factor = input._epsilon_update_factor;
      this->_sufficient_decrease_coefficient =
          input._sufficient_decrease_coefficient;
      this->_initial_inner_tolerance = input._initial_inner_tolerance;
      this->_solver_status = input._solver_status;
    }
    return *this;
  }

public:
  /* Setters */

  /**
   * @brief Set the ALM problem definition.
   * @param problem ALM_Problem instance with cost, gradient, constraints.
   */
  inline void set_problem(const _Problem_Type &problem) {
    this->_problem = problem;

    /* Configure PANOC with bounds from the problem */
    this->_panoc.set_u_min_matrix(problem.u_min_matrix);
    this->_panoc.set_u_max_matrix(problem.u_max_matrix);
  }

  /**
   * @brief Set the maximum number of outer and inner iterations.
   * @param outer_max Maximum outer ALM iterations.
   * @param inner_max Maximum inner PANOC iterations per outer iteration.
   */
  inline void set_solver_max_iteration(
      std::size_t outer_max = ALM_Constants::DEFAULT_MAX_OUTER_ITERATIONS,
      std::size_t inner_max = ALM_Constants::DEFAULT_MAX_INNER_ITERATIONS) {
    this->_max_outer_iterations = outer_max;
    this->_max_inner_iterations = inner_max;
  }

  /**
   * @brief Set the target tolerance for the inner solver (epsilon).
   * @param tol Positive convergence tolerance.
   */
  inline void set_epsilon_tolerance(const _T &tol) {
    this->_epsilon_tolerance = tol;
  }

  /**
   * @brief Set the infeasibility tolerance (delta).
   * @param tol Positive infeasibility tolerance.
   */
  inline void set_delta_tolerance(const _T &tol) {
    this->_delta_tolerance = tol;
  }

  /**
   * @brief Set the penalty update factor (rho > 1).
   * @param factor Penalty update factor.
   */
  inline void set_penalty_update_factor(const _T &factor) {
    this->_penalty_update_factor = factor;
  }

  /**
   * @brief Set the inner tolerance shrink factor (beta in (0,1)).
   * @param factor Epsilon update factor.
   */
  inline void set_epsilon_update_factor(const _T &factor) {
    this->_epsilon_update_factor = factor;
  }

  /**
   * @brief Set the sufficient decrease coefficient (theta in (0,1)).
   * @param coeff Sufficient decrease coefficient.
   */
  inline void set_sufficient_decrease_coefficient(const _T &coeff) {
    this->_sufficient_decrease_coefficient = coeff;
  }

  /**
   * @brief Set the initial inner tolerance (epsilon_0 >= epsilon).
   * @param tol Initial inner tolerance.
   */
  inline void set_initial_inner_tolerance(const _T &tol) {
    this->_initial_inner_tolerance = tol;
  }

  /**
   * @brief Set the initial penalty parameter c_0.
   * @param penalty Initial penalty value (must be positive).
   */
  inline void set_initial_penalty(const _T &penalty) {
    this->_cache.xi(0, 0) = penalty;
  }

  /**
   * @brief Set the initial Lagrange multiplier vector y_0.
   * @param y0 Initial Lagrange multipliers of size N1.
   */
  inline void set_initial_y(const _F1_Output_Type &y0) {
    for (std::size_t i = 0; i < N1; ++i) {
      this->_cache.xi(i + 1, 0) = y0(i, 0);
    }
  }

  /* Getters */

  /**
   * @brief Get the solver status from the last solve call.
   * @return Reference to the solver status.
   */
  inline auto get_solver_status(void) const -> const _SolverStatus_Type & {
    return this->_solver_status;
  }

  /**
   * @brief Get the number of outer and inner iterations from last solve.
   * @param outer_count Output: number of outer iterations.
   * @param inner_count Output: number of inner iterations.
   */
  inline void get_solver_step_iterated_number(std::size_t &outer_count,
                                              std::size_t &inner_count) const {
    outer_count = this->_solver_status.num_outer_iterations;
    inner_count = this->_solver_status.num_inner_iterations;
  }

  /**
   * @brief Get a reference to the internal cache.
   */
  inline auto get_cache(void) -> _Cache_Type & { return this->_cache; }
  inline auto get_cache(void) const -> const _Cache_Type & {
    return this->_cache;
  }

  /**
   * @brief Get a reference to the internal PANOC optimizer.
   */
  inline auto get_panoc(void) -> _PANOC_Type & { return this->_panoc; }
  inline auto get_panoc(void) const -> const _PANOC_Type & {
    return this->_panoc;
  }

  /* Function */

  /**
   * @brief Solve the ALM/PM problem.
   *
   * @param u_initial Initial guess for the control horizon.
   * @return Optimized control horizon.
   */
  inline auto solve(const U_Horizon_Type &u_initial) -> U_Horizon_Type {
    this->_cache.reset();
    this->_cache.panoc_cache.tolerance = this->_initial_inner_tolerance;

    U_Horizon_Type u = u_initial;

    std::size_t num_outer_iterations = 0;
    ALM_ExitStatus exit_status = ALM_ExitStatus::CONVERGED;
    bool should_continue = true;

    for (std::size_t outer_iter = 0; outer_iter < this->_max_outer_iterations;
         ++outer_iter) {
      num_outer_iterations += 1;
      should_continue = this->_step(u);

      if (!should_continue) {
        break;
      }
    }

    if (num_outer_iterations == this->_max_outer_iterations &&
        should_continue) {
      exit_status = ALM_ExitStatus::NOT_CONVERGED_ITERATIONS;
    }

    /* Extract final penalty parameter */
    _T c = this->_cache.xi(0, 0);

    /* Compute original cost at solution (penalty terms excluded) */
    _T cost_value = this->_compute_cost_at_solution(u);

    /* Build result */
    _F1_Output_Type lagrange = this->_cache.y_plus;

    this->_solver_status.exit_status = exit_status;
    this->_solver_status.num_outer_iterations = num_outer_iterations;
    this->_solver_status.num_inner_iterations =
        this->_cache.inner_iteration_count;
    this->_solver_status.last_problem_norm_fpr =
        this->_cache.last_inner_problem_norm_fpr;
    this->_solver_status.lagrange_multipliers = lagrange;
    this->_solver_status.penalty = c;
    this->_solver_status.delta_y_norm = this->_cache.delta_y_norm_plus;
    this->_solver_status.f2_norm = this->_cache.f2_norm_plus;
    this->_solver_status.cost = cost_value;

    return u;
  }

protected:
  /**
   * @brief Perform one ALM outer iteration.
   *
   * @param u Decision variable (modified in-place with PANOC solution).
   * @return true if optimization should continue, false if converged.
   */
  inline bool _step(U_Horizon_Type &u) {
    /* 1. Project y onto set Y */
    this->_project_on_set_y();

    /* 2. Solve inner problem via PANOC */
    _PANOC_SolverStatus_Type inner_status = this->_solve_inner_problem(u);
    this->_cache.last_inner_problem_norm_fpr =
        inner_status.norm_fixed_point_residual;
    this->_cache.inner_iteration_count += inner_status.number_of_iteration;

    /* 3. Update Lagrange multipliers:
     *    y^+ <- y + c * [F1(u) - Pi_C(F1(u) + y/c)] */
    this->_update_lagrange_multipliers(u);

    /* 4. Compute infeasibility measures */
    this->_compute_pm_infeasibility(u); /* ||F2(u)|| */
    this->_compute_alm_infeasibility(); /* ||y^+ - y|| */

    /* 5. Check exit criterion */
    if (this->_is_exit_criterion_satisfied()) {
      return false; /* converged */
    }

    /* 6. Update penalty parameter if insufficient decrease */
    if (!this->_is_penalty_stall_criterion()) {
      this->_update_penalty_parameter();
    }

    /* 7. Shrink inner tolerance */
    this->_update_inner_tolerance();

    /* 8. Final bookkeeping */
    this->_final_cache_update();

    return true;
  }

  /**
   * @brief Project Lagrange multipliers y onto set Y (in-place on xi).
   */
  inline void _project_on_set_y(void) {
    if (N1 > 0 && this->_problem.set_y_project) {
      _F1_Output_Type y;
      for (std::size_t i = 0; i < N1; ++i) {
        y(i, 0) = this->_cache.xi(i + 1, 0);
      }
      this->_problem.set_y_project(y);
      for (std::size_t i = 0; i < N1; ++i) {
        this->_cache.xi(i + 1, 0) = y(i, 0);
      }
    }
  }

  /**
   * @brief Solve the inner problem min_{u in U} psi(u; xi) using PANOC.
   *
   * @param u Decision variable (modified in-place with PANOC solution).
   * @return Inner PANOC solver status.
   */
  inline auto _solve_inner_problem(U_Horizon_Type &u)
      -> _PANOC_SolverStatus_Type {

    /* Capture xi by pointer for lambda closures */
    _Xi_Type *xi_ptr = &(this->_cache.xi);
    const typename _Problem_Type::Parametric_Cost_Type &param_cost =
        this->_problem.parametric_cost;
    const typename _Problem_Type::Parametric_Gradient_Type &param_grad =
        this->_problem.parametric_gradient;

    /* Build non-parametric cost/gradient by capturing xi */
    _Cost_Function_Type cost_func =
        [xi_ptr, &param_cost](const U_Horizon_Type &u_) -> _T {
      return param_cost(u_, *xi_ptr);
    };

    _Gradient_Function_Type grad_func =
        [xi_ptr, &param_grad](const U_Horizon_Type &u_) -> _Gradient_Type {
      return param_grad(u_, *xi_ptr);
    };

    this->_panoc.set_cost_function(cost_func);
    this->_panoc.set_gradient_function(grad_func);
    this->_panoc.set_max_iteration(this->_max_inner_iterations);
    this->_panoc.set_tolerance(this->_cache.panoc_cache.tolerance);

    u = this->_panoc.solve(u);

    return this->_panoc.get_solver_status();
  }

  /**
   * @brief Update Lagrange multipliers:
   *    y^+ = y + c * [F1(u) - Pi_C(F1(u) + y/c)]
   *
   * Steps:
   *   1. w = F1(u)
   *   2. y_plus = w + y/c
   *   3. y_plus = Pi_C(y_plus)
   *   4. y_plus = y + c * (w - y_plus)
   */
  inline void _update_lagrange_multipliers(const U_Horizon_Type &u) {
    if (N1 == 0) {
      return;
    }
    if (!this->_problem.mapping_f1 || !this->_problem.set_c_project) {
      return;
    }

    _T c = this->_cache.xi(0, 0);

    /* Extract y from xi */
    _F1_Output_Type y;
    for (std::size_t i = 0; i < N1; ++i) {
      y(i, 0) = this->_cache.xi(i + 1, 0);
    }

    /* Step 1: w = F1(u) */
    _F1_Output_Type w = this->_problem.mapping_f1(u);

    /* Store w in aux */
    this->_cache.w_alm_aux = w;

    /* Step 2: y_plus = F1(u) + y/c */
    this->_cache.y_plus = w + (static_cast<_T>(1) / c) * y;

    /* Step 3: y_plus = Pi_C(y_plus) */
    this->_problem.set_c_project(this->_cache.y_plus);

    /* Step 4: y_plus = y + c * (F1(u) - Pi_C(F1(u) + y/c)) */
    this->_cache.y_plus = y + c * (w - this->_cache.y_plus);
  }

  /**
   * @brief Compute ALM infeasibility: ||y^+ - y||.
   */
  inline void _compute_alm_infeasibility(void) {
    if (N1 > 0) {
      _F1_Output_Type y;
      for (std::size_t i = 0; i < N1; ++i) {
        y(i, 0) = this->_cache.xi(i + 1, 0);
      }
      _F1_Output_Type diff = this->_cache.y_plus - y;
      _T norm_sq = static_cast<_T>(0);
      for (std::size_t i = 0; i < N1; ++i) {
        norm_sq += diff(i, 0) * diff(i, 0);
      }
      this->_cache.delta_y_norm_plus =
          static_cast<_T>(std::sqrt(static_cast<double>(norm_sq)));
    }
  }

  /**
   * @brief Compute PM infeasibility: ||F2(u)||.
   */
  inline void _compute_pm_infeasibility(const U_Horizon_Type &u) {
    if (N2 > 0 && this->_problem.mapping_f2) {
      this->_cache.w_pm = this->_problem.mapping_f2(u);
      _T norm_sq = static_cast<_T>(0);
      for (std::size_t i = 0; i < N2; ++i) {
        norm_sq += this->_cache.w_pm(i, 0) * this->_cache.w_pm(i, 0);
      }
      this->_cache.f2_norm_plus =
          static_cast<_T>(std::sqrt(static_cast<double>(norm_sq)));
    }
  }

  /**
   * @brief Check if (epsilon, delta)-AKKT conditions are satisfied.
   *
   * Three criteria must hold simultaneously:
   *   1. ||delta y|| <= c * delta   (or no ALM constraints)
   *   2. ||F2(u)|| <= delta         (or no PM constraints)
   *   3. epsilon_nu <= epsilon      (inner tolerance has reached target)
   *
   * @return true if all criteria are satisfied.
   */
  inline bool _is_exit_criterion_satisfied(void) const {
    const _T small_eps = static_cast<_T>(ALM_Constants::SMALL_EPSILON);

    /* Criterion 1: ||delta y|| <= c * delta */
    bool criterion_1 = true;
    if (N1 > 0) {
      _T c = this->_cache.xi(0, 0);
      criterion_1 = (this->_cache.iteration > 0) &&
                    (this->_cache.delta_y_norm_plus <=
                     c * this->_delta_tolerance + small_eps);
    }

    /* Criterion 2: ||F2(u)|| <= delta */
    bool criterion_2 = true;
    if (N2 > 0) {
      criterion_2 =
          (this->_cache.f2_norm_plus <= this->_delta_tolerance + small_eps);
    }

    /* Criterion 3: current inner tolerance <= target epsilon */
    bool criterion_3 = (this->_cache.panoc_cache.tolerance <=
                        this->_epsilon_tolerance + small_eps);

    return criterion_1 && criterion_2 && criterion_3;
  }

  /**
   * @brief Check if penalty update should be skipped (sufficient
   * decrease).
   *
   * Returns true if the penalty should NOT be updated (stall),
   * which happens when iteration == 0 or there was sufficient
   * decrease in the infeasibility measures.
   *
   * @return true if sufficient decrease occurred (no penalty update
   * needed).
   */
  inline bool _is_penalty_stall_criterion(void) const {
    const _T small_eps = static_cast<_T>(ALM_Constants::SMALL_EPSILON);

    if (this->_cache.iteration == 0) {
      return true;
    }

    bool is_alm = (N1 > 0);
    bool is_pm = (N2 > 0);

    bool criterion_alm =
        (this->_cache.delta_y_norm_plus <=
         this->_sufficient_decrease_coefficient * this->_cache.delta_y_norm +
             small_eps);

    bool criterion_pm =
        (this->_cache.f2_norm_plus <=
         this->_sufficient_decrease_coefficient * this->_cache.f2_norm +
             small_eps);

    if (is_alm && !is_pm) {
      return criterion_alm;
    } else if (!is_alm && is_pm) {
      return criterion_pm;
    } else if (is_alm && is_pm) {
      return criterion_alm && criterion_pm;
    }

    return false;
  }

  /**
   * @brief Multiply penalty parameter c by penalty_update_factor.
   */
  inline void _update_penalty_parameter(void) {
    this->_cache.xi(0, 0) *= this->_penalty_update_factor;
  }

  /**
   * @brief Shrink inner tolerance:
   *        epsilon <- max(epsilon, beta * epsilon).
   */
  inline void _update_inner_tolerance(void) {
    _T current = this->_cache.panoc_cache.tolerance;
    _T candidate = current * this->_epsilon_update_factor;
    this->_cache.panoc_cache.tolerance = (candidate > this->_epsilon_tolerance)
                                             ? candidate
                                             : this->_epsilon_tolerance;
  }

  /**
   * @brief End-of-iteration bookkeeping: increment counter, shift
   * infeasibility measures, copy y^+ -> y, and reset PANOC cache.
   */
  inline void _final_cache_update(void) {
    this->_cache.iteration += 1;
    this->_cache.delta_y_norm = this->_cache.delta_y_norm_plus;
    this->_cache.f2_norm = this->_cache.f2_norm_plus;

    /* Copy y^+ into xi (update y) */
    if (N1 > 0) {
      for (std::size_t i = 0; i < N1; ++i) {
        this->_cache.xi(i + 1, 0) = this->_cache.y_plus(i, 0);
      }
    }

    /* Reset PANOC cache for next inner solve */
    this->_cache.panoc_cache.reset();
  }

  /**
   * @brief Compute the original cost f(u) at the solution, excluding
   * penalty terms.
   *
   * This is done by temporarily setting c = 0 in xi. The augmented
   * cost with c = 0 reduces to f(u) because the penalty term becomes
   * 0.
   *
   * @param u Decision variable at solution.
   * @return Original cost value f(u).
   */
  inline auto _compute_cost_at_solution(const U_Horizon_Type &u) -> _T {
    _T saved_c = this->_cache.xi(0, 0);
    this->_cache.xi(0, 0) = static_cast<_T>(0);
    _T cost = this->_problem.parametric_cost(u, this->_cache.xi);
    this->_cache.xi(0, 0) = saved_c;
    return cost;
  }

protected:
  /* Variables */
  _Cache_Type _cache;
  _Problem_Type _problem;
  _PANOC_Type _panoc;

  std::size_t _max_outer_iterations;
  std::size_t _max_inner_iterations;
  _T _epsilon_tolerance;
  _T _delta_tolerance;
  _T _penalty_update_factor;
  _T _epsilon_update_factor;
  _T _sufficient_decrease_coefficient;
  _T _initial_inner_tolerance;

  _SolverStatus_Type _solver_status;
};

// ============================================================================
// Factory and Type Alias
// ============================================================================

/**
 * @brief Factory function to create a default ALM_PM_Optimizer instance.
 *
 * @tparam CostMatrices_Type The cost matrices type.
 * @tparam N1 Dimension of F1 output (ALM constraints).
 * @tparam N2 Dimension of F2 output (PM constraints).
 * @tparam LBFGSMemory L-BFGS memory size.
 * @return A new default-initialized ALM_PM_Optimizer.
 */
template <typename CostMatrices_Type, std::size_t N1, std::size_t N2 = 0,
          std::size_t LBFGSMemory = 5>
inline auto make_ALM_PM_Optimizer(void)
    -> ALM_PM_Optimizer<CostMatrices_Type, N1, N2, LBFGSMemory> {
  return ALM_PM_Optimizer<CostMatrices_Type, N1, N2, LBFGSMemory>();
}

/**
 * @brief Alias template for ALM_PM_Optimizer.
 *
 * @tparam CostMatrices_Type The cost matrices type.
 * @tparam N1 Dimension of F1 output (ALM constraints).
 * @tparam N2 Dimension of F2 output (PM constraints).
 * @tparam LBFGSMemory L-BFGS memory size.
 */
template <typename CostMatrices_Type, std::size_t N1, std::size_t N2 = 0,
          std::size_t LBFGSMemory = 5>
using ALM_PM_Optimizer_Type =
    ALM_PM_Optimizer<CostMatrices_Type, N1, N2, LBFGSMemory>;

/**
 * @brief Alias template for ALM_Factory.
 *
 * @tparam CostMatrices_Type The cost matrices type.
 * @tparam N1 Dimension of F1 output (ALM constraints).
 * @tparam N2 Dimension of F2 output (PM constraints).
 */
template <typename CostMatrices_Type, std::size_t N1, std::size_t N2 = 0>
using ALM_Factory_Type = ALM_Factory<CostMatrices_Type, N1, N2>;

/**
 * @brief Alias template for ALM_Problem.
 *
 * @tparam CostMatrices_Type The cost matrices type.
 * @tparam N1 Dimension of F1 output (ALM constraints).
 * @tparam N2 Dimension of F2 output (PM constraints).
 */
template <typename CostMatrices_Type, std::size_t N1, std::size_t N2 = 0>
using ALM_Problem_Type = ALM_Problem<CostMatrices_Type, N1, N2>;

/**
 * @brief Alias template for ALM_Cache.
 *
 * @tparam T Scalar value type.
 * @tparam ProblemSize Decision variable dimension.
 * @tparam N1 ALM constraint dimension.
 * @tparam N2 PM constraint dimension.
 * @tparam LBFGSMemory L-BFGS memory size.
 */
template <typename T, std::size_t ProblemSize, std::size_t N1,
          std::size_t N2 = 0, std::size_t LBFGSMemory = 5>
using ALM_Cache_Type = ALM_Cache<T, ProblemSize, N1, N2, LBFGSMemory>;

/**
 * @brief Alias template for ALM_SolverStatus.
 *
 * @tparam T Scalar value type.
 * @tparam N1 ALM constraint dimension.
 */
template <typename T, std::size_t N1>
using ALM_SolverStatus_Type = ALM_SolverStatus<T, N1>;

/**
 * @brief Alias template for BoxProjectionOperator.
 */
template <typename T, std::size_t Size>
using BoxProjectionOperator_Type = BoxProjectionOperator<T, Size>;

/**
 * @brief Alias template for BallProjectionOperator.
 */
template <typename T, std::size_t Size>
using BallProjectionOperator_Type = BallProjectionOperator<T, Size>;

} // namespace PythonOptimization

#endif // __PYTHON_OPTIMIZATION_ALM_PM_OPTIMIZER_HPP__
