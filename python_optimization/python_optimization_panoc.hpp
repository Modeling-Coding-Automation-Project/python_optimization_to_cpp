#ifndef __PYTHON_OPTIMIZATION_PANOC_HPP__
#define __PYTHON_OPTIMIZATION_PANOC_HPP__

#include "python_optimization_common.hpp"
#include "python_optimization_utility.hpp"

#include "python_control.hpp"
#include "python_numpy.hpp"

#include <cmath>
#include <functional>
#include <type_traits>

namespace PythonOptimization {

namespace PANOC_Constants {

/** @brief Minimum estimated Lipschitz constant (initial estimate floor). */
static constexpr double MIN_L_ESTIMATE_DEFAULT = 1e-10;

/** @brief gamma = GAMMA_L_COEFFICIENT_DEFAULT / L. */
static constexpr double GAMMA_L_COEFFICIENT_DEFAULT = 0.95;

/** @brief Delta for Lipschitz estimation perturbation. */
static constexpr double DELTA_LIPSCHITZ_DEFAULT = 1e-12;

/** @brief Epsilon for Lipschitz estimation perturbation. */
static constexpr double EPSILON_LIPSCHITZ_DEFAULT = 1e-6;

/** @brief Safety parameter for strict inequality in Lipschitz update. */
static constexpr double LIPSCHITZ_UPDATE_EPSILON_DEFAULT = 1e-6;

/** @brief Maximum iterations for updating the Lipschitz constant. */
static constexpr std::size_t MAX_LIPSCHITZ_UPDATE_ITERATIONS_DEFAULT = 10;

/** @brief Maximum possible Lipschitz constant. */
static constexpr double MAX_LIPSCHITZ_CONSTANT_DEFAULT = 1e9;

/** @brief Maximum number of line-search iterations. */
static constexpr std::size_t MAX_LINESEARCH_ITERATIONS_DEFAULT = 10;

/** @brief Default maximum PANOC iterations. */
static constexpr std::size_t MAX_ITERATION_DEFAULT = 100;

/* L-BFGS defaults */

/** @brief Minimum accepted value of s^T y for an L-BFGS update. */
static constexpr double SY_EPSILON_DEFAULT = 1e-10;

/** @brief C-BFGS parameter epsilon. */
static constexpr double CBFGS_EPSILON_DEFAULT = 1e-8;

/** @brief C-BFGS parameter alpha (must be 0, 1, or 2). */
static constexpr std::size_t CBFGS_ALPHA_DEFAULT = 1;

/** @brief Threshold below which ||s||^2 is considered zero. */
static constexpr double NORM_S_SMALL_LIMIT = 1e-30;

} // namespace PANOC_Constants

/**
 * @brief Exit status of the PANOC solver.
 */
enum class PANOC_ExitStatus {
  CONVERGED = 0,
  NOT_CONVERGED_ITERATIONS = 1,
  NOT_FINITE_COMPUTATION = 2
};

/**
 * @brief Result returned by PANOC_Optimizer::solve.
 *
 * @tparam T Scalar value type.
 */
template <typename T> struct PANOC_SolverStatus {
  PANOC_ExitStatus exit_status;
  std::size_t number_of_iteration;
  T norm_fixed_point_residual;
  T cost_value;

  /**
   * @brief Check whether the solver converged.
   * @return true if converged.
   */
  inline bool has_converged(void) const {
    return exit_status == PANOC_ExitStatus::CONVERGED;
  }
};

/**
 * @brief Circular buffer that stores fixed-size column vectors (or scalars)
 * with O(1) push.
 *
 * Newest data is accessed via get(0), one-before-newest via get(1), etc.
 * No data copying occurs on push - only an internal index is advanced.
 *
 * @tparam T Scalar value type (e.g. double).
 * @tparam BufferSize Maximum number of entries the buffer can hold.
 * @tparam ElementSize Size of each vector element (number of rows).
 *                     If 0, each element is a scalar.
 */
template <typename T, std::size_t BufferSize, std::size_t ElementSize = 0>
class VectorRingBuffer {
public:
  /* Constructor */
  VectorRingBuffer()
      : _head(0), _active_size(0), _data_matrix(), _data_scalar() {}

  /* Copy / Move: use defaults */
  VectorRingBuffer(const VectorRingBuffer &) = default;
  VectorRingBuffer &operator=(const VectorRingBuffer &) = default;
  VectorRingBuffer(VectorRingBuffer &&) noexcept = default;
  VectorRingBuffer &operator=(VectorRingBuffer &&) noexcept = default;

  /**
   * @brief Reset the buffer (O(1) - only resets counters).
   */
  inline void reset(void) {
    this->_head = 0;
    this->_active_size = 0;
  }

  /**
   * @brief Push a new vector value, overwriting the oldest entry if full.
   * @param value Column vector (ElementSize x 1) to push.
   */
  template <std::size_t ES = ElementSize,
            typename std::enable_if<(ES > 0), int>::type = 0>
  inline void push(const PythonNumpy::DenseMatrix_Type<T, ES, 1> &value) {
    for (std::size_t i = 0; i < ES; ++i) {
      this->_data_matrix[i][this->_head] = value(0, i);
    }
    this->_advance_head();
  }

  /**
   * @brief Push a new scalar value, overwriting the oldest entry if full.
   * @param value Scalar to push.
   */
  template <std::size_t ES = ElementSize,
            typename std::enable_if<(ES == 0), int>::type = 0>
  inline void push(const T &value) {
    this->_data_scalar[this->_head] = value;
    this->_advance_head();
  }

  /**
   * @brief Get the i-th most recent vector element.
   *
   * @tparam ES_ Dispatching parameter (must equal ElementSize, used for
   * SFINAE).
   * @param index_from_latest 0 = most recent, 1 = one before newest, ...
   * @return Column vector (ElementSize x 1).
   */
  template <std::size_t ES_, typename std::enable_if<(ES_ > 0), int>::type = 0>
  inline auto get(std::size_t index_from_latest) const
      -> PythonNumpy::DenseMatrix_Type<T, ES_, 1> {
    std::size_t idx = this->_resolve_index(index_from_latest);
    PythonNumpy::DenseMatrix_Type<T, ES_, 1> result;
    for (std::size_t i = 0; i < ES_; ++i) {
      result(0, i) = this->_data_matrix[i][idx];
    }
    return result;
  }

  /**
   * @brief Get the i-th most recent scalar element.
   *
   * @tparam ES_ Dispatching parameter (must be 0, used for SFINAE).
   * @param index_from_latest 0 = most recent, 1 = one before newest, ...
   * @return Scalar value.
   */
  template <std::size_t ES_, typename std::enable_if<(ES_ == 0), int>::type = 0>
  inline auto get(std::size_t index_from_latest) const -> T {
    std::size_t idx = this->_resolve_index(index_from_latest);
    return this->_data_scalar[idx];
  }

  /**
   * @brief Number of valid entries currently in the buffer.
   */
  inline auto get_active_size(void) const -> std::size_t {
    return this->_active_size;
  }

private:
  inline void _advance_head(void) {
    this->_head += 1;
    if (this->_head >= BufferSize) {
      this->_head = 0;
    }
    if (this->_active_size < BufferSize) {
      this->_active_size += 1;
    }
  }

  inline auto _resolve_index(std::size_t index_from_latest) const
      -> std::size_t {
    /* index_from_latest must be < _active_size (caller's responsibility). */
    std::size_t idx;
    if (this->_head >= (index_from_latest + 1)) {
      idx = this->_head - 1 - index_from_latest;
    } else {
      idx = BufferSize + this->_head - 1 - index_from_latest;
    }
    return idx;
  }

  std::size_t _head;
  std::size_t _active_size;

  /* Storage: vectors stored column-wise, scalars in a flat array. */
  T _data_matrix[ElementSize > 0 ? ElementSize : 1][BufferSize];
  T _data_scalar[BufferSize];
};

/**
 * @brief Limited-memory BFGS buffer with C-BFGS safeguard.
 *
 * Stores pairs (s_k, y_k) and computes the product H * q using the standard
 * two-loop recursion, where H is the L-BFGS approximation of the inverse
 * Hessian.
 *
 * @tparam T Scalar value type.
 * @tparam ProblemSize Dimension of the decision variable (column vector size).
 * @tparam MemorySize Number of (s, y) pairs to store (L-BFGS memory).
 */
template <typename T, std::size_t ProblemSize, std::size_t MemorySize>
class L_BFGS_Buffer {
public:
  using Vector_Type = PythonNumpy::DenseMatrix_Type<T, ProblemSize, 1>;

  /* Constructor */
  L_BFGS_Buffer()
      : _sy_epsilon(static_cast<T>(PANOC_Constants::SY_EPSILON_DEFAULT)),
        _cbfgs_alpha(PANOC_Constants::CBFGS_ALPHA_DEFAULT),
        _cbfgs_epsilon(static_cast<T>(PANOC_Constants::CBFGS_EPSILON_DEFAULT)),
        _s(), _y(), _rho(), _gamma(static_cast<T>(1)), _old_state(), _old_g(),
        _first_old(true) {}

  /* Copy / Move: use defaults */
  L_BFGS_Buffer(const L_BFGS_Buffer &) = default;
  L_BFGS_Buffer &operator=(const L_BFGS_Buffer &) = default;
  L_BFGS_Buffer(L_BFGS_Buffer &&) noexcept = default;
  L_BFGS_Buffer &operator=(L_BFGS_Buffer &&) noexcept = default;

  /**
   * @brief Clear the buffer (cheap - just resets flags/counters).
   */
  inline void reset(void) {
    this->_s.reset();
    this->_y.reset();
    this->_rho.reset();
    this->_first_old = true;
  }

  /**
   * @brief Feed a new (gradient, state) pair to the buffer.
   *
   * @param g Current gradient (or FPR) vector.
   * @param state Current iterate (decision variable).
   * @return true if the pair was accepted, false if rejected.
   */
  inline bool update_hessian(const Vector_Type &g, const Vector_Type &state) {
    if (this->_first_old) {
      this->_first_old = false;
      this->_old_state = state;
      this->_old_g = g;
      return true;
    }

    /* Compute s and y */
    Vector_Type s_new = state - this->_old_state;
    Vector_Type y_new = g - this->_old_g;

    /* Validate and compute rho */
    T rho_new = static_cast<T>(0);
    bool valid = this->_compute_rho_if_valid(g, s_new, y_new, rho_new);
    if (!valid) {
      return false;
    }

    /* Save current as "old" */
    this->_old_state = state;
    this->_old_g = g;

    /* Push new pair into ring buffers */
    this->_s.push(s_new);
    this->_y.push(y_new);
    this->_rho.push(rho_new);

    /* Update H0 scaling: gamma = (s^T y) / (y^T y) */
    Vector_Type s0 = this->_s.template get<ProblemSize>(0);
    Vector_Type y0 = this->_y.template get<ProblemSize>(0);
    T ys = _inner_product(s0, y0);
    T yy = _inner_product(y0, y0);
    if (yy > static_cast<T>(0)) {
      this->_gamma = ys / yy;
    }

    return true;
  }

  /**
   * @brief Apply the L-BFGS inverse Hessian approximation in-place.
   *
   * On entry q is the gradient (or FPR); on exit it contains H * q.
   * Uses the standard two-loop recursion.
   *
   * @param q Vector to transform in-place.
   */
  inline void apply_hessian(Vector_Type &q) const {
    if (this->_s.get_active_size() == 0) {
      return; /* no curvature info yet - return q unchanged */
    }

    T alpha[MemorySize];

    /* --- forward pass --- */
    for (std::size_t i = 0; i < this->_s.get_active_size(); ++i) {
      Vector_Type si = this->_s.template get<ProblemSize>(i);
      Vector_Type yi = this->_y.template get<ProblemSize>(i);
      T rho_i = this->_rho.template get<0>(i);

      alpha[i] = rho_i * _inner_product(si, q);
      q = q - alpha[i] * yi;
    }

    /* Apply H0 = gamma * I */
    q = this->_gamma * q;

    /* --- backward pass --- */
    for (std::size_t ii = this->_s.get_active_size(); ii > 0; --ii) {
      std::size_t i = ii - 1;
      Vector_Type si = this->_s.template get<ProblemSize>(i);
      Vector_Type yi = this->_y.template get<ProblemSize>(i);
      T rho_i = this->_rho.template get<0>(i);

      T beta = rho_i * _inner_product(yi, q);
      q = q + (alpha[i] - beta) * si;
    }
  }

  /* Setter */
  inline void set_sy_epsilon(const T &val) { this->_sy_epsilon = val; }
  inline void set_cbfgs_alpha(std::size_t val) { this->_cbfgs_alpha = val; }
  inline void set_cbfgs_epsilon(const T &val) { this->_cbfgs_epsilon = val; }

private:
  /**
   * @brief Check C-BFGS and curvature conditions for the (s, y) pair.
   *
   * @param g Current gradient vector.
   * @param s Step difference vector.
   * @param y Gradient difference vector.
   * @param rho_out Output: 1 / (s^T y) if accepted.
   * @return true if the pair is accepted.
   */
  inline bool _compute_rho_if_valid(const Vector_Type &g, const Vector_Type &s,
                                    const Vector_Type &y, T &rho_out) const {
    T ys = _inner_product(s, y);
    T norm_s_sq = _inner_product(s, s);

    if (norm_s_sq <= static_cast<T>(PANOC_Constants::NORM_S_SMALL_LIMIT)) {
      return false;
    }
    if (this->_sy_epsilon > static_cast<T>(0) && ys <= this->_sy_epsilon) {
      return false;
    }

    if (this->_cbfgs_epsilon > static_cast<T>(0) && this->_cbfgs_alpha > 0) {
      T lhs = ys / norm_s_sq;
      T norm_g = PythonNumpy::norm(g);
      T rhs = this->_cbfgs_epsilon;
      if (this->_cbfgs_alpha == 1) {
        rhs *= norm_g;
      } else if (this->_cbfgs_alpha == 2) {
        rhs *= norm_g * norm_g;
      }
      if (lhs <= rhs) {
        return false;
      }
    }

    rho_out = static_cast<T>(1) / ys;
    return true;
  }

  /**
   * @brief Compute inner product (a^T b) for column vectors.
   */
  static inline T _inner_product(const Vector_Type &a, const Vector_Type &b) {
    T sum = static_cast<T>(0);
    for (std::size_t i = 0; i < ProblemSize; ++i) {
      sum += a(0, i) * b(0, i);
    }
    return sum;
  }

  T _sy_epsilon;
  std::size_t _cbfgs_alpha;
  T _cbfgs_epsilon;

  VectorRingBuffer<T, MemorySize, ProblemSize> _s;
  VectorRingBuffer<T, MemorySize, ProblemSize> _y;
  VectorRingBuffer<T, MemorySize, 0> _rho;

  T _gamma; /* initial Hessian scaling H0 = gamma * I */
  Vector_Type _old_state;
  Vector_Type _old_g;
  bool _first_old;
};

/**
 * @brief Pre-allocated working arrays for the PANOC algorithm.
 *
 * Create once and reuse across multiple solve calls to avoid repeated memory
 * allocation.
 *
 * @tparam T Scalar value type.
 * @tparam ProblemSize Dimension of the decision variable vector u.
 * @tparam LBFGSMemory L-BFGS memory size (number of stored pairs).
 */
template <typename T, std::size_t ProblemSize, std::size_t LBFGSMemory>
class PANOC_Cache {
public:
  using Vector_Type = PythonNumpy::DenseMatrix_Type<T, ProblemSize, 1>;
  using LBFGS_Type = L_BFGS_Buffer<T, ProblemSize, LBFGSMemory>;

  /* Constructor */
  PANOC_Cache()
      : tolerance(static_cast<T>(1e-4)), lbfgs(), gradient_u(), u_half_step(),
        gradient_step(), direction_lbfgs(), u_plus(), gamma_fpr(),
        gamma(static_cast<T>(0)), norm_gamma_fpr(static_cast<T>(1e30)),
        tau(static_cast<T>(1)), lipschitz_constant(static_cast<T>(0)),
        sigma(static_cast<T>(0)), cost_value(static_cast<T>(0)),
        rhs_ls(static_cast<T>(0)), lhs_ls(static_cast<T>(0)), iteration(0) {}

  explicit PANOC_Cache(const T &tolerance_in)
      : tolerance(tolerance_in), lbfgs(), gradient_u(), u_half_step(),
        gradient_step(), direction_lbfgs(), u_plus(), gamma_fpr(),
        gamma(static_cast<T>(0)), norm_gamma_fpr(static_cast<T>(1e30)),
        tau(static_cast<T>(1)), lipschitz_constant(static_cast<T>(0)),
        sigma(static_cast<T>(0)), cost_value(static_cast<T>(0)),
        rhs_ls(static_cast<T>(0)), lhs_ls(static_cast<T>(0)), iteration(0) {}

  /* Copy / Move: use defaults */
  PANOC_Cache(const PANOC_Cache &) = default;
  PANOC_Cache &operator=(const PANOC_Cache &) = default;
  PANOC_Cache(PANOC_Cache &&) noexcept = default;
  PANOC_Cache &operator=(PANOC_Cache &&) noexcept = default;

  /**
   * @brief Reset the cache to its initial state (called before each solve).
   */
  inline void reset(void) {
    this->lbfgs.reset();
    this->lhs_ls = static_cast<T>(0);
    this->rhs_ls = static_cast<T>(0);
    this->tau = static_cast<T>(1);
    this->lipschitz_constant = static_cast<T>(0);
    this->sigma = static_cast<T>(0);
    this->cost_value = static_cast<T>(0);
    this->iteration = 0;
    this->gamma = static_cast<T>(0);
    this->norm_gamma_fpr = static_cast<T>(1e30);
  }

  /**
   * @brief Check FPR convergence: ||gamma * FPR|| < tolerance.
   * @return true if converged.
   */
  inline bool exit_condition(void) const {
    return this->norm_gamma_fpr < this->tolerance;
  }

  /* Public members (working arrays and scalars) */
  T tolerance;

  LBFGS_Type lbfgs;

  Vector_Type gradient_u;
  Vector_Type u_half_step;
  Vector_Type gradient_step;
  Vector_Type direction_lbfgs;
  Vector_Type u_plus;
  Vector_Type gamma_fpr;

  T gamma;
  T norm_gamma_fpr;
  T tau;
  T lipschitz_constant;
  T sigma;
  T cost_value;
  T rhs_ls;
  T lhs_ls;
  std::size_t iteration;
};

/**
 * @brief PANOC solver for box-constrained nonlinear optimization.
 *
 * Solves:
 *
 *     min  cost_func(u)
 *      u
 *     s.t. u_min <= u <= u_max   (element-wise)
 *
 * This class is templated on the CostMatrices type, following the same
 * pattern as SQP_ActiveSet_PCG_PLS.
 *
 * @tparam CostMatrices_Type_In Type providing cost matrices, types, and
 *         dimensions (e.g. OptimizationEngine_CostMatrices).
 * @tparam LBFGSMemory L-BFGS memory size (number of stored pairs).
 */
template <typename CostMatrices_Type_In, std::size_t LBFGSMemory = 5>
class PANOC_Optimizer {
public:
  /* Constants */
  static constexpr std::size_t STATE_SIZE = CostMatrices_Type_In::STATE_SIZE;
  static constexpr std::size_t INPUT_SIZE = CostMatrices_Type_In::INPUT_SIZE;
  static constexpr std::size_t OUTPUT_SIZE = CostMatrices_Type_In::OUTPUT_SIZE;
  enum : std::size_t { NP = CostMatrices_Type_In::NP };

  /** @brief Problem size = INPUT_SIZE * NP (flattened U_Horizon). */
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

  /**
   * @brief Flat column vector type for PANOC internal computations.
   *
   * The PANOC algorithm treats the entire U_Horizon as a single
   * flat column vector of size (INPUT_SIZE * NP) x 1.
   */
  using _FlatVector_Type = PythonNumpy::DenseMatrix_Type<_T, PROBLEM_SIZE, 1>;

  using _Cache_Type = PANOC_Cache<_T, PROBLEM_SIZE, LBFGSMemory>;

  using _SolverStatus_Type = PANOC_SolverStatus<_T>;

  /**
   * @brief Cost function object type: f(U_horizon) -> T.
   *
   * The function takes U_Horizon_Type and returns a scalar cost.
   */
  using _Cost_Function_Type = std::function<_T(const U_Horizon_Type &)>;

  /**
   * @brief Gradient function object type: grad_f(U_horizon) -> Gradient_Type.
   *
   * The function takes U_Horizon_Type and returns the gradient.
   */
  using _Gradient_Function_Type =
      std::function<_Gradient_Type(const U_Horizon_Type &)>;

public:
  /* Constructor */
  PANOC_Optimizer()
      : _cost_func(nullptr), _gradient_func(nullptr), _cache(), _u_min_matrix(),
        _u_max_matrix(), _max_iteration(PANOC_Constants::MAX_ITERATION_DEFAULT),
        _max_lipschitz_update_iteration(
            PANOC_Constants::MAX_LIPSCHITZ_UPDATE_ITERATIONS_DEFAULT),
        _solver_status() {}

  /* Copy constructor */
  PANOC_Optimizer(const PANOC_Optimizer &input)
      : _cost_func(input._cost_func), _gradient_func(input._gradient_func),
        _cache(input._cache), _u_min_matrix(input._u_min_matrix),
        _u_max_matrix(input._u_max_matrix),
        _max_iteration(input._max_iteration),
        _max_lipschitz_update_iteration(input._max_lipschitz_update_iteration),
        _solver_status(input._solver_status) {}

  /* Copy assignment */
  PANOC_Optimizer &operator=(const PANOC_Optimizer &input) {
    if (this != &input) {
      this->_cost_func = input._cost_func;
      this->_gradient_func = input._gradient_func;
      this->_cache = input._cache;
      this->_u_min_matrix = input._u_min_matrix;
      this->_u_max_matrix = input._u_max_matrix;
      this->_max_iteration = input._max_iteration;
      this->_max_lipschitz_update_iteration =
          input._max_lipschitz_update_iteration;
      this->_solver_status = input._solver_status;
    }
    return *this;
  }

  /* Move constructor */
  PANOC_Optimizer(PANOC_Optimizer &&input) noexcept
      : _cost_func(std::move(input._cost_func)),
        _gradient_func(std::move(input._gradient_func)),
        _cache(std::move(input._cache)),
        _u_min_matrix(std::move(input._u_min_matrix)),
        _u_max_matrix(std::move(input._u_max_matrix)),
        _max_iteration(input._max_iteration),
        _max_lipschitz_update_iteration(input._max_lipschitz_update_iteration),
        _solver_status(input._solver_status) {}

  /* Move assignment */
  PANOC_Optimizer &operator=(PANOC_Optimizer &&input) noexcept {
    if (this != &input) {
      this->_cost_func = std::move(input._cost_func);
      this->_gradient_func = std::move(input._gradient_func);
      this->_cache = std::move(input._cache);
      this->_u_min_matrix = std::move(input._u_min_matrix);
      this->_u_max_matrix = std::move(input._u_max_matrix);
      this->_max_iteration = input._max_iteration;
      this->_max_lipschitz_update_iteration =
          input._max_lipschitz_update_iteration;
      this->_solver_status = input._solver_status;
    }
    return *this;
  }

public:
  /* Setter */

  /**
   * @brief Set the cost function object.
   * @param cost_func Cost function: f(U_horizon) -> T.
   */
  inline void set_cost_function(const _Cost_Function_Type &cost_func) {
    this->_cost_func = cost_func;
  }

  /**
   * @brief Set the gradient function object.
   * @param gradient_func Gradient function: grad_f(U_horizon) -> Gradient.
   */
  inline void
  set_gradient_function(const _Gradient_Function_Type &gradient_func) {
    this->_gradient_func = gradient_func;
  }

  /**
   * @brief Set the convergence tolerance.
   * @param tolerance Positive convergence tolerance on ||gamma * FPR||.
   */
  inline void set_tolerance(const _T &tolerance) {
    this->_cache.tolerance = tolerance;
  }

  /**
   * @brief Set the maximum number of PANOC iterations.
   * @param max_iteration Maximum iterations.
   */
  inline void set_max_iteration(std::size_t max_iteration) {
    this->_max_iteration = max_iteration;
  }

  /**
   * @brief Set the maximum number of Lipschitz update iterations.
   * @param max_iter Maximum Lipschitz update iterations.
   */
  inline void set_max_lipschitz_update_iteration(std::size_t max_iter) {
    this->_max_lipschitz_update_iteration = max_iter;
  }

  /**
   * @brief Set the element-wise lower bounds for box constraints.
   * @param u_min_matrix Lower bound matrix (INPUT_SIZE x NP tiled).
   */
  inline void set_u_min_matrix(const _U_Min_Matrix_Type &u_min_matrix) {
    this->_u_min_matrix = u_min_matrix;
  }

  /**
   * @brief Set the element-wise upper bounds for box constraints.
   * @param u_max_matrix Upper bound matrix (INPUT_SIZE x NP tiled).
   */
  inline void set_u_max_matrix(const _U_Max_Matrix_Type &u_max_matrix) {
    this->_u_max_matrix = u_max_matrix;
  }

  /* Getter */

  /**
   * @brief Get the solver status from the last solve call.
   * @return Reference to the solver status.
   */
  inline auto get_solver_status(void) const -> const _SolverStatus_Type & {
    return this->_solver_status;
  }

  /**
   * @brief Get a reference to the internal cache.
   * @return Reference to the PANOC cache.
   */
  inline auto get_cache(void) -> _Cache_Type & { return this->_cache; }
  inline auto get_cache(void) const -> const _Cache_Type & {
    return this->_cache;
  }

  /* Function */

  /**
   * @brief Run PANOC starting from initial guess u.
   *
   * @param u_initial Initial guess for the control horizon (U_Horizon_Type).
   * @return Optimized control horizon.
   */
  inline auto solve(const U_Horizon_Type &u_initial) -> U_Horizon_Type {
    _Cache_Type &c = this->_cache;
    c.reset();

    /* Convert U_Horizon_Type to flat vector for internal computation */
    _FlatVector_Type u = _to_flat(u_initial);

    /* --- Initialization --- */
    c.cost_value = this->_cost_func(_from_flat(u));
    this->_estimate_local_lipschitz(u);
    c.gamma =
        static_cast<_T>(PANOC_Constants::GAMMA_L_COEFFICIENT_DEFAULT) /
        _max_val(c.lipschitz_constant,
                 static_cast<_T>(PANOC_Constants::MIN_L_ESTIMATE_DEFAULT));
    c.sigma = (static_cast<_T>(1) -
               static_cast<_T>(PANOC_Constants::GAMMA_L_COEFFICIENT_DEFAULT)) /
              (static_cast<_T>(4) * c.gamma);
    this->_gradient_step(u);
    this->_half_step();

    /* --- Main loop --- */
    std::size_t number_of_iteration = 0;
    bool converged = false;

    for (std::size_t iter = 0; iter < this->_max_iteration; ++iter) {
      /* 1. Compute FPR */
      this->_compute_fpr(u);

      /* 2. Check exit condition */
      if (c.exit_condition()) {
        converged = true;
        break;
      }

      /* 3. Update Lipschitz constant */
      this->_update_lipschitz_constant(u);

      /* 4. L-BFGS direction */
      this->_lbfgs_direction(u);

      /* 5. Line search or direct update */
      if (c.iteration == 0) {
        this->_update_no_linesearch(u);
      } else {
        this->_linesearch(u);
      }

      c.iteration += 1;
      number_of_iteration += 1;
    }

    /* --- Determine exit status --- */
    PANOC_ExitStatus exit_status;
    if (!_is_all_finite(u)) {
      exit_status = PANOC_ExitStatus::NOT_FINITE_COMPUTATION;
    } else if (converged) {
      exit_status = PANOC_ExitStatus::CONVERGED;
    } else {
      exit_status = PANOC_ExitStatus::NOT_CONVERGED_ITERATIONS;
    }

    /* Return the feasible half-step (always satisfies constraints) */
    u = c.u_half_step;

    this->_solver_status.exit_status = exit_status;
    this->_solver_status.number_of_iteration = number_of_iteration;
    this->_solver_status.norm_fixed_point_residual = c.norm_gamma_fpr;
    this->_solver_status.cost_value = c.cost_value;

    return _from_flat(u);
  }

protected:
  /* ------------------------------------------------------------------ */
  /* Internal helper: flat vector <-> U_Horizon_Type conversion         */
  /* ------------------------------------------------------------------ */

  /**
   * @brief Convert U_Horizon_Type to a flat column vector.
   */
  static inline auto _to_flat(const U_Horizon_Type &u_horizon)
      -> _FlatVector_Type {
    _FlatVector_Type flat;
    for (std::size_t k = 0; k < NP; ++k) {
      for (std::size_t i = 0; i < INPUT_SIZE; ++i) {
        flat(0, k * INPUT_SIZE + i) = u_horizon(k, i);
      }
    }
    return flat;
  }

  /**
   * @brief Convert a flat column vector back to U_Horizon_Type.
   */
  static inline auto _from_flat(const _FlatVector_Type &flat)
      -> U_Horizon_Type {
    U_Horizon_Type u_horizon;
    for (std::size_t k = 0; k < NP; ++k) {
      for (std::size_t i = 0; i < INPUT_SIZE; ++i) {
        u_horizon(k, i) = flat(0, k * INPUT_SIZE + i);
      }
    }
    return u_horizon;
  }

  /**
   * @brief Project x onto the box [u_min, u_max] in-place (flat vector).
   */
  inline void _project(_FlatVector_Type &x) const {
    for (std::size_t k = 0; k < NP; ++k) {
      for (std::size_t i = 0; i < INPUT_SIZE; ++i) {
        std::size_t idx = k * INPUT_SIZE + i;
        _T val = x(0, idx);
        _T lo = this->_u_min_matrix(k, i);
        _T hi = this->_u_max_matrix(k, i);
        if (val < lo) {
          val = lo;
        }
        if (val > hi) {
          val = hi;
        }
        x(0, idx) = val;
      }
    }
  }

  /**
   * @brief Estimate the local Lipschitz constant of the gradient at u.
   *
   * Also fills cache.gradient_u with grad f(u).
   * The estimate is: L = ||grad(u+h) - grad(u)|| / ||h||
   * where h_i = max(delta, epsilon * |u_i|).
   */
  inline void _estimate_local_lipschitz(_FlatVector_Type &u) {
    _Cache_Type &c = this->_cache;
    const _T delta = static_cast<_T>(PANOC_Constants::DELTA_LIPSCHITZ_DEFAULT);
    const _T epsilon =
        static_cast<_T>(PANOC_Constants::EPSILON_LIPSCHITZ_DEFAULT);

    /* Evaluate gradient at u */
    _Gradient_Type grad_horizon = this->_gradient_func(_from_flat(u));
    c.gradient_u = _to_flat(grad_horizon);

    /* Build perturbation h: h_i = max(delta, epsilon * |u_i|) */
    _FlatVector_Type h;
    for (std::size_t i = 0; i < PROBLEM_SIZE; ++i) {
      _T abs_ui = u(0, i);
      if (abs_ui < static_cast<_T>(0)) {
        abs_ui = -abs_ui;
      }
      _T val = epsilon * abs_ui;
      h(0, i) = (delta > val) ? delta : val;
    }
    _T norm_h = PythonNumpy::norm(h);

    /* Evaluate gradient at u + h */
    _FlatVector_Type u_plus_h = u + h;
    _Gradient_Type grad_perturbed_horizon =
        this->_gradient_func(_from_flat(u_plus_h));
    _FlatVector_Type grad_perturbed = _to_flat(grad_perturbed_horizon);

    /* L = ||grad(u+h) - grad(u)|| / ||h|| */
    _FlatVector_Type diff = grad_perturbed - c.gradient_u;
    c.lipschitz_constant = PythonNumpy::norm(diff) / norm_h;
  }

  /**
   * @brief Compute the fixed-point residual: gamma_fpr = u - u_half_step.
   */
  inline void _compute_fpr(const _FlatVector_Type &u) {
    _Cache_Type &c = this->_cache;
    c.gamma_fpr = u - c.u_half_step;
    c.norm_gamma_fpr = PythonNumpy::norm(c.gamma_fpr);
  }

  /**
   * @brief gradient_step = u - gamma * gradient_u.
   */
  inline void _gradient_step(const _FlatVector_Type &u) {
    _Cache_Type &c = this->_cache;
    c.gradient_step = u - c.gamma * c.gradient_u;
  }

  /**
   * @brief gradient_step = u_plus - gamma * gradient_u.
   */
  inline void _gradient_step_uplus(void) {
    _Cache_Type &c = this->_cache;
    c.gradient_step = c.u_plus - c.gamma * c.gradient_u;
  }

  /**
   * @brief u_half_step = project(gradient_step) onto the constraint set.
   */
  inline void _half_step(void) {
    _Cache_Type &c = this->_cache;
    c.u_half_step = c.gradient_step;
    this->_project(c.u_half_step);
  }

  /**
   * @brief Update L-BFGS buffer and compute direction = H * gamma_fpr.
   */
  inline void _lbfgs_direction(const _FlatVector_Type &u) {
    _Cache_Type &c = this->_cache;
    c.lbfgs.update_hessian(c.gamma_fpr, u);
    if (c.iteration > 0) {
      c.direction_lbfgs = c.gamma_fpr;
      c.lbfgs.apply_hessian(c.direction_lbfgs);
    }
  }

  /**
   * @brief RHS of the Lipschitz update condition.
   *
   * rhs = f(u) + eps*|f(u)| - <grad, gamma_fpr> +
   *       (L_coeff / (2*gamma)) * ||gamma_fpr||^2
   */
  inline auto _lipschitz_check_rhs(void) const -> _T {
    const _Cache_Type &c = this->_cache;
    _T inner = _inner_product_flat(c.gradient_u, c.gamma_fpr);
    _T abs_cost = c.cost_value;
    if (abs_cost < static_cast<_T>(0)) {
      abs_cost = -abs_cost;
    }

    return c.cost_value +
           static_cast<_T>(PANOC_Constants::LIPSCHITZ_UPDATE_EPSILON_DEFAULT) *
               abs_cost -
           inner +
           (static_cast<_T>(PANOC_Constants::GAMMA_L_COEFFICIENT_DEFAULT) /
            (static_cast<_T>(2) * c.gamma)) *
               c.norm_gamma_fpr * c.norm_gamma_fpr;
  }

  /**
   * @brief Update the Lipschitz constant estimate (and gamma, sigma).
   */
  inline void _update_lipschitz_constant(_FlatVector_Type &u) {
    _Cache_Type &c = this->_cache;

    _T cost_half = this->_cost_func(_from_flat(c.u_half_step));
    c.cost_value = this->_cost_func(_from_flat(u));

    for (std::size_t lip_iter = 0;
         lip_iter < this->_max_lipschitz_update_iteration; ++lip_iter) {
      if (cost_half <= this->_lipschitz_check_rhs() ||
          c.lipschitz_constant >=
              static_cast<_T>(
                  PANOC_Constants::MAX_LIPSCHITZ_CONSTANT_DEFAULT)) {
        break;
      }

      c.lbfgs.reset();
      c.lipschitz_constant *= static_cast<_T>(2);
      c.gamma /= static_cast<_T>(2);

      this->_gradient_step(u);
      this->_half_step();
      cost_half = this->_cost_func(_from_flat(c.u_half_step));
      this->_compute_fpr(u);
    }

    c.sigma = (static_cast<_T>(1) -
               static_cast<_T>(PANOC_Constants::GAMMA_L_COEFFICIENT_DEFAULT)) /
              (static_cast<_T>(4) * c.gamma);
  }

  /**
   * @brief u_plus = u - (1 - tau)*gamma_fpr - tau * direction_lbfgs.
   */
  inline void _compute_u_plus(const _FlatVector_Type &u) {
    _Cache_Type &c = this->_cache;
    _T one_minus_tau = static_cast<_T>(1) - c.tau;
    c.u_plus = u - one_minus_tau * c.gamma_fpr - c.tau * c.direction_lbfgs;
  }

  /**
   * @brief Compute the RHS of the line-search condition (FBE - sigma *
   * ||fpr||^2).
   */
  inline void _compute_rhs_ls(void) {
    _Cache_Type &c = this->_cache;
    _FlatVector_Type diff = c.gradient_step - c.u_half_step;
    _T dist_sq = _inner_product_flat(diff, diff);
    _T grad_norm_sq = _inner_product_flat(c.gradient_u, c.gradient_u);
    _T fbe = c.cost_value - static_cast<_T>(0.5) * c.gamma * grad_norm_sq +
             static_cast<_T>(0.5) * dist_sq / c.gamma;
    c.rhs_ls = fbe - c.sigma * c.norm_gamma_fpr * c.norm_gamma_fpr;
  }

  /**
   * @brief Evaluate the line-search condition.
   *
   * Returns true if lhs > rhs (line search should continue).
   * Side effects: updates u_plus, cost_value, gradient_u,
   * gradient_step, u_half_step, lhs_ls.
   */
  inline bool _line_search_condition(const _FlatVector_Type &u) {
    _Cache_Type &c = this->_cache;

    /* Candidate next iterate */
    this->_compute_u_plus(u);

    /* Evaluate cost and gradient at u_plus */
    c.cost_value = this->_cost_func(_from_flat(c.u_plus));
    _Gradient_Type grad_horizon = this->_gradient_func(_from_flat(c.u_plus));
    c.gradient_u = _to_flat(grad_horizon);

    /* Gradient step and half step at u_plus */
    this->_gradient_step_uplus();
    this->_half_step();

    /* LHS of line-search condition (FBE at u_plus) */
    _FlatVector_Type diff = c.gradient_step - c.u_half_step;
    _T dist_sq = _inner_product_flat(diff, diff);
    _T grad_norm_sq = _inner_product_flat(c.gradient_u, c.gradient_u);
    c.lhs_ls = c.cost_value - static_cast<_T>(0.5) * c.gamma * grad_norm_sq +
               static_cast<_T>(0.5) * dist_sq / c.gamma;

    return c.lhs_ls > c.rhs_ls;
  }

  /**
   * @brief First-iteration update (no line search): u <- u_half_step.
   */
  inline void _update_no_linesearch(_FlatVector_Type &u) {
    _Cache_Type &c = this->_cache;
    u = c.u_half_step;
    c.cost_value = this->_cost_func(_from_flat(u));
    _Gradient_Type grad_horizon = this->_gradient_func(_from_flat(u));
    c.gradient_u = _to_flat(grad_horizon);
    this->_gradient_step(u);
    this->_half_step();
  }

  /**
   * @brief Perform a line search on tau to select the next iterate.
   */
  inline void _linesearch(_FlatVector_Type &u) {
    _Cache_Type &c = this->_cache;
    this->_compute_rhs_ls();
    c.tau = static_cast<_T>(1);
    std::size_t num_ls = 0;

    while (this->_line_search_condition(u) &&
           num_ls < PANOC_Constants::MAX_LINESEARCH_ITERATIONS_DEFAULT) {
      c.tau /= static_cast<_T>(2);
      num_ls += 1;
    }

    if (num_ls >= PANOC_Constants::MAX_LINESEARCH_ITERATIONS_DEFAULT) {
      /* Fall back to projected gradient step */
      c.tau = static_cast<_T>(0);
      u = c.u_half_step;
    }
    /* Accept the candidate */
    u = c.u_plus;
  }

  /**
   * @brief Inner product for flat vectors.
   */
  static inline auto _inner_product_flat(const _FlatVector_Type &a,
                                         const _FlatVector_Type &b) -> _T {
    _T sum = static_cast<_T>(0);
    for (std::size_t i = 0; i < PROBLEM_SIZE; ++i) {
      sum += a(0, i) * b(0, i);
    }
    return sum;
  }

  /**
   * @brief Check if all elements of the flat vector are finite.
   */
  static inline bool _is_all_finite(const _FlatVector_Type &v) {
    for (std::size_t i = 0; i < PROBLEM_SIZE; ++i) {
      if (!std::isfinite(static_cast<double>(v(0, i)))) {
        return false;
      }
    }
    return true;
  }

  /**
   * @brief Return the maximum of two values.
   */
  static inline auto _max_val(const _T &a, const _T &b) -> _T {
    return (a > b) ? a : b;
  }

protected:
  /* Variables */
  _Cost_Function_Type _cost_func;
  _Gradient_Function_Type _gradient_func;
  _Cache_Type _cache;

  _U_Min_Matrix_Type _u_min_matrix;
  _U_Max_Matrix_Type _u_max_matrix;

  std::size_t _max_iteration;
  std::size_t _max_lipschitz_update_iteration;

  _SolverStatus_Type _solver_status;
};

/**
 * @brief Factory function to create a default PANOC_Optimizer instance.
 *
 * @tparam CostMatrices_Type The cost matrices type.
 * @tparam LBFGSMemory L-BFGS memory size.
 * @return A new default-initialized PANOC_Optimizer.
 */
template <typename CostMatrices_Type, std::size_t LBFGSMemory = 5>
inline auto make_PANOC_Optimizer(void)
    -> PANOC_Optimizer<CostMatrices_Type, LBFGSMemory> {
  return PANOC_Optimizer<CostMatrices_Type, LBFGSMemory>();
}

/**
 * @brief Alias template for PANOC_Optimizer with specified cost matrices type.
 *
 * @tparam CostMatrices_Type The cost matrices type.
 * @tparam LBFGSMemory L-BFGS memory size.
 */
template <typename CostMatrices_Type, std::size_t LBFGSMemory = 5>
using PANOC_Optimizer_Type = PANOC_Optimizer<CostMatrices_Type, LBFGSMemory>;

} // namespace PythonOptimization

#endif // __PYTHON_OPTIMIZATION_PANOC_HPP__
