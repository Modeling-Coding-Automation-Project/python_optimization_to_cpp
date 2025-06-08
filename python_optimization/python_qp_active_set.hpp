#ifndef __PYTHON_QP_ACTIVE_SET_HPP__
#define __PYTHON_QP_ACTIVE_SET_HPP__

#include "python_numpy.hpp"

#include <array>
#include <functional>
#include <type_traits>

namespace PythonOptimization {

constexpr std::size_t MAX_ITERATION_DEFAULT = 100;
constexpr double TOL_DEFAULT = 1.0e-8;

/**
 * @class ActiveSet
 * @brief A class that manages an active set of constraints with a
 * fixed number of constraints, ensuring safe management of the active set
 * information.
 *
 * @details
 * Attributes
 * ----------
 * - active_flags : std::array<bool, NumberOfConstraints>
 *     An array indicating whether each constraint is active (length: number of
 * constraints).
 * - active_indices : std::array<std::size_t, NumberOfConstraints>
 *     An array storing the indices of active constraints
 *     (length: number of constraints, unused parts are set to 0, etc.).
 * - number_of_active : std::size_t
 *     The current number of active constraints.
 */
template <std::size_t NumberOfConstraints> class ActiveSet {
protected:
  /* Type */
  using _Active_Flags_Type = std::array<bool, NumberOfConstraints>;
  using _Active_Indices_Type = std::array<std::size_t, NumberOfConstraints>;

public:
  /* Constructor */
  ActiveSet()
      : _active_flags{}, _active_indices{},
        _number_of_active(static_cast<std::size_t>(0)) {}

  /* Copy Constructor */
  ActiveSet(const ActiveSet<NumberOfConstraints> &input)
      : _active_flags(input._active_flags),
        _active_indices(input._active_indices),
        _number_of_active(input._number_of_active) {}

  ActiveSet<NumberOfConstraints> &
  operator=(const ActiveSet<NumberOfConstraints> &input) {
    if (this != &input) {
      this->_active_flags = input._active_flags;
      this->_active_indices = input._active_indices;
      this->_number_of_active = input._number_of_active;
    }
    return *this;
  }

  /* Move Constructor */
  ActiveSet(ActiveSet<NumberOfConstraints> &&input) noexcept
      : _active_flags(std::move(input._active_flags)),
        _active_indices(std::move(input._active_indices)),
        _number_of_active(input._number_of_active) {}

  ActiveSet<NumberOfConstraints> &
  operator=(ActiveSet<NumberOfConstraints> &&input) noexcept {
    if (this != &input) {
      this->_active_flags = std::move(input._active_flags);
      this->_active_indices = std::move(input._active_indices);
      this->_number_of_active = input._number_of_active;
    }
    return *this;
  }

public:
  /* Function */
  inline void push_active(std::size_t index) {
    if (!this->_active_flags[index]) {
      this->_active_flags[index] = true;
      this->_active_indices[this->_number_of_active] = index;

      this->_number_of_active++;
    }
  }

  inline void push_inactive(std::size_t index) {
    if (this->_active_flags[index]) {
      this->_active_flags[index] = false;
      bool found = false;

      for (std::size_t i = 0; i < this->_number_of_active; ++i) {
        if (!found && this->_active_indices[i] == index) {
          found = true;
        }
        if (found && i < this->_number_of_active - 1) {
          this->_active_indices[i] = this->_active_indices[i + 1];
        }
      }
      if (found) {
        this->_active_indices[this->_number_of_active - 1] = 0;
        this->_number_of_active--;
      }
    }
  }

  inline auto get_active(std::size_t index) const -> std::size_t {
    if (index >= this->_number_of_active) {
      index = this->_number_of_active - 1;
    }
    return this->_active_indices[index];
  }

  inline auto get_active_indices() const -> _Active_Indices_Type & {
    return this->_active_indices;
  }

  inline auto get_number_of_active() const -> std::size_t {
    return this->_number_of_active;
  }

  inline auto is_active(std::size_t index) const -> bool {
    return this->_active_flags[index];
  }

public:
  /* Constant */
  static constexpr std::size_t NUMBER_OF_CONSTRAINTS = NumberOfConstraints;

protected:
  /* variables */
  _Active_Flags_Type _active_flags;
  _Active_Indices_Type _active_indices;
  std::size_t _number_of_active;
};

/* make Active Set */
template <std::size_t NumberOfConstraints>
inline auto make_ActiveSet(void) -> ActiveSet<NumberOfConstraints> {
  return ActiveSet<NumberOfConstraints>();
}

/* Active Set Type */
template <std::size_t NumberOfConstraints>
using ActiveSet_Type = ActiveSet<NumberOfConstraints>;

namespace QP_ActiveSetSolverOperation {

/* Set KKT Column */
template <std::size_t M, std::size_t J> struct SetKKTColumn {
  template <typename KKT_Type, typename ActiveSet_Type, typename M_Type>
  static void set(KKT_Type &KKT, const ActiveSet_Type &active_set,
                  M_Type &M_matrix, const std::size_t &i) {

    KKT.access(J, M + i) = M_matrix.access(active_set.get_active(i), J);
    KKT.access(M + i, J) = M_matrix.access(active_set.get_active(i), J);

    SetKKTColumn<M, (J - 1)>::set(KKT, active_set, M_matrix, i);
  }
};

template <std::size_t M> struct SetKKTColumn<M, 0> {
  template <typename KKT_Type, typename ActiveSet_Type, typename M_Type>
  static void set(KKT_Type &KKT, const ActiveSet_Type &active_set,
                  M_Type &M_matrix, const std::size_t &i) {

    KKT.access(0, M + i) = M_matrix.access(active_set.get_active(i), 0);
    KKT.access(M + i, 0) = M_matrix.access(active_set.get_active(i), 0);
  }
};

/* X from Sol */
template <std::size_t M> struct X_From_Sol {
  template <typename Sol_Type, typename X_Type>
  static void apply(X_Type &X, const Sol_Type &sol) {

    X.template set<M, 0>(sol.template get<M, 0>());

    X_From_Sol<M - 1>::apply(X, sol);
  }
};

template <> struct X_From_Sol<0> {
  template <typename Sol_Type, typename X_Type>
  static void apply(X_Type &X, const Sol_Type &sol) {

    X.template set<0, 0>(sol.template get<0, 0>());
  }
};

/* Sol from Sol */
template <std::size_t Number_Of_Variables, std::size_t M>
struct Lambda_From_Sol {
  template <typename Sol_Type, typename Lambda_Type>
  static void apply(Lambda_Type &Lambda, const Sol_Type &sol) {

    Lambda.template set<M, 0>(sol.template get<(Number_Of_Variables + M), 0>());

    Lambda_From_Sol<Number_Of_Variables, (M - 1)>::apply(Lambda, sol);
  }
};

template <std::size_t Number_Of_Variables>
struct Lambda_From_Sol<Number_Of_Variables, 0> {
  template <typename Sol_Type, typename Lambda_Type>
  static void apply(Lambda_Type &Lambda, const Sol_Type &sol) {

    Lambda.template set<0, 0>(sol.template get<Number_Of_Variables, 0>());
  }
};

/* Check gamma violation */
template <typename T, std::size_t J> struct CheckGammaViolation {
  template <typename Gamma_Type, typename M_X_Type>
  static void check(const Gamma_Type &gamma, const M_X_Type &M_X,
                    std::size_t &violation_index, bool &is_violated,
                    T &max_violation, const T &tol) {

    T gamma_tol = gamma.template get<J, 0>() + tol;
    if (M_X.template get<J, 0>() > gamma_tol) {

      T M_X_gamma = M_X.template get<J, 0>() - gamma.template get<J, 0>();
      if (M_X_gamma > max_violation) {
        max_violation = M_X_gamma;
        violation_index = J;
        is_violated = true;
      }
    }

    CheckGammaViolation<T, (J - 1)>::check(gamma, M_X, violation_index,
                                           is_violated, max_violation, tol);
  }
};

template <typename T> struct CheckGammaViolation<T, 0> {
  template <typename Gamma_Type, typename M_X_Type>
  static void check(const Gamma_Type &gamma, const M_X_Type &M_X,
                    std::size_t &violation_index, bool &is_violated,
                    T &max_violation, const T &tol) {

    T gamma_tol = gamma.template get<0, 0>() + tol;
    if (M_X.template get<0, 0>() > gamma_tol) {

      T M_X_gamma = M_X.template get<0, 0>() - gamma.template get<0, 0>();
      if (M_X_gamma > max_violation) {
        max_violation = M_X_gamma;
        violation_index = 0;
        is_violated = true;
      }
    }
  }
};

/* Check negative lambda  */
template <typename T, std::size_t J> struct CheckNegativeLambda {
  template <typename Lambda_Type>
  static void check(const Lambda_Type &Lambda_candidate, const T &tol,
                    std::size_t &min_lambda_index, bool &negative_lambda_found,
                    T &min_lambda_value) {
    T lambda_val = Lambda_candidate(J, 0);
    if ((lambda_val < -tol) && (lambda_val < min_lambda_value)) {
      min_lambda_value = lambda_val;
      min_lambda_index = J;
      negative_lambda_found = true;
    }
    CheckNegativeLambda<T, J - 1>::check(
        Lambda_candidate, tol, min_lambda_index, negative_lambda_found,
        min_lambda_value);
  }
};

template <typename T> struct CheckNegativeLambda<T, 0> {
  template <typename Lambda_Type>
  static void check(const Lambda_Type &Lambda_candidate, const T &tol,
                    std::size_t &min_lambda_index, bool &negative_lambda_found,
                    T &min_lambda_value) {
    T lambda_val = Lambda_candidate(0, 0);
    if ((lambda_val < -tol) && (lambda_val < min_lambda_value)) {
      min_lambda_value = lambda_val;
      min_lambda_index = 0;
      negative_lambda_found = true;
    }
  }
};

template <std::size_t J> struct DeactivateAllActiveConstraints {
  template <typename ActiveSetType>
  static void apply(ActiveSetType &active_set) {
    if (active_set.is_active(J)) {
      active_set.push_inactive(J);
    }
    DeactivateAllActiveConstraints<J - 1>::apply(active_set);
  }
};

template <> struct DeactivateAllActiveConstraints<0> {
  template <typename ActiveSetType>
  static void apply(ActiveSetType &active_set) {
    if (active_set.is_active(0)) {
      active_set.push_inactive(0);
    }
  }
};

} // namespace QP_ActiveSetSolverOperation

/**
 * @class QPActiveSetSolver
 * @brief Quadratic Programming (QP) solver using the Active Set method.
 *
 * @details
 * Problem: minimize (1/2) X^T E X - X^T L  subject to  M X <= gamma.
 *
 * Parameters
 * ----------
 * - E, L, M, gamma : Parameters of the above QP (typically as arrays or
 * matrices)
 * - max_iteration  : Maximum number of iterations (limit)
 * - tol            : Tolerance for numerical errors (used for constraint
 * violation and negative lambda checks)
 * - X              : Solution vector estimated as optimal
 * - active_set     : List of indices of constraints that were active at the end
 */
template <typename T, std::size_t Number_Of_Variables,
          std::size_t Number_Of_Constraints>
class QP_ActiveSetSolver {
public:
  /* Type */
  using Value_Type = T;

  using X_Type = PythonNumpy::DenseMatrix_Type<T, Number_Of_Variables, 1>;

  using Lambda_Type =
      PythonNumpy::DenseMatrix_Type<T, Number_Of_Constraints, 1>;

  using Sol_Type = PythonNumpy::DenseMatrix_Type<
      T, (Number_Of_Variables + Number_Of_Constraints), 1>;

  using KKT_Type = PythonNumpy::DenseMatrix_Type<
      T, (Number_Of_Variables + Number_Of_Constraints),
      (Number_Of_Variables + Number_Of_Constraints)>;

  using RHS_Type = PythonNumpy::DenseMatrix_Type<
      T, (Number_Of_Variables + Number_Of_Constraints), 1>;

  using KKT_Inv_Solver_Type =
      PythonNumpy::LinalgPartitionSolver_Type<KKT_Type, RHS_Type>;

protected:
  /* Type */
  using _T = T;

public:
  /* Constant */
  static constexpr std::size_t NUMBER_OF_VARIABLES = Number_Of_Variables;
  static constexpr std::size_t NUMBER_OF_CONSTRAINTS = Number_Of_Constraints;

public:
  /* Constructor */
  QP_ActiveSetSolver()
      : X(), active_set(make_ActiveSet<Number_Of_Constraints>()),
        max_iteration(PythonOptimization::MAX_ITERATION_DEFAULT),
        tol(static_cast<T>(PythonOptimization::TOL_DEFAULT)), _KKT(), _RHS(),
        _kkt_inv_solver(), _iteration_count(static_cast<std::size_t>(0)) {}

  QP_ActiveSetSolver(const X_Type &X_in)
      : X(X_in), active_set(make_ActiveSet<Number_Of_Constraints>()),
        max_iteration(PythonOptimization::MAX_ITERATION_DEFAULT),
        tol(static_cast<T>(PythonOptimization::TOL_DEFAULT)), _KKT(), _RHS(),
        _kkt_inv_solver(), _iteration_count(static_cast<std::size_t>(0)) {}

  QP_ActiveSetSolver(const X_Type &X_in,
                     const ActiveSet_Type<Number_Of_Constraints> &active_set_in)
      : X(X_in), active_set(active_set_in),
        max_iteration(PythonOptimization::MAX_ITERATION_DEFAULT),
        tol(static_cast<T>(PythonOptimization::TOL_DEFAULT)), _KKT(), _RHS(),
        _kkt_inv_solver(), _iteration_count(static_cast<std::size_t>(0)) {}

  /* Copy Constructor */
  QP_ActiveSetSolver(const QP_ActiveSetSolver<T, Number_Of_Variables,
                                              Number_Of_Constraints> &input)
      : X(input.X), active_set(input.active_set),
        max_iteration(input.max_iteration), tol(input.tol), _KKT(input._KKT),
        _RHS(input._RHS), _kkt_inv_solver(input._kkt_inv_solver),
        _iteration_count(input._iteration_count) {}

  QP_ActiveSetSolver<T, Number_Of_Variables, Number_Of_Constraints> &operator=(
      const QP_ActiveSetSolver<T, Number_Of_Variables, Number_Of_Constraints>
          &input) {
    if (this != &input) {
      this->X = input.X;
      this->active_set = input.active_set;
      this->max_iteration = input.max_iteration;
      this->tol = input.tol;
      this->_KKT = input._KKT;
      this->_RHS = input._RHS;
      this->_kkt_inv_solver = input._kkt_inv_solver;
      this->_iteration_count = input._iteration_count;
    }
    return *this;
  }

  /* Move Constructor */
  QP_ActiveSetSolver(QP_ActiveSetSolver<T, Number_Of_Variables,
                                        Number_Of_Constraints> &&input) noexcept
      : X(std::move(input.X)), active_set(std::move(input.active_set)),
        max_iteration(input.max_iteration), tol(input.tol),
        _KKT(std::move(input._KKT)), _RHS(std::move(input._RHS)),
        _kkt_inv_solver(std::move(input._kkt_inv_solver)),
        _iteration_count(input._iteration_count) {}

  QP_ActiveSetSolver<T, Number_Of_Variables, Number_Of_Constraints> &
  operator=(QP_ActiveSetSolver<T, Number_Of_Variables, Number_Of_Constraints>
                &&input) noexcept {
    if (this != &input) {
      this->X = std::move(input.X);
      this->active_set = std::move(input.active_set);
      this->max_iteration = input.max_iteration;
      this->tol = input.tol;
      this->_KKT = std::move(input._KKT);
      this->_RHS = std::move(input._RHS);
      this->_kkt_inv_solver = std::move(input._kkt_inv_solver);
      this->_iteration_count = input._iteration_count;
    }
    return *this;
  }

public:
  /* Function */
  template <typename E_Type> inline void update_E(E_Type E) {
    PythonNumpy::substitute_part_matrix<0, 0>(this->_KKT, E);
  }

  template <typename L_Type> inline void update_L(L_Type L) {
    PythonNumpy::substitute_part_matrix<0, 0>(this->_RHS, L);
  }

  template <typename E_Type, typename L_Type>
  inline auto solve_no_constrained_X(const E_Type &E, const L_Type &L)
      -> X_Type {

    X_Type X_out;

    this->update_E(E);
    this->update_L(L);

    auto sol = this->_kkt_inv_solver.solve(this->_KKT, this->_RHS,
                                           NUMBER_OF_VARIABLES);
    QP_ActiveSetSolverOperation::X_From_Sol<(NUMBER_OF_VARIABLES - 1)>::apply(
        X_out, sol);

    return X_out;
  }

  template <typename E_Type, typename L_Type, typename M_Type,
            typename Gamma_Type>
  inline void initialize_X(const E_Type &E, const L_Type &L, const M_Type &M,
                           const Gamma_Type &gamma) {

    if (static_cast<std::size_t>(0) ==
        this->active_set.get_number_of_active()) {
      this->X = this->solve_no_constrained_X(E, L);
    } else {
      this->_set_KKT(E, M);
      this->_set_rhs(L, gamma);

      auto sol = this->_solve_KKT_inv(this->active_set.get_number_of_active());
      QP_ActiveSetSolverOperation::X_From_Sol<(NUMBER_OF_VARIABLES - 1)>::apply(
          this->X, sol);
    }
  }

  template <typename E_Type, typename L_Type, typename M_Type,
            typename Gamma_Type>
  inline auto solve(const E_Type &E, const L_Type &L, const M_Type &M,
                    const Gamma_Type &gamma) -> X_Type {
    /* Check Compatibility */
    static_assert(E_Type::COLS == NUMBER_OF_VARIABLES &&
                      E_Type::ROWS == NUMBER_OF_VARIABLES,
                  "E must be a square matrix of size (n, n) where n is the "
                  "number of variables.");

    static_assert(L_Type::COLS == NUMBER_OF_VARIABLES && L_Type::ROWS == 1,
                  "L must be a column vector of size (n, 1) where n is the "
                  "number of variables.");

    static_assert(M_Type::COLS == NUMBER_OF_CONSTRAINTS &&
                      M_Type::ROWS == NUMBER_OF_VARIABLES,
                  "M must be a matrix of size (m, n) where m is the number of "
                  "constraints and n is the number of variables.");

    static_assert(Gamma_Type::COLS == NUMBER_OF_CONSTRAINTS &&
                      Gamma_Type::ROWS == 1,
                  "gamma must be a column vector of size (m, 1) where m is the "
                  "number of constraints.");

    /* Main iterative loop */
    X_Type X_candidate;
    Lambda_Type Lambda_candidate;
    bool lambda_candidate_exists = false;

    std::size_t iteration_count = 0;
    for (iteration_count = 0; iteration_count < this->max_iteration;
         ++iteration_count) {

      if (static_cast<std::size_t>(0) ==
          this->active_set.get_number_of_active()) {
        // If there are no active constraints, simply solve E X = L

        X_candidate = this->solve_no_constrained_X(E, L);
        lambda_candidate_exists = false;
      } else {

        this->_set_KKT(E, M);
        this->_set_rhs(L, gamma);

        auto sol =
            this->_solve_KKT_inv(this->active_set.get_number_of_active());
        QP_ActiveSetSolverOperation::X_From_Sol<(NUMBER_OF_VARIABLES -
                                                 1)>::apply(X_candidate, sol);
        QP_ActiveSetSolverOperation::Lambda_From_Sol<
            NUMBER_OF_VARIABLES,
            (NUMBER_OF_CONSTRAINTS - 1)>::apply(Lambda_candidate, sol);
        lambda_candidate_exists = true;
      }

      // (1) Check constraint violations for the candidate solution
      std::size_t violation_index = 0;
      bool is_violated = false;
      _T max_violation = static_cast<_T>(0);

      auto M_X = M * X_candidate;

      QP_ActiveSetSolverOperation::CheckGammaViolation<
          _T, (NUMBER_OF_CONSTRAINTS - 1)>::check(gamma, M_X, violation_index,
                                                  is_violated, max_violation,
                                                  this->tol);

      if (is_violated) {
        this->active_set.push_active(violation_index);

        // Since a constraint was added, re-optimize in the next loop
        this->X = X_candidate;
        continue;
      }

      // (2) All constraints are satisfied -> Check lambda
      if (this->active_set.get_number_of_active() > 0) {
        // Find negative lambda among the active constraints
        std::size_t min_lambda_index = 0;
        bool negative_lambda_found = false;
        _T min_lambda_value = static_cast<_T>(0);

        if (lambda_candidate_exists) {
          QP_ActiveSetSolverOperation::CheckNegativeLambda<
              _T, (NUMBER_OF_CONSTRAINTS - 1)>::check(Lambda_candidate,
                                                      this->tol,
                                                      min_lambda_index,
                                                      negative_lambda_found,
                                                      min_lambda_value);
        }

        if (negative_lambda_found) {
          this->active_set.push_inactive(min_lambda_index);
          // Since a constraint was removed, re-optimize in the next loop
          this->X = X_candidate;
          continue;
        }
      }

      // If there are no constraint violations and all lambda are non-negative,
      // consider as optimal solution.
      this->X = X_candidate;

      QP_ActiveSetSolverOperation::DeactivateAllActiveConstraints<(
          NUMBER_OF_CONSTRAINTS - 1)>::apply(this->active_set);

      break;
    }

    this->_iteration_count = iteration_count;

    return this->X;
  }

  /* Get */
  inline auto get_iteration_count() const -> std::size_t {
    return this->_iteration_count;
  }

  inline auto get_max_iteration() const -> std::size_t {
    return this->max_iteration;
  }

  inline auto get_tol() const -> T { return this->tol; }

  /* Set */
  inline void set_kkt_inv_solver_decay_rate(const T &decay_rate) {
    this->_kkt_inv_solver.set_decay_rate(decay_rate);
  }

  inline void set_kkt_inv_solver_division_min(const T &division_min) {
    this->_kkt_inv_solver.set_division_min(division_min);
  }

  inline void set_max_iteration(const std::size_t &max_iteration_in) {
    this->max_iteration = max_iteration_in;
  }

  inline void set_tol(const T &tol_in) { this->tol = tol_in; }

protected:
  /* Function */
  template <typename E_Type, typename M_Type>
  inline void _set_KKT(E_Type E, M_Type M) {

    this->update_E(E);

    for (std::size_t i = 0; i < this->active_set.get_number_of_active(); ++i) {

      QP_ActiveSetSolverOperation::SetKKTColumn<
          NUMBER_OF_VARIABLES, (NUMBER_OF_VARIABLES - 1)>::set(this->_KKT,
                                                               this->active_set,
                                                               M, i);
    }
  }

  template <typename L_Type, typename Gamma_Type>
  inline void _set_rhs(L_Type L, Gamma_Type gamma) {

    this->update_L(L);

    for (std::size_t i = 0; i < this->active_set.get_number_of_active(); ++i) {
      this->_RHS.access(NUMBER_OF_VARIABLES + i, 0) =
          gamma.access(this->active_set.get_active(i), 0);
    }
  }

  inline auto _solve_KKT_inv(const std::size_t &k) -> Sol_Type {

    return this->_kkt_inv_solver.cold_solve(this->_KKT, this->_RHS,
                                            NUMBER_OF_VARIABLES + k);
  }

public:
  /* variables */
  X_Type X;
  ActiveSet_Type<Number_Of_Constraints> active_set;

  std::size_t max_iteration;
  T tol;

protected:
  /* variables */
  KKT_Type _KKT;
  RHS_Type _RHS;
  KKT_Inv_Solver_Type _kkt_inv_solver;
  std::size_t _iteration_count;
};

/* make QP Active Set Solver */
template <typename T, std::size_t Number_Of_Variables,
          std::size_t Number_Of_Constraints>
inline auto make_QP_ActiveSetSolver(void)
    -> QP_ActiveSetSolver<T, Number_Of_Variables, Number_Of_Constraints> {
  return QP_ActiveSetSolver<T, Number_Of_Variables, Number_Of_Constraints>();
}

/* QP Active Set Solver Type */
template <typename T, std::size_t Number_Of_Variables,
          std::size_t Number_Of_Constraints>
using QP_ActiveSetSolver_Type =
    QP_ActiveSetSolver<T, Number_Of_Variables, Number_Of_Constraints>;

} // namespace PythonOptimization

#endif // __PYTHON_QP_ACTIVE_SET_HPP__
