#include <type_traits>
#include <iostream>
#include <cmath>

#include "python_optimization.hpp"

#include "MCAP_tester.hpp"

using namespace Tester;


template <typename T>
void test_active_set() {
    using namespace PythonNumpy;
    using namespace PythonOptimization;

    MCAPTester<T> tester;

    constexpr T NEAR_LIMIT_STRICT = std::is_same<T, double>::value ? T(1.0e-5) : T(1.0e-5);
    //const T NEAR_LIMIT_SOFT = 1.0e-2F;


    /* 定義 */
    constexpr std::size_t NUMBER_OF_CONSTRAINTS = 3;

    ActiveSet<NUMBER_OF_CONSTRAINTS> active_set;

    active_set.push_active(1);

    ActiveSet_Type<NUMBER_OF_CONSTRAINTS> active_set_copy = active_set;
    ActiveSet_Type<NUMBER_OF_CONSTRAINTS> active_set_move = std::move(active_set_copy);
    active_set = std::move(active_set_move);

    tester.expect_near(static_cast<T>(active_set.get_active(0)), static_cast<T>(1),
        NEAR_LIMIT_STRICT,
        "check get active.");

    tester.expect_near(static_cast<T>(active_set.is_active(1)), static_cast<T>(true),
        NEAR_LIMIT_STRICT,
        "check is active.");

    tester.expect_near(static_cast<T>(active_set.get_number_of_active()), static_cast<T>(1),
        NEAR_LIMIT_STRICT,
        "check get number of active.");

    /* active 動作確認 */
    active_set.push_active(2);

    tester.expect_near(static_cast<T>(active_set.get_active(1)), static_cast<T>(2),
        NEAR_LIMIT_STRICT,
        "check get active after push 2.");

    tester.expect_near(static_cast<T>(active_set.is_active(2)), static_cast<T>(true),
        NEAR_LIMIT_STRICT,
        "check is active after push 2.");

    tester.expect_near(static_cast<T>(active_set.get_number_of_active()), static_cast<T>(2),
        NEAR_LIMIT_STRICT,
        "check get number of active after push 2.");

    /* inactive 動作確認 */
    active_set.push_inactive(1);

    tester.expect_near(static_cast<T>(active_set.get_active(0)), static_cast<T>(2),
        NEAR_LIMIT_STRICT, 
        "check get active after push inactive 1.");

    tester.expect_near(static_cast<T>(active_set.is_active(1)), static_cast<T>(false),
        NEAR_LIMIT_STRICT,
        "check is active after push inactive 1.");

    tester.expect_near(static_cast<T>(active_set.get_number_of_active()), static_cast<T>(1),
        NEAR_LIMIT_STRICT,
        "check get number of active after push inactive 1.");


    tester.throw_error_if_test_failed();
}


template <typename T>
void test_active_set_2D() {
    using namespace PythonNumpy;
    using namespace PythonOptimization;

    MCAPTester<T> tester;

    constexpr T NEAR_LIMIT_STRICT = std::is_same<T, double>::value ? T(1.0e-5) : T(1.0e-5);
    //const T NEAR_LIMIT_SOFT = 1.0e-2F;

    /* 定義 */
    constexpr std::size_t COLUMNS = 3;
    constexpr std::size_t ROWS = 2;

    ActiveSet2D<COLUMNS, ROWS> active_set;
    active_set.push_active(1, 0);

    ActiveSet2D_Type<COLUMNS, ROWS> active_set_copy = active_set;
    ActiveSet2D_Type<COLUMNS, ROWS> active_set_move = std::move(active_set_copy);
    active_set = std::move(active_set_move);

    auto active_pair = active_set.get_active(0);

    tester.expect_near(static_cast<T>(active_pair[0]), static_cast<T>(1), NEAR_LIMIT_STRICT,
        "check get active col.");
    tester.expect_near(static_cast<T>(active_pair[1]), static_cast<T>(0), NEAR_LIMIT_STRICT,
        "check get active row.");

    /* active 動作確認 */
    active_set.push_active(2, 1);
    active_pair = active_set.get_active(1);

    tester.expect_near(static_cast<T>(active_pair[0]), static_cast<T>(2), NEAR_LIMIT_STRICT,
        "check get active col after push 2,1.");
    tester.expect_near(static_cast<T>(active_pair[1]), static_cast<T>(1), NEAR_LIMIT_STRICT,
        "check get active row after push 2,1.");

    /* inactive 動作確認 */
    active_set.push_inactive(1, 0);
    active_pair = active_set.get_active(0);

    tester.expect_near(static_cast<T>(active_pair[0]), static_cast<T>(2), NEAR_LIMIT_STRICT,
        "check get active col after push inactive 1,0.");
    tester.expect_near(static_cast<T>(active_pair[1]), static_cast<T>(1), NEAR_LIMIT_STRICT,
        "check get active row after push inactive 1,0.");

    /*  clear 動作確認 */
    active_set.push_active(1, 1);
    active_set.clear();

    tester.expect_near(static_cast<T>(active_set.get_number_of_active()), static_cast<T>(0),
        NEAR_LIMIT_STRICT,
        "check get number of active after clear.");

    active_pair = active_set.get_active(0);

    tester.expect_near(static_cast<T>(active_pair[0]), static_cast<T>(0), NEAR_LIMIT_STRICT,
        "check get active col after clear.");
    tester.expect_near(static_cast<T>(active_pair[1]), static_cast<T>(0), NEAR_LIMIT_STRICT,
        "check get active row after clear.");

    /* ActiveSet2D MatrixOperator */
    auto A = PythonNumpy::make_DenseMatrix<COLUMNS, ROWS>(
        static_cast<T>(1), static_cast<T>(2),
        static_cast<T>(3), static_cast<T>(4),
        static_cast<T>(5), static_cast<T>(6)
    );

    auto B = PythonNumpy::make_DenseMatrix<COLUMNS, ROWS>(
        static_cast<T>(7), static_cast<T>(8),
        static_cast<T>(9), static_cast<T>(10),
        static_cast<T>(11), static_cast<T>(12)
    );

    active_set.push_active(2, 1);
    active_set.push_active(1, 1);

    auto C = ActiveSet2D_MatrixOperator::element_wise_product(A, B, active_set);

    auto C_answer = PythonNumpy::make_DenseMatrix<COLUMNS, ROWS>(
        static_cast<T>(0), static_cast<T>(0),
        static_cast<T>(0), static_cast<T>(40),
        static_cast<T>(0), static_cast<T>(72)
    );

    tester.expect_near(C.matrix.data, C_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check element wise product.");



    tester.throw_error_if_test_failed();
}


template <typename T>
void test_qp_active_set_solver() {
    using namespace PythonNumpy;
    using namespace PythonOptimization;

    MCAPTester<T> tester;

    constexpr T NEAR_LIMIT_STRICT = std::is_same<T, double>::value ? T(1.0e-5) : T(1.0e-4);
    //const T NEAR_LIMIT_SOFT = 1.0e-2F;


    /* 定義 */
    constexpr std::size_t NUMBER_OF_VARIABLES = 2;
    constexpr std::size_t NUMBER_OF_CONSTRAINTS = 4;

    QP_ActiveSetSolver_Type<T, NUMBER_OF_VARIABLES, NUMBER_OF_CONSTRAINTS> qp_solver;

    qp_solver.max_iteration = 10;
    qp_solver.tol = static_cast<T>(1.0e-8);

    QP_ActiveSetSolver_Type<T, NUMBER_OF_VARIABLES, NUMBER_OF_CONSTRAINTS> qp_solver_copy = qp_solver;
    QP_ActiveSetSolver_Type<T, NUMBER_OF_VARIABLES, NUMBER_OF_CONSTRAINTS> qp_solver_move = std::move(qp_solver_copy);
    qp_solver = std::move(qp_solver_move);

    tester.expect_near(static_cast<T>(qp_solver.max_iteration), static_cast<T>(10),
        NEAR_LIMIT_STRICT,
        "check max iteration.");
    tester.expect_near(static_cast<T>(qp_solver.tol), static_cast<T>(1.0e-8),
        NEAR_LIMIT_STRICT,
        "check tol.");

    /* demo 1 */
    auto E_1 = PythonNumpy::make_DiagMatrixIdentity<T, NUMBER_OF_VARIABLES>();
    auto L_1 = PythonNumpy::make_DenseMatrix<NUMBER_OF_VARIABLES, 1>(
        static_cast<T>(5),
        static_cast<T>(8)
    );

    auto M_1 = PythonNumpy::make_DenseMatrix<NUMBER_OF_CONSTRAINTS, NUMBER_OF_VARIABLES>(
        static_cast<T>(1), static_cast<T>(0),
        static_cast<T>(0), static_cast<T>(1),
        static_cast<T>(-1), static_cast<T>(0),
        static_cast<T>(0), static_cast<T>(-1)
    );

    auto Gamma_1 = PythonNumpy::make_DenseMatrix<NUMBER_OF_CONSTRAINTS, 1>(
        static_cast<T>(4),
        static_cast<T>(6),
        static_cast<T>(0),
        static_cast<T>(0)
    );

    auto x_opt = qp_solver.solve(E_1, L_1, M_1, Gamma_1);

    auto x_opt_answer = PythonNumpy::make_DenseMatrix<NUMBER_OF_VARIABLES, 1>(
        static_cast<T>(4),
        static_cast<T>(6)
    );

    tester.expect_near(x_opt.matrix.data, x_opt_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check x_opt, demo 1.");

    /* demo 2 */
    constexpr std::size_t NUMBER_OF_VARIABLES_2 = 3;
    constexpr std::size_t NUMBER_OF_CONSTRAINTS_2 = 3;

    QP_ActiveSetSolver_Type<T, NUMBER_OF_VARIABLES_2, NUMBER_OF_CONSTRAINTS_2> qp_solver_2;

    qp_solver_2.set_max_iteration(10);
    qp_solver_2.set_tol(static_cast<T>(1.0e-8));
    qp_solver_2.set_kkt_inv_solver_division_min(static_cast<T>(1.0e-5));

    /* 空代入チェック */
    auto E_empty = PythonNumpy::make_SparseMatrixEmpty<
        T, NUMBER_OF_VARIABLES_2, NUMBER_OF_VARIABLES_2>();
    auto L_empty = PythonNumpy::make_SparseMatrixEmpty<T, NUMBER_OF_VARIABLES_2, 1>();

    auto E_2 = PythonNumpy::make_DiagMatrixIdentity<T, NUMBER_OF_VARIABLES_2>();
    auto L_2 = PythonNumpy::make_DenseMatrix<NUMBER_OF_VARIABLES_2, 1>(
        static_cast<T>(2),
        static_cast<T>(3),
        static_cast<T>(1)
    );

    qp_solver_2.update_E(E_2);
    qp_solver_2.update_L(L_2);

    qp_solver_2.update_E(E_empty);
    qp_solver_2.update_L(L_empty);


    auto M_2 = PythonNumpy::make_DenseMatrix<NUMBER_OF_CONSTRAINTS_2, NUMBER_OF_VARIABLES_2>(
        static_cast<T>(1), static_cast<T>(1), static_cast<T>(1),
        static_cast<T>(3), static_cast<T>(-2), static_cast<T>(-3),
        static_cast<T>(1), static_cast<T>(-3), static_cast<T>(2)
    );

    auto Gamma_2 = PythonNumpy::make_DenseMatrix<NUMBER_OF_CONSTRAINTS_2, 1>(
        static_cast<T>(1),
        static_cast<T>(1),
        static_cast<T>(1)
    );

    auto x_opt_2 = qp_solver_2.solve(E_2, L_2, M_2, Gamma_2);

    auto x_opt_answer_2 = PythonNumpy::make_DenseMatrix<NUMBER_OF_VARIABLES_2, 1>(
        static_cast<T>(0.33333333),
        static_cast<T>(1.33333333),
        static_cast<T>(-0.66666667)
    );

    tester.expect_near(x_opt_2.matrix.data, x_opt_answer_2.matrix.data, NEAR_LIMIT_STRICT,
        "check x_opt, demo 2.");


    tester.throw_error_if_test_failed();
}

int main() {

    test_active_set<double>();

    test_active_set<float>();

    test_active_set_2D<double>();

    test_active_set_2D<float>();

    test_qp_active_set_solver<double>();

    test_qp_active_set_solver<float>();


    return 0;
}
