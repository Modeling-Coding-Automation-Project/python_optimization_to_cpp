#include <type_traits>
#include <iostream>
#include <cmath>

#include "python_optimization.hpp"

#include "MCAP_tester.hpp"

#include "sqp_test_data.hpp"

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

    auto vdot_value = ActiveSet2D_MatrixOperator::vdot(A, B, active_set);

    auto vdot_value_answer = static_cast<T>(112);

    tester.expect_near(vdot_value, vdot_value_answer, NEAR_LIMIT_STRICT,
        "check vdot.");

    auto D = ActiveSet2D_MatrixOperator::matrix_multiply_scalar(A, static_cast<T>(3), active_set);

    auto D_answer = PythonNumpy::make_DenseMatrix<COLUMNS, ROWS>(
        static_cast<T>(0), static_cast<T>(0),
        static_cast<T>(0), static_cast<T>(12),
        static_cast<T>(0), static_cast<T>(18)
    );

    tester.expect_near(D.matrix.data, D_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check matrix multiply scalar.");

    auto norm_value = ActiveSet2D_MatrixOperator::norm(A, active_set);

    auto norm_value_answer = std::sqrt(static_cast<T>(52));

    tester.expect_near(norm_value, norm_value_answer, NEAR_LIMIT_STRICT,
        "check norm.");



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


template <typename T>
void test_SQP_CostMatrices_NMPC() {
    using namespace PythonNumpy;
    using namespace PythonControl;
    using namespace PythonOptimization;
    using namespace SQP_TestData;

    MCAPTester<T> tester;

    constexpr T NEAR_LIMIT_STRICT = std::is_same<T, double>::value ? T(1.0e-5) : T(1.0e-4);
    //const T NEAR_LIMIT_SOFT = 1.0e-2F;

    /* 定義 */
    constexpr std::size_t STATE_SIZE = 4;
    constexpr std::size_t INPUT_SIZE = 2;
    constexpr std::size_t OUTPUT_SIZE = 2;

    constexpr std::size_t NP = 10;

    using Parameter_Type = sqp_2_mass_spring_damper_demo_parameter::Parameter<T>;

    using State_Jacobian_X_Matrix_Type =
        sqp_2_mass_spring_damper_demo_sqp_state_jacobian_x::State_Jacobian_x_Type<T>;
    using State_Jacobian_U_Matrix_Type =
        sqp_2_mass_spring_damper_demo_sqp_state_jacobian_u::State_Jacobian_u_Type<T>;
    using Measurement_Jacobian_X_Matrix_Type =
        sqp_2_mass_spring_damper_demo_sqp_measurement_jacobian_x::Measurement_Jacobian_x_Type<T>;
    using State_Hessian_XX_Matrix_Type =
        sqp_2_mass_spring_damper_demo_sqp_hessian_f_xx::State_Hessian_xx_Type<T>;
    using State_Hessian_XU_Matrix_Type =
        sqp_2_mass_spring_damper_demo_sqp_hessian_f_xu::State_Hessian_xu_Type<T>;
    using State_Hessian_UX_Matrix_Type =
        sqp_2_mass_spring_damper_demo_sqp_hessian_f_ux::State_Hessian_ux_Type<T>;
    using State_Hessian_UU_Matrix_Type =
        sqp_2_mass_spring_damper_demo_sqp_hessian_f_uu::State_Hessian_uu_Type<T>;
    using Measurement_Hessian_XX_Matrix_Type =
        sqp_2_mass_spring_damper_demo_sqp_hessian_h_xx::Measurement_Hessian_xx_Type<T>;

    using X_Type = StateSpaceState_Type<T, STATE_SIZE>;
    using U_Type = StateSpaceInput_Type<T, INPUT_SIZE>;
    using Y_Type = StateSpaceOutput_Type<T, OUTPUT_SIZE>;

    PythonOptimization::StateFunction_Object<X_Type, U_Type, Parameter_Type> state_function =
        sqp_2_mass_spring_damper_demo_sqp_state_function::Function<T, X_Type, U_Type, Parameter_Type>::function;
    PythonOptimization::MeasurementFunction_Object<Y_Type, X_Type, U_Type, Parameter_Type> measurement_function =
        sqp_2_mass_spring_damper_demo_sqp_measurement_function::Function<
        T, X_Type, U_Type, Parameter_Type, Y_Type>::function;

    PythonOptimization::StateFunctionJacobian_X_Object<
        State_Jacobian_X_Matrix_Type, X_Type, U_Type, Parameter_Type> state_jacobian_x_function =
        sqp_2_mass_spring_damper_demo_sqp_state_jacobian_x::Function<T, X_Type, U_Type, Parameter_Type>::function;
    PythonOptimization::StateFunctionJacobian_U_Object<
        State_Jacobian_U_Matrix_Type, X_Type, U_Type, Parameter_Type> state_jacobian_u_function =
        sqp_2_mass_spring_damper_demo_sqp_state_jacobian_u::Function<T, X_Type, U_Type, Parameter_Type>::function;
    PythonOptimization::MeasurementFunctionJacobian_X_Object<
        Measurement_Jacobian_X_Matrix_Type, X_Type, U_Type, Parameter_Type> measurement_jacobian_x_function =
        sqp_2_mass_spring_damper_demo_sqp_measurement_jacobian_x::Function<T, X_Type, U_Type, Parameter_Type>::function;

    PythonOptimization::StateFunctionHessian_XX_Object<
        State_Hessian_XX_Matrix_Type, X_Type, U_Type, Parameter_Type> state_hessian_xx_function =
        sqp_2_mass_spring_damper_demo_sqp_hessian_f_xx::Function<T, X_Type, U_Type, Parameter_Type>::function;
    PythonOptimization::StateFunctionHessian_XU_Object<
        State_Hessian_XU_Matrix_Type, X_Type, U_Type, Parameter_Type> state_hessian_xu_function =
        sqp_2_mass_spring_damper_demo_sqp_hessian_f_xu::Function<T, X_Type, U_Type, Parameter_Type>::function;
    PythonOptimization::StateFunctionHessian_UX_Object<
        State_Hessian_UX_Matrix_Type, X_Type, U_Type, Parameter_Type> state_hessian_ux_function =
        sqp_2_mass_spring_damper_demo_sqp_hessian_f_ux::Function<T, X_Type, U_Type, Parameter_Type>::function;
    PythonOptimization::StateFunctionHessian_UU_Object<
        State_Hessian_UU_Matrix_Type, X_Type, U_Type, Parameter_Type> state_hessian_uu_function =
        sqp_2_mass_spring_damper_demo_sqp_hessian_f_uu::Function<T, X_Type, U_Type, Parameter_Type>::function;
    PythonOptimization::MeasurementFunctionHessian_XX_Object<
        Measurement_Hessian_XX_Matrix_Type, X_Type, U_Type, Parameter_Type> measurement_hessian_xx_function =
        sqp_2_mass_spring_damper_demo_sqp_hessian_h_xx::Function<T, X_Type, U_Type, Parameter_Type>::function;

    using Qx_Type = DiagMatrix_Type<T, STATE_SIZE>;
    using R_Type = DiagMatrix_Type<T, INPUT_SIZE>;
    using Qy_Type = DiagMatrix_Type<T, OUTPUT_SIZE>;

    Qx_Type Qx = make_DiagMatrix<STATE_SIZE>(
        static_cast<T>(0.5), static_cast<T>(0.1), static_cast<T>(0.5), static_cast<T>(0.1)
    );
    R_Type R = make_DiagMatrix<INPUT_SIZE>(
        static_cast<T>(0.1), static_cast<T>(0.1)
    );
    Qy_Type Qy = make_DiagMatrix<OUTPUT_SIZE>(
        static_cast<T>(0.5), static_cast<T>(0.5)
    );

    using U_Min_Type = StateSpaceInput_Type<T, INPUT_SIZE>;
    using U_Max_Type = StateSpaceInput_Type<T, INPUT_SIZE>;

    U_Min_Type u_min = make_DenseMatrix<INPUT_SIZE, 1>(
        static_cast<T>(-1),
        static_cast<T>(-1)
    );
    U_Max_Type u_max = make_DenseMatrix<INPUT_SIZE, 1>(
        static_cast<T>(1),
        static_cast<T>(1)
    );

    using Y_Min_Type = SparseMatrixEmpty_Type<T, OUTPUT_SIZE, 1>;
    using Y_Max_Type = SparseMatrixEmpty_Type<T, OUTPUT_SIZE, 1>;

    Y_Min_Type y_min;
    Y_Max_Type y_max;

    using Reference_Trajectory_Type = DenseMatrix_Type<T, OUTPUT_SIZE, (NP + 1)>;

    Reference_Trajectory_Type reference_trajectory;

    /* コンストラクタ */
    using Cost_Matrices_Type = SQP_CostMatrices_NMPC_Type<T, NP, Parameter_Type,
        U_Min_Type, U_Max_Type, Y_Min_Type, Y_Max_Type,
        State_Jacobian_X_Matrix_Type,
        State_Jacobian_U_Matrix_Type,
        Measurement_Jacobian_X_Matrix_Type,
        State_Hessian_XX_Matrix_Type,
        State_Hessian_XU_Matrix_Type,
        State_Hessian_UX_Matrix_Type,
        State_Hessian_UU_Matrix_Type,
        Measurement_Hessian_XX_Matrix_Type>;

    Cost_Matrices_Type cost_matrices =
        make_SQP_CostMatrices_NMPC<T, NP, Parameter_Type,
        U_Min_Type, U_Max_Type, Y_Min_Type, Y_Max_Type,
        State_Jacobian_X_Matrix_Type,
        State_Jacobian_U_Matrix_Type,
        Measurement_Jacobian_X_Matrix_Type,
        State_Hessian_XX_Matrix_Type,
        State_Hessian_XU_Matrix_Type,
        State_Hessian_UX_Matrix_Type,
        State_Hessian_UU_Matrix_Type,
        Measurement_Hessian_XX_Matrix_Type>(
            Qx, R, Qy, u_min, u_max, y_min, y_max);

    /* コピー、ムーブ */
    Cost_Matrices_Type cost_matrices_copy = cost_matrices;
    Cost_Matrices_Type cost_matrices_move = std::move(cost_matrices_copy);
    cost_matrices = std::move(cost_matrices_move);

    T u_min_value = cost_matrices.state_space_parameters.k1;

    tester.expect_near(u_min_value, static_cast<T>(10), NEAR_LIMIT_STRICT,
        "check u_min value.");

    cost_matrices.set_function_objects(
        state_function,
        measurement_function,
        state_jacobian_x_function,
        state_jacobian_u_function,
        measurement_jacobian_x_function,
        state_hessian_xx_function,
        state_hessian_xu_function,
        state_hessian_ux_function,
        state_hessian_uu_function,
        measurement_hessian_xx_function
    );

    cost_matrices.reference_trajectory = reference_trajectory;

    auto X_initial = make_DenseMatrix<STATE_SIZE, 1>(
        static_cast<T>(5),
        static_cast<T>(0),
        static_cast<T>(5),
        static_cast<T>(0)
    );

    DenseMatrix_Type<T, INPUT_SIZE, NP> U_horizon_initial;

    /* 関数 */
    U_Type U_temp = make_DenseMatrix<INPUT_SIZE, 1>(
        static_cast<T>(1),
        static_cast<T>(2)
    );

    auto l_xx_return = cost_matrices.l_xx(X_initial, U_temp);

    auto l_xx_return_answer = make_DiagMatrix<STATE_SIZE>(
        static_cast<T>(1.0), static_cast<T>(0.2), static_cast<T>(1.0), static_cast<T>(0.2)
    );

    tester.expect_near(l_xx_return.matrix.data, l_xx_return_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check l_xx.");

    auto l_uu_return = cost_matrices.l_uu(X_initial, U_temp);

    auto l_uu_return_answer = make_DiagMatrix<INPUT_SIZE>(
        static_cast<T>(0.2), static_cast<T>(0.2)
    );

    tester.expect_near(l_uu_return.matrix.data, l_uu_return_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check l_uu.");

    auto l_xu_return = cost_matrices.l_xu(X_initial, U_temp);
    auto l_xu_return_dense = l_xu_return.create_dense();

    auto l_xu_return_answer = make_DenseMatrixZeros<T, STATE_SIZE, INPUT_SIZE>();

    tester.expect_near(l_xu_return_dense.matrix.data, l_xu_return_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check l_xu.");

    auto l_ux_return = cost_matrices.l_ux(X_initial, U_temp);
    auto l_ux_return_dense = l_ux_return.create_dense();

    auto l_ux_return_answer = make_DenseMatrixZeros<T, INPUT_SIZE, STATE_SIZE>();

    tester.expect_near(l_ux_return_dense.matrix.data, l_ux_return_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check l_ux.");


    tester.throw_error_if_test_failed();
}

template <typename T>
void test_sqp_active_set_pcg_pls() {
    using namespace PythonNumpy;
    using namespace PythonControl;
    using namespace PythonOptimization;
    using namespace SQP_TestData;

    MCAPTester<T> tester;

    constexpr T NEAR_LIMIT_STRICT = std::is_same<T, double>::value ? T(1.0e-5) : T(1.0e-4);
    //const T NEAR_LIMIT_SOFT = 1.0e-2F;

    /* Cost Matrices 定義 */
    constexpr std::size_t STATE_SIZE = 4;
    constexpr std::size_t INPUT_SIZE = 2;
    constexpr std::size_t OUTPUT_SIZE = 2;

    constexpr std::size_t NP = 10;

    using Parameter_Type = sqp_2_mass_spring_damper_demo_parameter::Parameter<T>;

    using State_Jacobian_X_Matrix_Type =
        sqp_2_mass_spring_damper_demo_sqp_state_jacobian_x::State_Jacobian_x_Type<T>;
    using State_Jacobian_U_Matrix_Type =
        sqp_2_mass_spring_damper_demo_sqp_state_jacobian_u::State_Jacobian_u_Type<T>;
    using Measurement_Jacobian_X_Matrix_Type =
        sqp_2_mass_spring_damper_demo_sqp_measurement_jacobian_x::Measurement_Jacobian_x_Type<T>;
    using State_Hessian_XX_Matrix_Type =
        sqp_2_mass_spring_damper_demo_sqp_hessian_f_xx::State_Hessian_xx_Type<T>;
    using State_Hessian_XU_Matrix_Type =
        sqp_2_mass_spring_damper_demo_sqp_hessian_f_xu::State_Hessian_xu_Type<T>;
    using State_Hessian_UX_Matrix_Type =
        sqp_2_mass_spring_damper_demo_sqp_hessian_f_ux::State_Hessian_ux_Type<T>;
    using State_Hessian_UU_Matrix_Type =
        sqp_2_mass_spring_damper_demo_sqp_hessian_f_uu::State_Hessian_uu_Type<T>;
    using Measurement_Hessian_XX_Matrix_Type =
        sqp_2_mass_spring_damper_demo_sqp_hessian_h_xx::Measurement_Hessian_xx_Type<T>;

    using X_Type = StateSpaceState_Type<T, STATE_SIZE>;
    using U_Type = StateSpaceInput_Type<T, INPUT_SIZE>;
    using Y_Type = StateSpaceOutput_Type<T, OUTPUT_SIZE>;

    PythonOptimization::StateFunction_Object<X_Type, U_Type, Parameter_Type> state_function =
        sqp_2_mass_spring_damper_demo_sqp_state_function::Function<T, X_Type, U_Type, Parameter_Type>::function;
    PythonOptimization::MeasurementFunction_Object<Y_Type, X_Type, U_Type, Parameter_Type> measurement_function =
        sqp_2_mass_spring_damper_demo_sqp_measurement_function::Function<
        T, X_Type, U_Type, Parameter_Type, Y_Type>::function;

    PythonOptimization::StateFunctionJacobian_X_Object<
        State_Jacobian_X_Matrix_Type, X_Type, U_Type, Parameter_Type> state_jacobian_x_function =
        sqp_2_mass_spring_damper_demo_sqp_state_jacobian_x::Function<T, X_Type, U_Type, Parameter_Type>::function;
    PythonOptimization::StateFunctionJacobian_U_Object<
        State_Jacobian_U_Matrix_Type, X_Type, U_Type, Parameter_Type> state_jacobian_u_function =
        sqp_2_mass_spring_damper_demo_sqp_state_jacobian_u::Function<T, X_Type, U_Type, Parameter_Type>::function;
    PythonOptimization::MeasurementFunctionJacobian_X_Object<
        Measurement_Jacobian_X_Matrix_Type, X_Type, U_Type, Parameter_Type> measurement_jacobian_x_function =
        sqp_2_mass_spring_damper_demo_sqp_measurement_jacobian_x::Function<T, X_Type, U_Type, Parameter_Type>::function;

    PythonOptimization::StateFunctionHessian_XX_Object<
        State_Hessian_XX_Matrix_Type, X_Type, U_Type, Parameter_Type> state_hessian_xx_function =
        sqp_2_mass_spring_damper_demo_sqp_hessian_f_xx::Function<T, X_Type, U_Type, Parameter_Type>::function;
    PythonOptimization::StateFunctionHessian_XU_Object<
        State_Hessian_XU_Matrix_Type, X_Type, U_Type, Parameter_Type> state_hessian_xu_function =
        sqp_2_mass_spring_damper_demo_sqp_hessian_f_xu::Function<T, X_Type, U_Type, Parameter_Type>::function;
    PythonOptimization::StateFunctionHessian_UX_Object<
        State_Hessian_UX_Matrix_Type, X_Type, U_Type, Parameter_Type> state_hessian_ux_function =
        sqp_2_mass_spring_damper_demo_sqp_hessian_f_ux::Function<T, X_Type, U_Type, Parameter_Type>::function;
    PythonOptimization::StateFunctionHessian_UU_Object<
        State_Hessian_UU_Matrix_Type, X_Type, U_Type, Parameter_Type> state_hessian_uu_function =
        sqp_2_mass_spring_damper_demo_sqp_hessian_f_uu::Function<T, X_Type, U_Type, Parameter_Type>::function;
    PythonOptimization::MeasurementFunctionHessian_XX_Object<
        Measurement_Hessian_XX_Matrix_Type, X_Type, U_Type, Parameter_Type> measurement_hessian_xx_function =
        sqp_2_mass_spring_damper_demo_sqp_hessian_h_xx::Function<T, X_Type, U_Type, Parameter_Type>::function;

    using Qx_Type = DiagMatrix_Type<T, STATE_SIZE>;
    using R_Type = DiagMatrix_Type<T, INPUT_SIZE>;
    using Qy_Type = DiagMatrix_Type<T, OUTPUT_SIZE>;

    Qx_Type Qx = make_DiagMatrix<STATE_SIZE>(
        static_cast<T>(0.5), static_cast<T>(0.1), static_cast<T>(0.5), static_cast<T>(0.1)
    );
    R_Type R = make_DiagMatrix<INPUT_SIZE>(
        static_cast<T>(0.1), static_cast<T>(0.1)
    );
    Qy_Type Qy = make_DiagMatrix<OUTPUT_SIZE>(
        static_cast<T>(0.5), static_cast<T>(0.5)
    );

    using U_Min_Type = StateSpaceInput_Type<T, INPUT_SIZE>;
    using U_Max_Type = StateSpaceInput_Type<T, INPUT_SIZE>;

    U_Min_Type u_min = make_DenseMatrix<INPUT_SIZE, 1>(
        static_cast<T>(-1),
        static_cast<T>(-1)
    );
    U_Max_Type u_max = make_DenseMatrix<INPUT_SIZE, 1>(
        static_cast<T>(1),
        static_cast<T>(1)
    );

    using Y_Min_Type = SparseMatrixEmpty_Type<T, OUTPUT_SIZE, 1>;
    using Y_Max_Type = SparseMatrixEmpty_Type<T, OUTPUT_SIZE, 1>;

    Y_Min_Type y_min;
    Y_Max_Type y_max;

    using Reference_Trajectory_Type = DenseMatrix_Type<T, OUTPUT_SIZE, (NP + 1)>;

    Reference_Trajectory_Type reference_trajectory;

    using Cost_Matrices_Type = SQP_CostMatrices_NMPC_Type<T, NP, Parameter_Type,
        U_Min_Type, U_Max_Type, Y_Min_Type, Y_Max_Type,
        State_Jacobian_X_Matrix_Type,
        State_Jacobian_U_Matrix_Type,
        Measurement_Jacobian_X_Matrix_Type,
        State_Hessian_XX_Matrix_Type,
        State_Hessian_XU_Matrix_Type,
        State_Hessian_UX_Matrix_Type,
        State_Hessian_UU_Matrix_Type,
        Measurement_Hessian_XX_Matrix_Type>;

    Cost_Matrices_Type cost_matrices =
        make_SQP_CostMatrices_NMPC<T, NP, Parameter_Type,
        U_Min_Type, U_Max_Type, Y_Min_Type, Y_Max_Type,
        State_Jacobian_X_Matrix_Type,
        State_Jacobian_U_Matrix_Type,
        Measurement_Jacobian_X_Matrix_Type,
        State_Hessian_XX_Matrix_Type,
        State_Hessian_XU_Matrix_Type,
        State_Hessian_UX_Matrix_Type,
        State_Hessian_UU_Matrix_Type,
        Measurement_Hessian_XX_Matrix_Type>(
            Qx, R, Qy, u_min, u_max, y_min, y_max);

    cost_matrices.set_function_objects(
        state_function,
        measurement_function,
        state_jacobian_x_function,
        state_jacobian_u_function,
        measurement_jacobian_x_function,
        state_hessian_xx_function,
        state_hessian_xu_function,
        state_hessian_ux_function,
        state_hessian_uu_function,
        measurement_hessian_xx_function
    );

    /* 関数オブジェクト定義 */
    using U_horizon_Type = DenseMatrix_Type<T, INPUT_SIZE, NP>;
    using Gradient_Type = U_horizon_Type;
    using V_Horizon_Type = U_horizon_Type;
    using HVP_Type = U_horizon_Type;

    CostFunction_Object<X_Type, U_horizon_Type> cost_function =
        [&cost_matrices](const X_Type& X, const U_horizon_Type& U)
        -> typename X_Type::Value_Type {
        return cost_matrices.compute_cost(X, U);
        };
    CostAndGradientFunction_Object<X_Type, U_horizon_Type, Gradient_Type>
        cost_and_gradient_function =
        [&cost_matrices](const X_Type& X, const U_horizon_Type& U,
            typename X_Type::Value_Type& J,
            Gradient_Type& gradient) {
                cost_matrices.compute_cost_and_gradient(X, U, J, gradient);
        };
    HVP_Function_Object<X_Type, U_horizon_Type, V_Horizon_Type, HVP_Type>
        hvp_function = [&cost_matrices](const X_Type& X, const U_horizon_Type& U,
            const V_Horizon_Type& V) -> HVP_Type {
                return cost_matrices.hvp_analytic(X, U, V);
        };

    cost_matrices.reference_trajectory = reference_trajectory;

    auto X_initial = make_DenseMatrix<STATE_SIZE, 1>(
        static_cast<T>(5),
        static_cast<T>(0),
        static_cast<T>(5),
        static_cast<T>(0)
    );

    DenseMatrix_Type<T, INPUT_SIZE, NP> U_horizon_initial;

    /* SQP Active Set PCG PLS 定義 */
    SQP_ActiveSet_PCG_PLS_Type<Cost_Matrices_Type> solver =
        make_SQP_ActiveSet_PCG_PLS<Cost_Matrices_Type>();

    /* コピー、ムーブ */
    solver.X_initial = X_initial;

    SQP_ActiveSet_PCG_PLS_Type<Cost_Matrices_Type> solver_copy = solver;
    SQP_ActiveSet_PCG_PLS_Type<Cost_Matrices_Type> solver_move = std::move(solver_copy);
    solver = std::move(solver_move);

    tester.expect_near(solver.X_initial.matrix.data, X_initial.matrix.data, NEAR_LIMIT_STRICT,
        "check X_initial.");

    /* solve */
    solver.set_solver_max_iteration(20);

    auto U_horizon_opt = solver.solve(
        U_horizon_initial,
        cost_and_gradient_function,
        cost_function,
        hvp_function,
        X_initial,
        cost_matrices.get_U_min_matrix(),
        cost_matrices.get_U_max_matrix()
    );

    auto U_horizon_opt_answer = Matrix<DefDense, T, INPUT_SIZE, NP>({
        {static_cast<T>(1), static_cast<T>(1), static_cast<T>(1),
         static_cast<T>(1), static_cast<T>(1), static_cast<T>(1),
         static_cast<T>(1), static_cast<T>(1),
         static_cast<T>(0.88495493), static_cast<T>(-0.04190774) },
        {static_cast<T>(1), static_cast<T>(1), static_cast<T>(1),
         static_cast<T>(1), static_cast<T>(1), static_cast<T>(1),
         static_cast<T>(1), static_cast<T>(1), 
         static_cast<T>(0.88495493), static_cast<T>(-0.04190774) }
    });

    tester.expect_near(U_horizon_opt.matrix.data, U_horizon_opt_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check U_horizon_opt.");


    tester.throw_error_if_test_failed();
}



template <typename T>
void test_OptimizationEngine_CostMatrices() {
    using namespace PythonNumpy;
    using namespace PythonControl;
    using namespace PythonOptimization;
    using namespace SQP_TestData;

    MCAPTester<T> tester;

    constexpr T NEAR_LIMIT_STRICT = std::is_same<T, double>::value ? T(1.0e-5) : T(1.0e-4);
    //const T NEAR_LIMIT_SOFT = 1.0e-2F;

    /* 定義 */
    constexpr std::size_t STATE_SIZE = 4;
    constexpr std::size_t INPUT_SIZE = 2;
    constexpr std::size_t OUTPUT_SIZE = 2;

    constexpr std::size_t NP = 10;

    using Parameter_Type = sqp_2_mass_spring_damper_demo_parameter::Parameter<T>;

    using State_Jacobian_X_Matrix_Type =
        sqp_2_mass_spring_damper_demo_sqp_state_jacobian_x::State_Jacobian_x_Type<T>;
    using State_Jacobian_U_Matrix_Type =
        sqp_2_mass_spring_damper_demo_sqp_state_jacobian_u::State_Jacobian_u_Type<T>;
    using Measurement_Jacobian_X_Matrix_Type =
        sqp_2_mass_spring_damper_demo_sqp_measurement_jacobian_x::Measurement_Jacobian_x_Type<T>;

    using X_Type = StateSpaceState_Type<T, STATE_SIZE>;
    using U_Type = StateSpaceInput_Type<T, INPUT_SIZE>;
    using Y_Type = StateSpaceOutput_Type<T, OUTPUT_SIZE>;

    PythonOptimization::StateFunction_Object<X_Type, U_Type, Parameter_Type> state_function =
        sqp_2_mass_spring_damper_demo_sqp_state_function::Function<T, X_Type, U_Type, Parameter_Type>::function;
    PythonOptimization::MeasurementFunction_Object<Y_Type, X_Type, U_Type, Parameter_Type> measurement_function =
        sqp_2_mass_spring_damper_demo_sqp_measurement_function::Function<
        T, X_Type, U_Type, Parameter_Type, Y_Type>::function;

    PythonOptimization::StateFunctionJacobian_X_Object<
        State_Jacobian_X_Matrix_Type, X_Type, U_Type, Parameter_Type> state_jacobian_x_function =
        sqp_2_mass_spring_damper_demo_sqp_state_jacobian_x::Function<T, X_Type, U_Type, Parameter_Type>::function;
    PythonOptimization::StateFunctionJacobian_U_Object<
        State_Jacobian_U_Matrix_Type, X_Type, U_Type, Parameter_Type> state_jacobian_u_function =
        sqp_2_mass_spring_damper_demo_sqp_state_jacobian_u::Function<T, X_Type, U_Type, Parameter_Type>::function;
    PythonOptimization::MeasurementFunctionJacobian_X_Object<
        Measurement_Jacobian_X_Matrix_Type, X_Type, U_Type, Parameter_Type> measurement_jacobian_x_function =
        sqp_2_mass_spring_damper_demo_sqp_measurement_jacobian_x::Function<T, X_Type, U_Type, Parameter_Type>::function;

    using Qx_Type = DiagMatrix_Type<T, STATE_SIZE>;
    using R_Type = DiagMatrix_Type<T, INPUT_SIZE>;
    using Qy_Type = DiagMatrix_Type<T, OUTPUT_SIZE>;

    Qx_Type Qx = make_DiagMatrix<STATE_SIZE>(
        static_cast<T>(0.5), static_cast<T>(0.1), static_cast<T>(0.5), static_cast<T>(0.1)
    );
    R_Type R = make_DiagMatrix<INPUT_SIZE>(
        static_cast<T>(0.1), static_cast<T>(0.1)
    );
    Qy_Type Qy = make_DiagMatrix<OUTPUT_SIZE>(
        static_cast<T>(0.5), static_cast<T>(0.5)
    );

    using U_Min_Type = StateSpaceInput_Type<T, INPUT_SIZE>;
    using U_Max_Type = StateSpaceInput_Type<T, INPUT_SIZE>;

    U_Min_Type u_min = make_DenseMatrix<INPUT_SIZE, 1>(
        static_cast<T>(-1),
        static_cast<T>(-1)
    );
    U_Max_Type u_max = make_DenseMatrix<INPUT_SIZE, 1>(
        static_cast<T>(1),
        static_cast<T>(1)
    );

    using Y_Min_Type = SparseMatrixEmpty_Type<T, OUTPUT_SIZE, 1>;
    using Y_Max_Type = SparseMatrixEmpty_Type<T, OUTPUT_SIZE, 1>;

    Y_Min_Type y_min;
    Y_Max_Type y_max;

    using Reference_Trajectory_Type = DenseMatrix_Type<T, OUTPUT_SIZE, (NP + 1)>;

    Reference_Trajectory_Type reference_trajectory;

    /* コンストラクタ */
    using Cost_Matrices_Type = OptimizationEngine_CostMatrices_Type<T, NP, Parameter_Type,
        U_Min_Type, U_Max_Type, Y_Min_Type, Y_Max_Type,
        State_Jacobian_X_Matrix_Type,
        State_Jacobian_U_Matrix_Type,
        Measurement_Jacobian_X_Matrix_Type>;

    Cost_Matrices_Type cost_matrices =
        make_OptimizationEngine_CostMatrices<T, NP, Parameter_Type,
        U_Min_Type, U_Max_Type, Y_Min_Type, Y_Max_Type,
        State_Jacobian_X_Matrix_Type,
        State_Jacobian_U_Matrix_Type,
        Measurement_Jacobian_X_Matrix_Type>(
            Qx, R, Qy, u_min, u_max, y_min, y_max);

    /* コピー、ムーブ */
    Cost_Matrices_Type cost_matrices_copy = cost_matrices;
    Cost_Matrices_Type cost_matrices_move = std::move(cost_matrices_copy);
    cost_matrices = std::move(cost_matrices_move);

    T u_min_value = cost_matrices.state_space_parameters.k1;

    tester.expect_near(u_min_value, static_cast<T>(10), NEAR_LIMIT_STRICT,
        "check u_min value.");

    cost_matrices.set_function_objects(
        state_function,
        measurement_function,
        state_jacobian_x_function,
        state_jacobian_u_function,
        measurement_jacobian_x_function
    );

    cost_matrices.reference_trajectory = reference_trajectory;

    auto X_initial = make_DenseMatrix<STATE_SIZE, 1>(
        static_cast<T>(5),
        static_cast<T>(0),
        static_cast<T>(5),
        static_cast<T>(0)
    );

    cost_matrices.X_initial = X_initial;

    typename Cost_Matrices_Type::U_Horizon_Type U_horizon_initial;

    /* compute_cost 動作確認 */
    auto cost_value = cost_matrices.compute_cost(U_horizon_initial);

    tester.expect_near(cost_value > static_cast<T>(0), static_cast<T>(true),
        NEAR_LIMIT_STRICT,
        "check compute_cost positive.");

    /* compute_gradient 動作確認 */
    auto gradient = cost_matrices.compute_gradient(U_horizon_initial);

    auto gradient_row_0 = MatrixOperation::get_row(gradient, 0);

    T gradient_0_norm = static_cast<T>(0);
    for (std::size_t i = 0; i < gradient_row_0.matrix.data.size(); i++) {
        gradient_0_norm += gradient_row_0(i, 0) * gradient_row_0(i, 0);
    }
    gradient_0_norm = std::sqrt(gradient_0_norm);

    tester.expect_near(gradient_0_norm > static_cast<T>(0), static_cast<T>(true),
        NEAR_LIMIT_STRICT,
        "check compute_gradient non-zero.");

    /* compute_output_mapping 動作確認 */
    auto Y_horizon = cost_matrices.compute_output_mapping(U_horizon_initial);

    auto Y_horizon_row_0 = MatrixOperation::get_row(Y_horizon, 0);

    T y_horizon_0_norm = static_cast<T>(0);
    for (std::size_t i = 0; i < Y_horizon_row_0.matrix.data.size(); i++) {
        y_horizon_0_norm += Y_horizon_row_0(i, 0) * Y_horizon_row_0(i, 0);
    }
    y_horizon_0_norm = std::sqrt(y_horizon_0_norm);

    tester.expect_near(y_horizon_0_norm > static_cast<T>(0), static_cast<T>(true),
        NEAR_LIMIT_STRICT,
        "check compute_output_mapping non-zero.");

    /* compute_output_jacobian_trans 動作確認 */
    typename Cost_Matrices_Type::Y_Horizon_Type D;

    for (std::size_t k = 0; k < (NP + 1); k++) {
        auto d_row = make_DenseMatrix<OUTPUT_SIZE, 1>(
            static_cast<T>(1), static_cast<T>(1)
        );
        MatrixOperation::set_row(D, d_row, k);
    }

    auto jf1_trans_d = cost_matrices.compute_output_jacobian_trans(U_horizon_initial, D);

    auto jf1_row_0 = MatrixOperation::get_row(jf1_trans_d, 0);

    T jf1_0_norm = static_cast<T>(0);
    for (std::size_t i = 0; i < jf1_row_0.matrix.data.size(); i++) {
        jf1_0_norm += jf1_row_0(i, 0) * jf1_row_0(i, 0);
    }
    jf1_0_norm = std::sqrt(jf1_0_norm);

    tester.expect_near(jf1_0_norm > static_cast<T>(0), static_cast<T>(true),
        NEAR_LIMIT_STRICT,
        "check compute_output_jacobian_trans non-zero.");

    /* getter 動作確認 */
    auto Qx_get = cost_matrices.get_Qx();
    tester.expect_near(Qx_get.matrix.data, Qx.matrix.data, NEAR_LIMIT_STRICT,
        "check get_Qx.");

    auto R_get = cost_matrices.get_R();
    tester.expect_near(R_get.matrix.data, R.matrix.data, NEAR_LIMIT_STRICT,
        "check get_R.");

    auto Qy_get = cost_matrices.get_Qy();
    tester.expect_near(Qy_get.matrix.data, Qy.matrix.data, NEAR_LIMIT_STRICT,
        "check get_Qy.");


    tester.throw_error_if_test_failed();
}


template <typename T>
void test_alm_pm_optimizer() {
    using namespace PythonNumpy;
    using namespace PythonControl;
    using namespace PythonOptimization;
    using namespace SQP_TestData;

    MCAPTester<T> tester;

    constexpr T NEAR_LIMIT_STRICT = std::is_same<T, double>::value ? T(1.0e-5) : T(1.0e-4);
    const T NEAR_LIMIT_SOFT = std::is_same<T, double>::value ? T(1.0e-2) : T(1.0e-1);

    /* Cost Matrices 定義 */
    constexpr std::size_t STATE_SIZE = 4;
    constexpr std::size_t INPUT_SIZE = 2;
    constexpr std::size_t OUTPUT_SIZE = 2;

    constexpr std::size_t NP = 10;

    using Parameter_Type = sqp_2_mass_spring_damper_demo_parameter::Parameter<T>;

    using State_Jacobian_X_Matrix_Type =
        sqp_2_mass_spring_damper_demo_sqp_state_jacobian_x::State_Jacobian_x_Type<T>;
    using State_Jacobian_U_Matrix_Type =
        sqp_2_mass_spring_damper_demo_sqp_state_jacobian_u::State_Jacobian_u_Type<T>;
    using Measurement_Jacobian_X_Matrix_Type =
        sqp_2_mass_spring_damper_demo_sqp_measurement_jacobian_x::Measurement_Jacobian_x_Type<T>;

    using X_Type = StateSpaceState_Type<T, STATE_SIZE>;
    using U_Type = StateSpaceInput_Type<T, INPUT_SIZE>;
    using Y_Type = StateSpaceOutput_Type<T, OUTPUT_SIZE>;

    PythonOptimization::StateFunction_Object<X_Type, U_Type, Parameter_Type> state_function =
        sqp_2_mass_spring_damper_demo_sqp_state_function::Function<T, X_Type, U_Type, Parameter_Type>::function;
    PythonOptimization::MeasurementFunction_Object<Y_Type, X_Type, U_Type, Parameter_Type> measurement_function =
        sqp_2_mass_spring_damper_demo_sqp_measurement_function::Function<
        T, X_Type, U_Type, Parameter_Type, Y_Type>::function;

    PythonOptimization::StateFunctionJacobian_X_Object<
        State_Jacobian_X_Matrix_Type, X_Type, U_Type, Parameter_Type> state_jacobian_x_function =
        sqp_2_mass_spring_damper_demo_sqp_state_jacobian_x::Function<T, X_Type, U_Type, Parameter_Type>::function;
    PythonOptimization::StateFunctionJacobian_U_Object<
        State_Jacobian_U_Matrix_Type, X_Type, U_Type, Parameter_Type> state_jacobian_u_function =
        sqp_2_mass_spring_damper_demo_sqp_state_jacobian_u::Function<T, X_Type, U_Type, Parameter_Type>::function;
    PythonOptimization::MeasurementFunctionJacobian_X_Object<
        Measurement_Jacobian_X_Matrix_Type, X_Type, U_Type, Parameter_Type> measurement_jacobian_x_function =
        sqp_2_mass_spring_damper_demo_sqp_measurement_jacobian_x::Function<T, X_Type, U_Type, Parameter_Type>::function;

    using Qx_Type = DiagMatrix_Type<T, STATE_SIZE>;
    using R_Type = DiagMatrix_Type<T, INPUT_SIZE>;
    using Qy_Type = DiagMatrix_Type<T, OUTPUT_SIZE>;

    Qx_Type Qx = make_DiagMatrix<STATE_SIZE>(
        static_cast<T>(0.5), static_cast<T>(0.1), static_cast<T>(0.5), static_cast<T>(0.1)
    );
    R_Type R = make_DiagMatrix<INPUT_SIZE>(
        static_cast<T>(0.1), static_cast<T>(0.1)
    );
    Qy_Type Qy = make_DiagMatrix<OUTPUT_SIZE>(
        static_cast<T>(0.5), static_cast<T>(0.5)
    );

    using U_Min_Type = StateSpaceInput_Type<T, INPUT_SIZE>;
    using U_Max_Type = StateSpaceInput_Type<T, INPUT_SIZE>;

    U_Min_Type u_min = make_DenseMatrix<INPUT_SIZE, 1>(
        static_cast<T>(-1),
        static_cast<T>(-1)
    );
    U_Max_Type u_max = make_DenseMatrix<INPUT_SIZE, 1>(
        static_cast<T>(1),
        static_cast<T>(1)
    );

    using Y_Min_Type = SparseMatrixEmpty_Type<T, OUTPUT_SIZE, 1>;
    using Y_Max_Type = SparseMatrixEmpty_Type<T, OUTPUT_SIZE, 1>;

    Y_Min_Type y_min;
    Y_Max_Type y_max;

    using Reference_Trajectory_Type = DenseMatrix_Type<T, OUTPUT_SIZE, (NP + 1)>;

    Reference_Trajectory_Type reference_trajectory;

    using Cost_Matrices_Type = OptimizationEngine_CostMatrices_Type<T, NP, Parameter_Type,
        U_Min_Type, U_Max_Type, Y_Min_Type, Y_Max_Type,
        State_Jacobian_X_Matrix_Type,
        State_Jacobian_U_Matrix_Type,
        Measurement_Jacobian_X_Matrix_Type>;

    Cost_Matrices_Type cost_matrices =
        make_OptimizationEngine_CostMatrices<T, NP, Parameter_Type,
        U_Min_Type, U_Max_Type, Y_Min_Type, Y_Max_Type,
        State_Jacobian_X_Matrix_Type,
        State_Jacobian_U_Matrix_Type,
        Measurement_Jacobian_X_Matrix_Type>(
            Qx, R, Qy, u_min, u_max, y_min, y_max);

    cost_matrices.set_function_objects(
        state_function,
        measurement_function,
        state_jacobian_x_function,
        state_jacobian_u_function,
        measurement_jacobian_x_function
    );

    cost_matrices.reference_trajectory = reference_trajectory;

    auto X_initial = make_DenseMatrix<STATE_SIZE, 1>(
        static_cast<T>(5),
        static_cast<T>(0),
        static_cast<T>(5),
        static_cast<T>(0)
    );

    cost_matrices.X_initial = X_initial;

    /* ALM 定義 */
    constexpr std::size_t N1 = OUTPUT_SIZE * (NP + 1);
    constexpr std::size_t N2 = 0;

    using U_Horizon_Type = typename Cost_Matrices_Type::U_Horizon_Type;
    using Y_Horizon_Type = typename Cost_Matrices_Type::Y_Horizon_Type;
    using F1_Output_Type = DenseMatrix_Type<T, N1, 1>;
    using Xi_Type = DenseMatrix_Type<T, (1 + N1), 1>;

    /* ALM Factory */
    ALM_Factory_Type<Cost_Matrices_Type, N1, N2> alm_factory;

    alm_factory.set_cost_function(
        [&cost_matrices](const U_Horizon_Type& u) -> T {
            return cost_matrices.compute_cost(u);
        }
    );

    alm_factory.set_gradient_function(
        [&cost_matrices](const U_Horizon_Type& u) -> U_Horizon_Type {
            return cost_matrices.compute_gradient(u);
        }
    );

    auto mapping_f1_func =
        [&cost_matrices](const U_Horizon_Type& u) -> F1_Output_Type {
            auto Y_horizon = cost_matrices.compute_output_mapping(u);
            F1_Output_Type f1;
            for (std::size_t k = 0; k < (NP + 1); k++) {
                auto y_k = MatrixOperation::get_row(Y_horizon, k);
                for (std::size_t j = 0; j < OUTPUT_SIZE; j++) {
                    f1(0, k * OUTPUT_SIZE + j) = y_k(0, j);
                }
            }
            return f1;
        };

    alm_factory.set_mapping_f1(mapping_f1_func);

    alm_factory.set_jacobian_f1_trans(
        [&cost_matrices](const U_Horizon_Type& u,
            const F1_Output_Type& d) -> U_Horizon_Type {
            Y_Horizon_Type D;
            for (std::size_t k = 0; k < (NP + 1); k++) {
                Y_Type d_k;
                for (std::size_t j = 0; j < OUTPUT_SIZE; j++) {
                    d_k(0, j) = d(0, k * OUTPUT_SIZE + j);
                }
                MatrixOperation::set_row(D, d_k, k);
            }
            return cost_matrices.compute_output_jacobian_trans(u, D);
        }
    );

    /* 出力制約の Box Projection (広い範囲で非拘束) */
    F1_Output_Type y_min_flat, y_max_flat;
    for (std::size_t k = 0; k < (NP + 1); k++) {
        for (std::size_t j = 0; j < OUTPUT_SIZE; j++) {
            y_min_flat(0, k * OUTPUT_SIZE + j) = static_cast<T>(-100);
            y_max_flat(0, k * OUTPUT_SIZE + j) = static_cast<T>(100);
        }
    }

    BoxProjectionOperator_Type<T, N1> box_proj_c(y_min_flat, y_max_flat);

    alm_factory.set_c_projection(
        [&box_proj_c](F1_Output_Type& x) {
            box_proj_c.project(x);
        }
    );

    /* ALM Problem */
    ALM_Problem_Type<Cost_Matrices_Type, N1, N2> problem;

    problem.set_parametric_cost(
        [&alm_factory](const U_Horizon_Type& u,
            const Xi_Type& xi) -> T {
            return alm_factory.psi(u, xi);
        }
    );

    problem.set_parametric_gradient(
        [&alm_factory](const U_Horizon_Type& u,
            const Xi_Type& xi) -> U_Horizon_Type {
            return alm_factory.d_psi(u, xi);
        }
    );

    problem.set_u_min_matrix(cost_matrices.get_U_min_matrix());
    problem.set_u_max_matrix(cost_matrices.get_U_max_matrix());
    problem.set_mapping_f1(mapping_f1_func);

    problem.set_c_projection(
        [&box_proj_c](F1_Output_Type& x) {
            box_proj_c.project(x);
        }
    );

    /* ALM_PM_Optimizer */
    ALM_PM_Optimizer_Type<Cost_Matrices_Type, N1, N2> solver =
        make_ALM_PM_Optimizer<Cost_Matrices_Type, N1, N2>();

    solver.set_problem(problem);

    /* コピー、ムーブ */
    ALM_PM_Optimizer_Type<Cost_Matrices_Type, N1, N2> solver_copy = solver;
    ALM_PM_Optimizer_Type<Cost_Matrices_Type, N1, N2> solver_move = std::move(solver_copy);
    solver = std::move(solver_move);

    /* solve */
    solver.set_solver_max_iteration(50, 500);
    solver.set_epsilon_tolerance(static_cast<T>(1.0e-4));
    solver.set_delta_tolerance(static_cast<T>(1.0e-4));
    solver.set_initial_penalty(static_cast<T>(10));
    solver.set_initial_inner_tolerance(static_cast<T>(0.1));

    U_Horizon_Type U_horizon_initial;

    auto U_horizon_opt = solver.solve(U_horizon_initial);

    auto U_horizon_opt_answer = Matrix<DefDense, T, INPUT_SIZE, NP>({
        {static_cast<T>(1), static_cast<T>(1), static_cast<T>(1),
         static_cast<T>(1), static_cast<T>(1), static_cast<T>(1),
         static_cast<T>(1), static_cast<T>(1),
         static_cast<T>(0.88495493), static_cast<T>(-0.04190774) },
        {static_cast<T>(1), static_cast<T>(1), static_cast<T>(1),
         static_cast<T>(1), static_cast<T>(1), static_cast<T>(1),
         static_cast<T>(1), static_cast<T>(1),
         static_cast<T>(0.88495493), static_cast<T>(-0.04190774) }
    });

    tester.expect_near(U_horizon_opt.matrix.data, U_horizon_opt_answer.matrix.data, NEAR_LIMIT_SOFT,
        "check U_horizon_opt.");

    /* solver status 確認 */
    auto status = solver.get_solver_status();

    tester.expect_near(static_cast<T>(status.has_converged()), static_cast<T>(true),
        NEAR_LIMIT_STRICT,
        "check convergence.");

    tester.expect_near(status.cost > static_cast<T>(0), static_cast<T>(true),
        NEAR_LIMIT_STRICT,
        "check cost positive.");

    std::size_t outer_count = 0;
    std::size_t inner_count = 0;
    solver.get_solver_step_iterated_number(outer_count, inner_count);

    tester.expect_near(static_cast<T>(outer_count > 0), static_cast<T>(true),
        NEAR_LIMIT_STRICT,
        "check outer iteration count positive.");

    tester.expect_near(static_cast<T>(inner_count > 0), static_cast<T>(true),
        NEAR_LIMIT_STRICT,
        "check inner iteration count positive.");


    tester.throw_error_if_test_failed();
}


int main() {

    test_active_set<double>();

    test_active_set<float>();

    test_active_set_2D<double>();

    test_active_set_2D<float>();

    test_qp_active_set_solver<double>();

    test_qp_active_set_solver<float>();

    test_SQP_CostMatrices_NMPC<double>();

    test_SQP_CostMatrices_NMPC<float>();

    test_sqp_active_set_pcg_pls<double>();

    test_sqp_active_set_pcg_pls<float>();

    test_OptimizationEngine_CostMatrices<double>();

    test_OptimizationEngine_CostMatrices<float>();

    test_alm_pm_optimizer<double>();

    test_alm_pm_optimizer<float>();


    return 0;
}
