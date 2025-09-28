#ifndef __SQP_TEST_DATA_HPP__
#define __SQP_TEST_DATA_HPP__

#include "python_math.hpp"
#include "python_numpy.hpp"


namespace SQP_TestData {

namespace sqp_2_mass_spring_damper_demo_parameter {

template <typename T>
class Parameter {
public:
    T m1 = static_cast<T>(1.0);
    T m2 = static_cast<T>(1.0);
    T k1 = static_cast<T>(10.0);
    T k2 = static_cast<T>(15.0);
    T k3 = static_cast<T>(10.0);
    T b1 = static_cast<T>(1.0);
    T b2 = static_cast<T>(2.0);
    T b3 = static_cast<T>(1.0);
    T dt = static_cast<T>(0.1);
};

} // namespace sqp_2_mass_spring_damper_demo_parameter

namespace sqp_2_mass_spring_damper_demo_sqp_state_function {

using namespace PythonMath;

template <typename T, typename X_Type, typename U_Type, typename Parameter_Type>
class Function {
public:
    static inline auto sympy_function(const T dt, const T x1, const T k2, const T u1, const T k3, const T b2, const T b3, const T b1, const T x2, const T v1, const T v2, const T u2, const T m1, const T m2, const T k1) -> X_Type {

        X_Type result;

        T x0 = v1 - v2;

        T x3 = x1 - x2;

        result.template set<0, 0>(static_cast<T>(dt * v1 + x1));
        result.template set<1, 0>(static_cast<T>(dt * (-b1 * v1 - b2 * x0 - k1 * x1 - k2 * x3 + u1) / m1 + v1));
        result.template set<2, 0>(static_cast<T>(dt * v2 + x2));
        result.template set<3, 0>(static_cast<T>(dt * (b2 * x0 - b3 * v2 + k2 * x3 - k3 * x2 + u2) / m2 + v2));

        return result;
    }

    static inline auto function(const X_Type X, const U_Type U, const Parameter_Type Parameters) -> X_Type {

        T x1 = X.template get<0, 0>();

        T v1 = X.template get<1, 0>();

        T x2 = X.template get<2, 0>();

        T v2 = X.template get<3, 0>();

        T u1 = U.template get<0, 0>();

        T u2 = U.template get<1, 0>();

        T dt = Parameters.dt;

        T k2 = Parameters.k2;

        T k3 = Parameters.k3;

        T b2 = Parameters.b2;

        T b3 = Parameters.b3;

        T b1 = Parameters.b1;

        T m1 = Parameters.m1;

        T m2 = Parameters.m2;

        T k1 = Parameters.k1;

        return sympy_function(dt, x1, k2, u1, k3, b2, b3, b1, x2, v1, v2, u2, m1, m2, k1);
    }

};

} // namespace sqp_2_mass_spring_damper_demo_sqp_state_function

namespace sqp_2_mass_spring_damper_demo_sqp_measurement_function {

using namespace PythonMath;

template <typename T, typename X_Type, typename U_Type, typename Parameter_Type, typename Y_Type>
class Function {
public:
    static inline auto sympy_function(const T x1, const T x2) -> Y_Type {

        Y_Type result;

        result.template set<0, 0>(static_cast<T>(x1));
        result.template set<1, 0>(static_cast<T>(x2));

        return result;
    }

    static inline auto function(const X_Type X, const U_Type U, const Parameter_Type Parameters) -> Y_Type {
        static_cast<void>(U);
        static_cast<void>(Parameters);

        T x1 = X.template get<0, 0>();

        T x2 = X.template get<2, 0>();

        return sympy_function(x1, x2);
    }

};

} // namespace sqp_2_mass_spring_damper_demo_sqp_measurement_function


namespace sqp_2_mass_spring_damper_demo_sqp_state_jacobian_x {

using namespace PythonMath;
using namespace PythonNumpy;

using State_Jacobian_x_Type_SparseAvailable = SparseAvailable<
    ColumnAvailable<true, true, false, false>,
    ColumnAvailable<true, true, true, true>,
    ColumnAvailable<false, false, true, true>,
    ColumnAvailable<true, true, true, true>
>;

template <typename T>
using State_Jacobian_x_Type = SparseMatrix_Type<T, State_Jacobian_x_Type_SparseAvailable>;

template <typename T, typename X_Type, typename U_Type, typename Parameter_Type>
class Function {
public:
    static inline auto sympy_function(const T b3, const T b1, const T dt, const T k2, const T k3, const T m1, const T b2, const T m2, const T k1) -> State_Jacobian_x_Type<T> {

        State_Jacobian_x_Type<T> result;

        T x0 = dt / m1;

        T x1 = dt / m2;

        result.template set<0, 0>(static_cast<T>(1));
        result.template set<0, 1>(static_cast<T>(dt));
        result.template set<0, 2>(static_cast<T>(0));
        result.template set<0, 3>(static_cast<T>(0));
        result.template set<1, 0>(static_cast<T>(x0 * (-k1 - k2)));
        result.template set<1, 1>(static_cast<T>(x0 * (-b1 - b2) + 1));
        result.template set<1, 2>(static_cast<T>(k2 * x0));
        result.template set<1, 3>(static_cast<T>(b2 * x0));
        result.template set<2, 0>(static_cast<T>(0));
        result.template set<2, 1>(static_cast<T>(0));
        result.template set<2, 2>(static_cast<T>(1));
        result.template set<2, 3>(static_cast<T>(dt));
        result.template set<3, 0>(static_cast<T>(k2 * x1));
        result.template set<3, 1>(static_cast<T>(b2 * x1));
        result.template set<3, 2>(static_cast<T>(x1 * (-k2 - k3)));
        result.template set<3, 3>(static_cast<T>(x1 * (-b2 - b3) + 1));

        return result;
    }

    static inline auto function(const X_Type X, const U_Type U, const Parameter_Type Parameters) -> State_Jacobian_x_Type<T> {
        static_cast<void>(X);
        static_cast<void>(U);

        T b3 = Parameters.b3;

        T b1 = Parameters.b1;

        T dt = Parameters.dt;

        T k2 = Parameters.k2;

        T k3 = Parameters.k3;

        T m1 = Parameters.m1;

        T b2 = Parameters.b2;

        T m2 = Parameters.m2;

        T k1 = Parameters.k1;

        return sympy_function(b3, b1, dt, k2, k3, m1, b2, m2, k1);
    }

};

} // namespace sqp_2_mass_spring_damper_demo_sqp_state_jacobian_x

namespace sqp_2_mass_spring_damper_demo_sqp_state_jacobian_u {

using namespace PythonMath;
using namespace PythonNumpy;

using State_Jacobian_u_Type_SparseAvailable = SparseAvailable<
    ColumnAvailable<false, false>,
    ColumnAvailable<true, false>,
    ColumnAvailable<false, false>,
    ColumnAvailable<false, true>
>;

template <typename T>
using State_Jacobian_u_Type = SparseMatrix_Type<T, State_Jacobian_u_Type_SparseAvailable>;

template <typename T, typename X_Type, typename U_Type, typename Parameter_Type>
class Function {
public:
    static inline auto sympy_function(const T m2, const T m1, const T dt) -> State_Jacobian_u_Type<T> {

        State_Jacobian_u_Type<T> result;

        result.template set<0, 0>(static_cast<T>(0));
        result.template set<0, 1>(static_cast<T>(0));
        result.template set<1, 0>(static_cast<T>(dt / m1));
        result.template set<1, 1>(static_cast<T>(0));
        result.template set<2, 0>(static_cast<T>(0));
        result.template set<2, 1>(static_cast<T>(0));
        result.template set<3, 0>(static_cast<T>(0));
        result.template set<3, 1>(static_cast<T>(dt / m2));

        return result;
    }

    static inline auto function(const X_Type X, const U_Type U, const Parameter_Type Parameters) -> State_Jacobian_u_Type<T> {
        static_cast<void>(X);
        static_cast<void>(U);

        T m2 = Parameters.m2;

        T m1 = Parameters.m1;

        T dt = Parameters.dt;

        return sympy_function(m2, m1, dt);
    }

};

} // namespace sqp_2_mass_spring_damper_demo_sqp_state_jacobian_u

namespace sqp_2_mass_spring_damper_demo_sqp_measurement_jacobian_x {

using namespace PythonMath;
using namespace PythonNumpy;

using Measurement_Jacobian_x_Type_SparseAvailable = SparseAvailable<
    ColumnAvailable<true, false, false, false>,
    ColumnAvailable<false, false, true, false>
>;

template <typename T>
using Measurement_Jacobian_x_Type = SparseMatrix_Type<T, Measurement_Jacobian_x_Type_SparseAvailable>;

template <typename T, typename X_Type, typename U_Type, typename Parameter_Type>
class Function {
public:
    static inline auto sympy_function() -> Measurement_Jacobian_x_Type<T> {

        Measurement_Jacobian_x_Type<T> result;

        result.template set<0, 0>(static_cast<T>(1));
        result.template set<0, 1>(static_cast<T>(0));
        result.template set<0, 2>(static_cast<T>(0));
        result.template set<0, 3>(static_cast<T>(0));
        result.template set<1, 0>(static_cast<T>(0));
        result.template set<1, 1>(static_cast<T>(0));
        result.template set<1, 2>(static_cast<T>(1));
        result.template set<1, 3>(static_cast<T>(0));

        return result;
    }

    static inline auto function(const X_Type X, const U_Type U, const Parameter_Type Parameters) -> Measurement_Jacobian_x_Type<T> {
        static_cast<void>(X);
        static_cast<void>(U);
        static_cast<void>(Parameters);


        return sympy_function();
    }

};

} // namespace sqp_2_mass_spring_damper_demo_sqp_measurement_jacobian_x

namespace sqp_2_mass_spring_damper_demo_sqp_hessian_f_xx {

using namespace PythonMath;
using namespace PythonNumpy;

using State_Hessian_xx_Type_SparseAvailable = SparseAvailable<
    ColumnAvailable<false, false, false, false>,
    ColumnAvailable<false, false, false, false>,
    ColumnAvailable<false, false, false, false>,
    ColumnAvailable<false, false, false, false>,
    ColumnAvailable<false, false, false, false>,
    ColumnAvailable<false, false, false, false>,
    ColumnAvailable<false, false, false, false>,
    ColumnAvailable<false, false, false, false>,
    ColumnAvailable<false, false, false, false>,
    ColumnAvailable<false, false, false, false>,
    ColumnAvailable<false, false, false, false>,
    ColumnAvailable<false, false, false, false>,
    ColumnAvailable<false, false, false, false>,
    ColumnAvailable<false, false, false, false>,
    ColumnAvailable<false, false, false, false>,
    ColumnAvailable<false, false, false, false>
>;

template <typename T>
using State_Hessian_xx_Type = SparseMatrix_Type<T, State_Hessian_xx_Type_SparseAvailable>;

template <typename T, typename X_Type, typename U_Type, typename Parameter_Type>
class Function {
public:
    static inline auto sympy_function() -> State_Hessian_xx_Type<T> {

        State_Hessian_xx_Type<T> result;

        result.template set<0, 0>(static_cast<T>(0));
        result.template set<0, 1>(static_cast<T>(0));
        result.template set<0, 2>(static_cast<T>(0));
        result.template set<0, 3>(static_cast<T>(0));
        result.template set<1, 0>(static_cast<T>(0));
        result.template set<1, 1>(static_cast<T>(0));
        result.template set<1, 2>(static_cast<T>(0));
        result.template set<1, 3>(static_cast<T>(0));
        result.template set<2, 0>(static_cast<T>(0));
        result.template set<2, 1>(static_cast<T>(0));
        result.template set<2, 2>(static_cast<T>(0));
        result.template set<2, 3>(static_cast<T>(0));
        result.template set<3, 0>(static_cast<T>(0));
        result.template set<3, 1>(static_cast<T>(0));
        result.template set<3, 2>(static_cast<T>(0));
        result.template set<3, 3>(static_cast<T>(0));
        result.template set<4, 0>(static_cast<T>(0));
        result.template set<4, 1>(static_cast<T>(0));
        result.template set<4, 2>(static_cast<T>(0));
        result.template set<4, 3>(static_cast<T>(0));
        result.template set<5, 0>(static_cast<T>(0));
        result.template set<5, 1>(static_cast<T>(0));
        result.template set<5, 2>(static_cast<T>(0));
        result.template set<5, 3>(static_cast<T>(0));
        result.template set<6, 0>(static_cast<T>(0));
        result.template set<6, 1>(static_cast<T>(0));
        result.template set<6, 2>(static_cast<T>(0));
        result.template set<6, 3>(static_cast<T>(0));
        result.template set<7, 0>(static_cast<T>(0));
        result.template set<7, 1>(static_cast<T>(0));
        result.template set<7, 2>(static_cast<T>(0));
        result.template set<7, 3>(static_cast<T>(0));
        result.template set<8, 0>(static_cast<T>(0));
        result.template set<8, 1>(static_cast<T>(0));
        result.template set<8, 2>(static_cast<T>(0));
        result.template set<8, 3>(static_cast<T>(0));
        result.template set<9, 0>(static_cast<T>(0));
        result.template set<9, 1>(static_cast<T>(0));
        result.template set<9, 2>(static_cast<T>(0));
        result.template set<9, 3>(static_cast<T>(0));
        result.template set<10, 0>(static_cast<T>(0));
        result.template set<10, 1>(static_cast<T>(0));
        result.template set<10, 2>(static_cast<T>(0));
        result.template set<10, 3>(static_cast<T>(0));
        result.template set<11, 0>(static_cast<T>(0));
        result.template set<11, 1>(static_cast<T>(0));
        result.template set<11, 2>(static_cast<T>(0));
        result.template set<11, 3>(static_cast<T>(0));
        result.template set<12, 0>(static_cast<T>(0));
        result.template set<12, 1>(static_cast<T>(0));
        result.template set<12, 2>(static_cast<T>(0));
        result.template set<12, 3>(static_cast<T>(0));
        result.template set<13, 0>(static_cast<T>(0));
        result.template set<13, 1>(static_cast<T>(0));
        result.template set<13, 2>(static_cast<T>(0));
        result.template set<13, 3>(static_cast<T>(0));
        result.template set<14, 0>(static_cast<T>(0));
        result.template set<14, 1>(static_cast<T>(0));
        result.template set<14, 2>(static_cast<T>(0));
        result.template set<14, 3>(static_cast<T>(0));
        result.template set<15, 0>(static_cast<T>(0));
        result.template set<15, 1>(static_cast<T>(0));
        result.template set<15, 2>(static_cast<T>(0));
        result.template set<15, 3>(static_cast<T>(0));

        return result;
    }

    static inline auto function(const X_Type X, const U_Type U, const Parameter_Type Parameters) -> State_Hessian_xx_Type<T> {
        static_cast<void>(X);
        static_cast<void>(U);
        static_cast<void>(Parameters);

        return sympy_function();
    }

};

} // namespace sqp_2_mass_spring_damper_demo_sqp_hessian_f_xx

namespace sqp_2_mass_spring_damper_demo_sqp_hessian_f_xu {

using namespace PythonMath;
using namespace PythonNumpy;

using State_Hessian_xu_Type_SparseAvailable = SparseAvailable<
    ColumnAvailable<false, false>,
    ColumnAvailable<false, false>,
    ColumnAvailable<false, false>,
    ColumnAvailable<false, false>,
    ColumnAvailable<false, false>,
    ColumnAvailable<false, false>,
    ColumnAvailable<false, false>,
    ColumnAvailable<false, false>,
    ColumnAvailable<false, false>,
    ColumnAvailable<false, false>,
    ColumnAvailable<false, false>,
    ColumnAvailable<false, false>,
    ColumnAvailable<false, false>,
    ColumnAvailable<false, false>,
    ColumnAvailable<false, false>,
    ColumnAvailable<false, false>
>;

template <typename T>
using State_Hessian_xu_Type = SparseMatrix_Type<T, State_Hessian_xu_Type_SparseAvailable>;

template <typename T, typename X_Type, typename U_Type, typename Parameter_Type>
class Function {
public:
    static inline auto sympy_function() -> State_Hessian_xu_Type<T> {

        State_Hessian_xu_Type<T> result;

        result.template set<0, 0>(static_cast<T>(0));
        result.template set<0, 1>(static_cast<T>(0));
        result.template set<1, 0>(static_cast<T>(0));
        result.template set<1, 1>(static_cast<T>(0));
        result.template set<2, 0>(static_cast<T>(0));
        result.template set<2, 1>(static_cast<T>(0));
        result.template set<3, 0>(static_cast<T>(0));
        result.template set<3, 1>(static_cast<T>(0));
        result.template set<4, 0>(static_cast<T>(0));
        result.template set<4, 1>(static_cast<T>(0));
        result.template set<5, 0>(static_cast<T>(0));
        result.template set<5, 1>(static_cast<T>(0));
        result.template set<6, 0>(static_cast<T>(0));
        result.template set<6, 1>(static_cast<T>(0));
        result.template set<7, 0>(static_cast<T>(0));
        result.template set<7, 1>(static_cast<T>(0));
        result.template set<8, 0>(static_cast<T>(0));
        result.template set<8, 1>(static_cast<T>(0));
        result.template set<9, 0>(static_cast<T>(0));
        result.template set<9, 1>(static_cast<T>(0));
        result.template set<10, 0>(static_cast<T>(0));
        result.template set<10, 1>(static_cast<T>(0));
        result.template set<11, 0>(static_cast<T>(0));
        result.template set<11, 1>(static_cast<T>(0));
        result.template set<12, 0>(static_cast<T>(0));
        result.template set<12, 1>(static_cast<T>(0));
        result.template set<13, 0>(static_cast<T>(0));
        result.template set<13, 1>(static_cast<T>(0));
        result.template set<14, 0>(static_cast<T>(0));
        result.template set<14, 1>(static_cast<T>(0));
        result.template set<15, 0>(static_cast<T>(0));
        result.template set<15, 1>(static_cast<T>(0));

        return result;
    }

    static inline auto function(const X_Type X, const U_Type U, const Parameter_Type Parameters) -> State_Hessian_xu_Type<T> {
        static_cast<void>(X);
        static_cast<void>(U);
        static_cast<void>(Parameters);

        return sympy_function();
    }

};

} // namespace sqp_2_mass_spring_damper_demo_sqp_hessian_f_xu

namespace sqp_2_mass_spring_damper_demo_sqp_hessian_f_ux {

using namespace PythonMath;
using namespace PythonNumpy;

using State_Hessian_ux_Type_SparseAvailable = SparseAvailable<
    ColumnAvailable<false, false, false, false>,
    ColumnAvailable<false, false, false, false>,
    ColumnAvailable<false, false, false, false>,
    ColumnAvailable<false, false, false, false>,
    ColumnAvailable<false, false, false, false>,
    ColumnAvailable<false, false, false, false>,
    ColumnAvailable<false, false, false, false>,
    ColumnAvailable<false, false, false, false>
>;

template <typename T>
using State_Hessian_ux_Type = SparseMatrix_Type<T, State_Hessian_ux_Type_SparseAvailable>;

template <typename T, typename X_Type, typename U_Type, typename Parameter_Type>
class Function {
public:
    static inline auto sympy_function() -> State_Hessian_ux_Type<T> {

        State_Hessian_ux_Type<T> result;

        result.template set<0, 0>(static_cast<T>(0));
        result.template set<0, 1>(static_cast<T>(0));
        result.template set<0, 2>(static_cast<T>(0));
        result.template set<0, 3>(static_cast<T>(0));
        result.template set<1, 0>(static_cast<T>(0));
        result.template set<1, 1>(static_cast<T>(0));
        result.template set<1, 2>(static_cast<T>(0));
        result.template set<1, 3>(static_cast<T>(0));
        result.template set<2, 0>(static_cast<T>(0));
        result.template set<2, 1>(static_cast<T>(0));
        result.template set<2, 2>(static_cast<T>(0));
        result.template set<2, 3>(static_cast<T>(0));
        result.template set<3, 0>(static_cast<T>(0));
        result.template set<3, 1>(static_cast<T>(0));
        result.template set<3, 2>(static_cast<T>(0));
        result.template set<3, 3>(static_cast<T>(0));
        result.template set<4, 0>(static_cast<T>(0));
        result.template set<4, 1>(static_cast<T>(0));
        result.template set<4, 2>(static_cast<T>(0));
        result.template set<4, 3>(static_cast<T>(0));
        result.template set<5, 0>(static_cast<T>(0));
        result.template set<5, 1>(static_cast<T>(0));
        result.template set<5, 2>(static_cast<T>(0));
        result.template set<5, 3>(static_cast<T>(0));
        result.template set<6, 0>(static_cast<T>(0));
        result.template set<6, 1>(static_cast<T>(0));
        result.template set<6, 2>(static_cast<T>(0));
        result.template set<6, 3>(static_cast<T>(0));
        result.template set<7, 0>(static_cast<T>(0));
        result.template set<7, 1>(static_cast<T>(0));
        result.template set<7, 2>(static_cast<T>(0));
        result.template set<7, 3>(static_cast<T>(0));

        return result;
    }

    static inline auto function(const X_Type X, const U_Type U, const Parameter_Type Parameters) -> State_Hessian_ux_Type<T> {
        static_cast<void>(X);
        static_cast<void>(U);
        static_cast<void>(Parameters);

        return sympy_function();
    }

};

} // namespace sqp_2_mass_spring_damper_demo_sqp_hessian_f_ux

namespace sqp_2_mass_spring_damper_demo_sqp_hessian_f_uu {

using namespace PythonMath;
using namespace PythonNumpy;

using State_Hessian_uu_Type_SparseAvailable = SparseAvailable<
    ColumnAvailable<false, false>,
    ColumnAvailable<false, false>,
    ColumnAvailable<false, false>,
    ColumnAvailable<false, false>,
    ColumnAvailable<false, false>,
    ColumnAvailable<false, false>,
    ColumnAvailable<false, false>,
    ColumnAvailable<false, false>
>;

template <typename T>
using State_Hessian_uu_Type = SparseMatrix_Type<T, State_Hessian_uu_Type_SparseAvailable>;

template <typename T, typename X_Type, typename U_Type, typename Parameter_Type>
class Function {
public:
    static inline auto sympy_function() -> State_Hessian_uu_Type<T> {

        State_Hessian_uu_Type<T> result;

        result.template set<0, 0>(static_cast<T>(0));
        result.template set<0, 1>(static_cast<T>(0));
        result.template set<1, 0>(static_cast<T>(0));
        result.template set<1, 1>(static_cast<T>(0));
        result.template set<2, 0>(static_cast<T>(0));
        result.template set<2, 1>(static_cast<T>(0));
        result.template set<3, 0>(static_cast<T>(0));
        result.template set<3, 1>(static_cast<T>(0));
        result.template set<4, 0>(static_cast<T>(0));
        result.template set<4, 1>(static_cast<T>(0));
        result.template set<5, 0>(static_cast<T>(0));
        result.template set<5, 1>(static_cast<T>(0));
        result.template set<6, 0>(static_cast<T>(0));
        result.template set<6, 1>(static_cast<T>(0));
        result.template set<7, 0>(static_cast<T>(0));
        result.template set<7, 1>(static_cast<T>(0));

        return result;
    }

    static inline auto function(const X_Type X, const U_Type U, const Parameter_Type Parameters) -> State_Hessian_uu_Type<T> {
        static_cast<void>(X);
        static_cast<void>(U);
        static_cast<void>(Parameters);

        return sympy_function();
    }

};

} // namespace sqp_2_mass_spring_damper_demo_sqp_hessian_f_uu

namespace sqp_2_mass_spring_damper_demo_sqp_hessian_h_xx {

using namespace PythonMath;
using namespace PythonNumpy;

using Measurement_Hessian_xx_Type_SparseAvailable = SparseAvailable<
    ColumnAvailable<false, false, false, false>,
    ColumnAvailable<false, false, false, false>,
    ColumnAvailable<false, false, false, false>,
    ColumnAvailable<false, false, false, false>,
    ColumnAvailable<false, false, false, false>,
    ColumnAvailable<false, false, false, false>,
    ColumnAvailable<false, false, false, false>,
    ColumnAvailable<false, false, false, false>
>;

template <typename T>
using Measurement_Hessian_xx_Type = SparseMatrix_Type<T, Measurement_Hessian_xx_Type_SparseAvailable>;

template <typename T, typename X_Type, typename U_Type, typename Parameter_Type>
class Function {
public:
    static inline auto sympy_function() -> Measurement_Hessian_xx_Type<T> {

        Measurement_Hessian_xx_Type<T> result;

        result.template set<0, 0>(static_cast<T>(0));
        result.template set<0, 1>(static_cast<T>(0));
        result.template set<0, 2>(static_cast<T>(0));
        result.template set<0, 3>(static_cast<T>(0));
        result.template set<1, 0>(static_cast<T>(0));
        result.template set<1, 1>(static_cast<T>(0));
        result.template set<1, 2>(static_cast<T>(0));
        result.template set<1, 3>(static_cast<T>(0));
        result.template set<2, 0>(static_cast<T>(0));
        result.template set<2, 1>(static_cast<T>(0));
        result.template set<2, 2>(static_cast<T>(0));
        result.template set<2, 3>(static_cast<T>(0));
        result.template set<3, 0>(static_cast<T>(0));
        result.template set<3, 1>(static_cast<T>(0));
        result.template set<3, 2>(static_cast<T>(0));
        result.template set<3, 3>(static_cast<T>(0));
        result.template set<4, 0>(static_cast<T>(0));
        result.template set<4, 1>(static_cast<T>(0));
        result.template set<4, 2>(static_cast<T>(0));
        result.template set<4, 3>(static_cast<T>(0));
        result.template set<5, 0>(static_cast<T>(0));
        result.template set<5, 1>(static_cast<T>(0));
        result.template set<5, 2>(static_cast<T>(0));
        result.template set<5, 3>(static_cast<T>(0));
        result.template set<6, 0>(static_cast<T>(0));
        result.template set<6, 1>(static_cast<T>(0));
        result.template set<6, 2>(static_cast<T>(0));
        result.template set<6, 3>(static_cast<T>(0));
        result.template set<7, 0>(static_cast<T>(0));
        result.template set<7, 1>(static_cast<T>(0));
        result.template set<7, 2>(static_cast<T>(0));
        result.template set<7, 3>(static_cast<T>(0));

        return result;
    }

    static inline auto function(const X_Type X, const U_Type U, const Parameter_Type Parameters) -> Measurement_Hessian_xx_Type<T> {
        static_cast<void>(X);
        static_cast<void>(U);
        static_cast<void>(Parameters);

        return sympy_function();
    }

};

} // namespace sqp_2_mass_spring_damper_demo_sqp_hessian_h_xx


} // namespace SQP_TestData

#endif // __SQP_TEST_DATA_HPP__
