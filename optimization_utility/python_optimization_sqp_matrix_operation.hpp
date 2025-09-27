#ifndef __PYTHON_OPTIMIZATION_SQP_MATRIX_OPERATION_HPP__
#define __PYTHON_OPTIMIZATION_SQP_MATRIX_OPERATION_HPP__

#include "python_control.hpp"
#include "python_numpy.hpp"

#include <array>
#include <functional>
#include <type_traits>

namespace PythonOptimization {

namespace MatrixOperation {

template <typename Matrix_Out_Type, typename Matrix_In_Type>
inline void set_row(Matrix_Out_Type &out_matrix,
                    const Matrix_In_Type &in_matrix,
                    const std::size_t &row_index) {

  static_assert(Matrix_Out_Type::COLS == Matrix_In_Type::COLS,
                "Matrix_Out_Type::COLS != Matrix_In_Type::COLS");
  static_assert(Matrix_In_Type::ROWS == 1, "Matrix_In_Type::ROWS != 1");

  for (std::size_t i = 0; i < Matrix_In_Type::ROWS; i++) {
    out_matrix(i, row_index) = in_matrix(i, 0);
  }
}

template <typename Matrix_In_Type>
inline auto get_row(const Matrix_In_Type &in_matrix,
                    const std::size_t &row_index)
    -> PythonNumpy::DenseMatrix_Type<typename Matrix_In_Type::Value_Type,
                                     Matrix_In_Type::COLS, 1> {

  PythonNumpy::DenseMatrix_Type<typename Matrix_In_Type::Value_Type,
                                Matrix_In_Type::COLS, 1>
      out;

  for (std::size_t i = 0; i < Matrix_In_Type::COLS; i++) {
    out(i, 0) = in_matrix(i, row_index);
  }

  return out;
}

template <typename Matrix_A_Type, typename Matrix_B_Type>
inline auto element_wise_multiply(const Matrix_A_Type &A,
                                  const Matrix_B_Type &B)
    -> PythonNumpy::DenseMatrix_Type<typename Matrix_A_Type::Value_Type,
                                     Matrix_A_Type::COLS, Matrix_A_Type::ROWS> {

  static_assert(Matrix_A_Type::COLS == Matrix_B_Type::COLS,
                "Matrix_A_Type::COLS != Matrix_B_Type::COLS");
  static_assert(Matrix_A_Type::ROWS == Matrix_B_Type::ROWS,
                "Matrix_A_Type::ROWS != Matrix_B_Type::ROWS");

  PythonNumpy::DenseMatrix_Type<typename Matrix_A_Type::Value_Type,
                                Matrix_A_Type::COLS, Matrix_A_Type::ROWS>
      out;

  for (std::size_t i = 0; i < Matrix_A_Type::COLS; i++) {
    for (std::size_t j = 0; j < Matrix_A_Type::ROWS; j++) {
      out(i, j) = A(i, j) * B(i, j);
    }
  }

  return out;
}

template <typename X_Type, typename W_Type>
inline auto calculate_quadratic_form(const X_Type &X, const W_Type &W) ->
    typename X_Type::Value_Type {

  static_assert(W_Type::ROWS == X_Type::COLS, "W_Type::ROWS != X_Type::COLS");
  static_assert(X_Type::ROWS == 1, "X_Type::ROWS != 1");

  auto result = PythonNumpy::ATranspose_mul_B(X, W) * X;

  return result.template get<0, 0>();
}

template <typename X_Type>
inline auto calculate_quadratic_no_weighted(const X_Type &X) ->
    typename X_Type::Value_Type {

  static_assert(X_Type::ROWS == 1, "X_Type::ROWS != 1");

  auto result = PythonNumpy::ATranspose_mul_B(X, X);

  return result.template get<0, 0>();
}

/* calculate Y_limit penalty */

namespace CalculateY_LimitPenalty {

// Column recursion when J_idx > 0
template <typename Y_Mat_Type, typename Y_Min_Matrix_Type,
          typename Y_Max_Matrix_Type, typename Out_Type, std::size_t M,
          std::size_t N, std::size_t I, std::size_t J_idx>
struct Column {
  static void compute(const Y_Mat_Type &Y_horizon,
                      const Y_Min_Matrix_Type &Y_min_matrix,
                      const Y_Max_Matrix_Type &Y_max_matrix,
                      Out_Type &Y_limit_penalty) {

    const auto y = Y_horizon.template get<I, J_idx>();
    const auto y_min = Y_min_matrix.template get<I, J_idx>();
    const auto y_max = Y_max_matrix.template get<I, J_idx>();

    if (y < y_min) {
      Y_limit_penalty.template set<I, J_idx>(y - y_min);
    } else if (y > y_max) {
      Y_limit_penalty.template set<I, J_idx>(y - y_max);
    } else {
      /* Do Nothing */
    }

    Column<Y_Mat_Type, Y_Min_Matrix_Type, Y_Max_Matrix_Type, Out_Type, M, N, I,
           (J_idx - 1)>::compute(Y_horizon, Y_min_matrix, Y_max_matrix,
                                 Y_limit_penalty);
  }
};

// Column recursion termination for J_idx == 0
template <typename Y_Mat_Type, typename Y_Min_Matrix_Type,
          typename Y_Max_Matrix_Type, typename Out_Type, std::size_t M,
          std::size_t N, std::size_t I>
struct Column<Y_Mat_Type, Y_Min_Matrix_Type, Y_Max_Matrix_Type, Out_Type, M, N,
              I, 0> {
  static void compute(const Y_Mat_Type &Y_horizon,
                      const Y_Min_Matrix_Type &Y_min_matrix,
                      const Y_Max_Matrix_Type &Y_max_matrix,
                      Out_Type &Y_limit_penalty) {

    const auto y = Y_horizon.template get<I, 0>();
    const auto y_min = Y_min_matrix.template get<I, 0>();
    const auto y_max = Y_max_matrix.template get<I, 0>();

    if (y < y_min) {
      Y_limit_penalty.template set<I, 0>(y - y_min);
    } else if (y > y_max) {
      Y_limit_penalty.template set<I, 0>(y - y_max);
    } else {
      /* Do Nothing */
    }
  }
};

// Row recursion when I_idx > 0
template <typename Y_Mat_Type, typename Y_Min_Matrix_Type,
          typename Y_Max_Matrix_Type, typename Out_Type, std::size_t M,
          std::size_t N, std::size_t I_idx>
struct Row {
  static void compute(const Y_Mat_Type &Y_horizon,
                      const Y_Min_Matrix_Type &Y_min_matrix,
                      const Y_Max_Matrix_Type &Y_max_matrix,
                      Out_Type &Y_limit_penalty) {
    Column<Y_Mat_Type, Y_Min_Matrix_Type, Y_Max_Matrix_Type, Out_Type, M, N,
           I_idx, (N - 1)>::compute(Y_horizon, Y_min_matrix, Y_max_matrix,
                                    Y_limit_penalty);
    Row<Y_Mat_Type, Y_Min_Matrix_Type, Y_Max_Matrix_Type, Out_Type, M, N,
        (I_idx - 1)>::compute(Y_horizon, Y_min_matrix, Y_max_matrix,
                              Y_limit_penalty);
  }
};

// Row recursion termination for I_idx == 0
template <typename Y_Mat_Type, typename Y_Min_Matrix_Type,
          typename Y_Max_Matrix_Type, typename Out_Type, std::size_t M,
          std::size_t N>
struct Row<Y_Mat_Type, Y_Min_Matrix_Type, Y_Max_Matrix_Type, Out_Type, M, N,
           0> {
  static void compute(const Y_Mat_Type &Y_horizon,
                      const Y_Min_Matrix_Type &Y_min_matrix,
                      const Y_Max_Matrix_Type &Y_max_matrix,
                      Out_Type &Y_limit_penalty) {
    Column<Y_Mat_Type, Y_Min_Matrix_Type, Y_Max_Matrix_Type, Out_Type, M, N, 0,
           (N - 1)>::compute(Y_horizon, Y_min_matrix, Y_max_matrix,
                             Y_limit_penalty);
  }
};

} // namespace CalculateY_LimitPenalty

template <typename Y_Mat_Type, typename Y_Min_Matrix_Type,
          typename Y_Max_Matrix_Type, typename Out_Type>
inline void calculate_Y_limit_penalty(const Y_Mat_Type &Y_horizon,
                                      const Y_Min_Matrix_Type &Y_min_matrix,
                                      const Y_Max_Matrix_Type &Y_max_matrix,
                                      Out_Type &Y_limit_penalty) {

  constexpr std::size_t M = Out_Type::COLS;
  constexpr std::size_t N = Out_Type::ROWS;
  static_assert(M > 0 && N > 0, "Matrix dimensions must be positive");

  CalculateY_LimitPenalty::Row<Y_Mat_Type, Y_Min_Matrix_Type, Y_Max_Matrix_Type,
                               Out_Type, M, N,
                               (M - 1)>::compute(Y_horizon, Y_min_matrix,
                                                 Y_max_matrix, Y_limit_penalty);
}

/* calculate Y_limit penalty and active */

namespace CalculateY_LimitPenaltyAndActive {

// Column recursion when J_idx > 0
template <typename Y_Mat_Type, typename Y_Min_Matrix_Type,
          typename Y_Max_Matrix_Type, typename Out_Penalty_Type,
          typename Active_Type, std::size_t M, std::size_t N, std::size_t I,
          std::size_t J_idx>
struct Column {
  static void compute(const Y_Mat_Type &Y_horizon,
                      const Y_Min_Matrix_Type &Y_min_matrix,
                      const Y_Max_Matrix_Type &Y_max_matrix,
                      Out_Penalty_Type &Y_limit_penalty,
                      Active_Type &Y_limit_active) {

    const auto y = Y_horizon.template get<I, J_idx>();
    const auto y_min = Y_min_matrix.template get<I, J_idx>();
    const auto y_max = Y_max_matrix.template get<I, J_idx>();

    if (y < y_min) {
      Y_limit_penalty.template set<I, J_idx>(y - y_min);
      Y_limit_active.template set<I, J_idx>(
          static_cast<typename Active_Type::Value_Type>(1));
    } else if (y > y_max) {
      Y_limit_penalty.template set<I, J_idx>(y - y_max);
      Y_limit_active.template set<I, J_idx>(
          static_cast<typename Active_Type::Value_Type>(1));
    } else {
      /* Do Nothing */
    }

    Column<Y_Mat_Type, Y_Min_Matrix_Type, Y_Max_Matrix_Type, Out_Penalty_Type,
           Active_Type, M, N, I, (J_idx - 1)>::compute(Y_horizon, Y_min_matrix,
                                                       Y_max_matrix,
                                                       Y_limit_penalty,
                                                       Y_limit_active);
  }
};

// Column recursion termination for J_idx == 0
template <typename Y_Mat_Type, typename Y_Min_Matrix_Type,
          typename Y_Max_Matrix_Type, typename Out_Penalty_Type,
          typename Active_Type, std::size_t M, std::size_t N, std::size_t I>
struct Column<Y_Mat_Type, Y_Min_Matrix_Type, Y_Max_Matrix_Type,
              Out_Penalty_Type, Active_Type, M, N, I, 0> {
  static void compute(const Y_Mat_Type &Y_horizon,
                      const Y_Min_Matrix_Type &Y_min_matrix,
                      const Y_Max_Matrix_Type &Y_max_matrix,
                      Out_Penalty_Type &Y_limit_penalty,
                      Active_Type &Y_limit_active) {

    const auto y = Y_horizon.template get<I, 0>();
    const auto y_min = Y_min_matrix.template get<I, 0>();
    const auto y_max = Y_max_matrix.template get<I, 0>();

    if (y < y_min) {
      Y_limit_penalty.template set<I, 0>(y - y_min);
      Y_limit_active.template set<I, 0>(
          static_cast<typename Active_Type::Value_Type>(1));
    } else if (y > y_max) {
      Y_limit_penalty.template set<I, 0>(y - y_max);
      Y_limit_active.template set<I, 0>(
          static_cast<typename Active_Type::Value_Type>(1));
    } else {
      /* Do Nothing */
    }
  }
};

// Row recursion when I_idx > 0
template <typename Y_Mat_Type, typename Y_Min_Matrix_Type,
          typename Y_Max_Matrix_Type, typename Out_Penalty_Type,
          typename Active_Type, std::size_t M, std::size_t N, std::size_t I_idx>
struct Row {
  static void compute(const Y_Mat_Type &Y_horizon,
                      const Y_Min_Matrix_Type &Y_min_matrix,
                      const Y_Max_Matrix_Type &Y_max_matrix,
                      Out_Penalty_Type &Y_limit_penalty,
                      Active_Type &Y_limit_active) {
    Column<Y_Mat_Type, Y_Min_Matrix_Type, Y_Max_Matrix_Type, Out_Penalty_Type,
           Active_Type, M, N, I_idx, (N - 1)>::compute(Y_horizon, Y_min_matrix,
                                                       Y_max_matrix,
                                                       Y_limit_penalty,
                                                       Y_limit_active);
    Row<Y_Mat_Type, Y_Min_Matrix_Type, Y_Max_Matrix_Type, Out_Penalty_Type,
        Active_Type, M, N, (I_idx - 1)>::compute(Y_horizon, Y_min_matrix,
                                                 Y_max_matrix, Y_limit_penalty,
                                                 Y_limit_active);
  }
};

// Row recursion termination for I_idx == 0
template <typename Y_Mat_Type, typename Y_Min_Matrix_Type,
          typename Y_Max_Matrix_Type, typename Out_Penalty_Type,
          typename Active_Type, std::size_t M, std::size_t N>
struct Row<Y_Mat_Type, Y_Min_Matrix_Type, Y_Max_Matrix_Type, Out_Penalty_Type,
           Active_Type, M, N, 0> {
  static void compute(const Y_Mat_Type &Y_horizon,
                      const Y_Min_Matrix_Type &Y_min_matrix,
                      const Y_Max_Matrix_Type &Y_max_matrix,
                      Out_Penalty_Type &Y_limit_penalty,
                      Active_Type &Y_limit_active) {
    Column<Y_Mat_Type, Y_Min_Matrix_Type, Y_Max_Matrix_Type, Out_Penalty_Type,
           Active_Type, M, N, 0, (N - 1)>::compute(Y_horizon, Y_min_matrix,
                                                   Y_max_matrix,
                                                   Y_limit_penalty,
                                                   Y_limit_active);
  }
};

} // namespace CalculateY_LimitPenaltyAndActive

template <typename Y_Mat_Type, typename Y_Min_Matrix_Type,
          typename Y_Max_Matrix_Type, typename Out_Penalty_Type,
          typename Active_Type>
inline void calculate_Y_limit_penalty_and_active(
    const Y_Mat_Type &Y_horizon, const Y_Min_Matrix_Type &Y_min_matrix,
    const Y_Max_Matrix_Type &Y_max_matrix, Out_Penalty_Type &Y_limit_penalty,
    Active_Type &Y_limit_active) {

  constexpr std::size_t M = Out_Penalty_Type::COLS;
  constexpr std::size_t N = Out_Penalty_Type::ROWS;
  static_assert(M > 0 && N > 0, "Matrix dimensions must be positive");
  static_assert(Out_Penalty_Type::COLS == Active_Type::COLS,
                "Penalty and Active COLS mismatch");
  static_assert(Out_Penalty_Type::ROWS == Active_Type::ROWS,
                "Penalty and Active ROWS mismatch");

  CalculateY_LimitPenaltyAndActive::Row<
      Y_Mat_Type, Y_Min_Matrix_Type, Y_Max_Matrix_Type, Out_Penalty_Type,
      Active_Type, M, N, (M - 1)>::compute(Y_horizon, Y_min_matrix,
                                           Y_max_matrix, Y_limit_penalty,
                                           Y_limit_active);
}

/* fx xx lambda contract */

namespace FxxLambdaContract {

// K-accumulation: recursively accumulate over k (STATE_SIZE dimension)
template <typename Fxx_Type, typename dX_Type, typename Value_Type,
          std::size_t OUTPUT_SIZE, std::size_t STATE_SIZE, std::size_t I,
          std::size_t J, std::size_t K_idx>
struct AccumulateK {
  static void compute(const Fxx_Type &Hf_xx, const dX_Type &dX,
                      Value_Type &acc) {
    acc += Hf_xx.template get<I * STATE_SIZE + J, K_idx>() *
           dX.template get<K_idx, 0>();
    AccumulateK<Fxx_Type, dX_Type, Value_Type, OUTPUT_SIZE, STATE_SIZE, I, J,
                (K_idx - 1)>::compute(Hf_xx, dX, acc);
  }
};

// K-accumulation termination (K_idx == 0)
template <typename Fxx_Type, typename dX_Type, typename Value_Type,
          std::size_t OUTPUT_SIZE, std::size_t STATE_SIZE, std::size_t I,
          std::size_t J>
struct AccumulateK<Fxx_Type, dX_Type, Value_Type, OUTPUT_SIZE, STATE_SIZE, I, J,
                   0> {
  static void compute(const Fxx_Type &Hf_xx, const dX_Type &dX,
                      Value_Type &acc) {
    acc +=
        Hf_xx.template get<I * STATE_SIZE + J, 0>() * dX.template get<0, 0>();
  }
};

// Column recursion over j (STATE_SIZE): computes contribution to out(j,0)
template <typename Fxx_Type, typename dX_Type, typename Weight_Type,
          typename Out_Type, std::size_t OUTPUT_SIZE, std::size_t STATE_SIZE,
          std::size_t I, std::size_t J_idx>
struct Column {
  static void compute(const Fxx_Type &Hf_xx, const dX_Type &dX,
                      const Weight_Type &lam_next, Out_Type &out) {
    using Value = typename Out_Type::Value_Type;
    Value acc = static_cast<Value>(0);
    AccumulateK<Fxx_Type, dX_Type, Value, OUTPUT_SIZE, STATE_SIZE, I, J_idx,
                (STATE_SIZE - 1)>::compute(Hf_xx, dX, acc);

    const auto w = lam_next.template get<I, 0>();
    const auto updated = out.template get<J_idx, 0>() + w * acc;
    out.template set<J_idx, 0>(updated);

    Column<Fxx_Type, dX_Type, Weight_Type, Out_Type, OUTPUT_SIZE, STATE_SIZE, I,
           (J_idx - 1)>::compute(Hf_xx, dX, lam_next, out);
  }
};

// Column recursion termination for j == 0
template <typename Fxx_Type, typename dX_Type, typename Weight_Type,
          typename Out_Type, std::size_t OUTPUT_SIZE, std::size_t STATE_SIZE,
          std::size_t I>
struct Column<Fxx_Type, dX_Type, Weight_Type, Out_Type, OUTPUT_SIZE, STATE_SIZE,
              I, 0> {
  static void compute(const Fxx_Type &Hf_xx, const dX_Type &dX,
                      const Weight_Type &lam_next, Out_Type &out) {
    using Value = typename Out_Type::Value_Type;
    Value acc = static_cast<Value>(0);
    AccumulateK<Fxx_Type, dX_Type, Value, OUTPUT_SIZE, STATE_SIZE, I, 0,
                (STATE_SIZE - 1)>::compute(Hf_xx, dX, acc);

    const auto w = lam_next.template get<I, 0>();
    const auto updated = out.template get<0, 0>() + w * acc;
    out.template set<0, 0>(updated);
  }
};

// Row recursion over i (OUTPUT_SIZE): iterates outputs
template <typename Fxx_Type, typename dX_Type, typename Weight_Type,
          typename Out_Type, std::size_t OUTPUT_SIZE, std::size_t STATE_SIZE,
          std::size_t I_idx>
struct Row {
  static void compute(const Fxx_Type &Hf_xx, const dX_Type &dX,
                      const Weight_Type &lam_next, Out_Type &out) {
    Column<Fxx_Type, dX_Type, Weight_Type, Out_Type, OUTPUT_SIZE, STATE_SIZE,
           I_idx, (STATE_SIZE - 1)>::compute(Hf_xx, dX, lam_next, out);
    Row<Fxx_Type, dX_Type, Weight_Type, Out_Type, OUTPUT_SIZE, STATE_SIZE,
        (I_idx - 1)>::compute(Hf_xx, dX, lam_next, out);
  }
};

// Row recursion termination for i == 0
template <typename Fxx_Type, typename dX_Type, typename Weight_Type,
          typename Out_Type, std::size_t OUTPUT_SIZE, std::size_t STATE_SIZE>
struct Row<Fxx_Type, dX_Type, Weight_Type, Out_Type, OUTPUT_SIZE, STATE_SIZE,
           0> {
  static void compute(const Fxx_Type &Hf_xx, const dX_Type &dX,
                      const Weight_Type &lam_next, Out_Type &out) {
    Column<Fxx_Type, dX_Type, Weight_Type, Out_Type, OUTPUT_SIZE, STATE_SIZE, 0,
           (STATE_SIZE - 1)>::compute(Hf_xx, dX, lam_next, out);
  }
};

} // namespace FxxLambdaContract

// Public wrapper to run the unrolled recursion.
// Out_Type must be pre-initialized (e.g., zero) before calling, since this
// performs accumulation with "+=" semantics.
template <typename Fxx_Type, typename dX_Type, typename Weight_Type,
          typename Out_Type>
inline void
compute_fxx_lambda_contract(const Fxx_Type &Hf_xx, const dX_Type &dX,
                            const Weight_Type &lam_next, Out_Type &out) {
  static_assert(dX_Type::ROWS == 1, "dX must be a (STATE_SIZE x 1) vector");
  static_assert(Weight_Type::ROWS == 1,
                "lam_next must be a (OUTPUT_SIZE x 1) vector");
  static_assert(Out_Type::ROWS == 1,
                "out must be a (STATE_SIZE x 1) vector (ROWS == 1)");

  constexpr std::size_t STATE_SIZE = dX_Type::COLS;
  constexpr std::size_t OUTPUT_SIZE = Weight_Type::COLS;

  static_assert(STATE_SIZE > 0 && OUTPUT_SIZE > 0,
                "STATE_SIZE and OUTPUT_SIZE must be positive");
  static_assert(Fxx_Type::ROWS == STATE_SIZE,
                "Hf_xx ROWS must equal STATE_SIZE");
  static_assert(Fxx_Type::COLS == OUTPUT_SIZE * STATE_SIZE,
                "Hf_xx COLS must equal OUTPUT_SIZE * STATE_SIZE");
  static_assert(Out_Type::COLS == STATE_SIZE, "out COLS must equal STATE_SIZE");

  FxxLambdaContract::Row<Fxx_Type, dX_Type, Weight_Type, Out_Type, OUTPUT_SIZE,
                         STATE_SIZE, (OUTPUT_SIZE - 1)>::compute(Hf_xx, dX,
                                                                 lam_next, out);
}

/* fx xu lambda contract */

namespace FxuLambdaContract {

// K-accumulation: recursively accumulate over k (INPUT_SIZE dimension)
template <typename Fxu_Type, typename dU_Type, typename Value_Type,
          std::size_t OUTPUT_SIZE, std::size_t STATE_SIZE,
          std::size_t INPUT_SIZE, std::size_t I, std::size_t J,
          std::size_t K_idx>
struct AccumulateK {
  static void compute(const Fxu_Type &Hf_xu, const dU_Type &dU,
                      Value_Type &acc) {
    acc += Hf_xu.template get<I * STATE_SIZE + J, K_idx>() *
           dU.template get<K_idx, 0>();
    AccumulateK<Fxu_Type, dU_Type, Value_Type, OUTPUT_SIZE, STATE_SIZE,
                INPUT_SIZE, I, J, (K_idx - 1)>::compute(Hf_xu, dU, acc);
  }
};

// K-accumulation termination (K_idx == 0)
template <typename Fxu_Type, typename dU_Type, typename Value_Type,
          std::size_t OUTPUT_SIZE, std::size_t STATE_SIZE,
          std::size_t INPUT_SIZE, std::size_t I, std::size_t J>
struct AccumulateK<Fxu_Type, dU_Type, Value_Type, OUTPUT_SIZE, STATE_SIZE,
                   INPUT_SIZE, I, J, 0> {
  static void compute(const Fxu_Type &Hf_xu, const dU_Type &dU,
                      Value_Type &acc) {
    acc +=
        Hf_xu.template get<I * STATE_SIZE + J, 0>() * dU.template get<0, 0>();
  }
};

// Column recursion over j (STATE_SIZE): computes contribution to out(j,0)
template <typename Fxu_Type, typename dU_Type, typename Weight_Type,
          typename Out_Type, std::size_t OUTPUT_SIZE, std::size_t STATE_SIZE,
          std::size_t INPUT_SIZE, std::size_t I, std::size_t J_idx>
struct Column {
  static void compute(const Fxu_Type &Hf_xu, const dU_Type &dU,
                      const Weight_Type &lam_next, Out_Type &out) {
    using Value = typename Out_Type::Value_Type;
    Value acc = static_cast<Value>(0);
    AccumulateK<Fxu_Type, dU_Type, Value, OUTPUT_SIZE, STATE_SIZE, INPUT_SIZE,
                I, J_idx, (INPUT_SIZE - 1)>::compute(Hf_xu, dU, acc);

    const auto w = lam_next.template get<I, 0>();
    const auto updated = out.template get<J_idx, 0>() + w * acc;
    out.template set<J_idx, 0>(updated);

    Column<Fxu_Type, dU_Type, Weight_Type, Out_Type, OUTPUT_SIZE, STATE_SIZE,
           INPUT_SIZE, I, (J_idx - 1)>::compute(Hf_xu, dU, lam_next, out);
  }
};

// Column recursion termination for j == 0
template <typename Fxu_Type, typename dU_Type, typename Weight_Type,
          typename Out_Type, std::size_t OUTPUT_SIZE, std::size_t STATE_SIZE,
          std::size_t INPUT_SIZE, std::size_t I>
struct Column<Fxu_Type, dU_Type, Weight_Type, Out_Type, OUTPUT_SIZE, STATE_SIZE,
              INPUT_SIZE, I, 0> {
  static void compute(const Fxu_Type &Hf_xu, const dU_Type &dU,
                      const Weight_Type &lam_next, Out_Type &out) {
    using Value = typename Out_Type::Value_Type;
    Value acc = static_cast<Value>(0);
    AccumulateK<Fxu_Type, dU_Type, Value, OUTPUT_SIZE, STATE_SIZE, INPUT_SIZE,
                I, 0, (INPUT_SIZE - 1)>::compute(Hf_xu, dU, acc);

    const auto w = lam_next.template get<I, 0>();
    const auto updated = out.template get<0, 0>() + w * acc;
    out.template set<0, 0>(updated);
  }
};

// Row recursion over i (OUTPUT_SIZE): iterates outputs
template <typename Fxu_Type, typename dU_Type, typename Weight_Type,
          typename Out_Type, std::size_t OUTPUT_SIZE, std::size_t STATE_SIZE,
          std::size_t INPUT_SIZE, std::size_t I_idx>
struct Row {
  static void compute(const Fxu_Type &Hf_xu, const dU_Type &dU,
                      const Weight_Type &lam_next, Out_Type &out) {
    Column<Fxu_Type, dU_Type, Weight_Type, Out_Type, OUTPUT_SIZE, STATE_SIZE,
           INPUT_SIZE, I_idx, (STATE_SIZE - 1)>::compute(Hf_xu, dU, lam_next,
                                                         out);
    Row<Fxu_Type, dU_Type, Weight_Type, Out_Type, OUTPUT_SIZE, STATE_SIZE,
        INPUT_SIZE, (I_idx - 1)>::compute(Hf_xu, dU, lam_next, out);
  }
};

// Row recursion termination for i == 0
template <typename Fxu_Type, typename dU_Type, typename Weight_Type,
          typename Out_Type, std::size_t OUTPUT_SIZE, std::size_t STATE_SIZE,
          std::size_t INPUT_SIZE>
struct Row<Fxu_Type, dU_Type, Weight_Type, Out_Type, OUTPUT_SIZE, STATE_SIZE,
           INPUT_SIZE, 0> {
  static void compute(const Fxu_Type &Hf_xu, const dU_Type &dU,
                      const Weight_Type &lam_next, Out_Type &out) {
    Column<Fxu_Type, dU_Type, Weight_Type, Out_Type, OUTPUT_SIZE, STATE_SIZE,
           INPUT_SIZE, 0, (STATE_SIZE - 1)>::compute(Hf_xu, dU, lam_next, out);
  }
};

template <typename Fxu_Type, typename dU_Type, typename Weight_Type,
          typename Out_Type, std::size_t OUTPUT_SIZE, std::size_t STATE_SIZE,
          std::size_t INPUT_SIZE, std::size_t I_idx, bool Activate>
struct Conditional {};

template <typename Fxu_Type, typename dU_Type, typename Weight_Type,
          typename Out_Type, std::size_t OUTPUT_SIZE, std::size_t STATE_SIZE,
          std::size_t INPUT_SIZE, std::size_t I_idx>
struct Conditional<Fxu_Type, dU_Type, Weight_Type, Out_Type, OUTPUT_SIZE,
                   STATE_SIZE, INPUT_SIZE, I_idx, true> {
  static void compute(const Fxu_Type &Hf_xu, const dU_Type &dU,
                      const Weight_Type &lam_next, Out_Type &out) {

    Row<Fxu_Type, dU_Type, Weight_Type, Out_Type, OUTPUT_SIZE, STATE_SIZE,
        INPUT_SIZE, (OUTPUT_SIZE - 1)>::compute(Hf_xu, dU, lam_next, out);
  }
};

template <typename Fxu_Type, typename dU_Type, typename Weight_Type,
          typename Out_Type, std::size_t OUTPUT_SIZE, std::size_t STATE_SIZE,
          std::size_t INPUT_SIZE, std::size_t I_idx>
struct Conditional<Fxu_Type, dU_Type, Weight_Type, Out_Type, OUTPUT_SIZE,
                   STATE_SIZE, INPUT_SIZE, I_idx, false> {
  static void compute(const Fxu_Type &Hf_xu, const dU_Type &dU,
                      const Weight_Type &lam_next, Out_Type &out) {

    static_cast<void>(Hf_xu);
    static_cast<void>(dU);
    static_cast<void>(lam_next);
    static_cast<void>(out);
    /* Do Nothing */
  }
};

} // namespace FxuLambdaContract

// Public wrapper to run the unrolled recursion.
// Out_Type must be pre-initialized (e.g., zero) before calling, since this
// performs accumulation with "+=" semantics.
template <typename Fxu_Type, typename dU_Type, typename Weight_Type,
          typename Out_Type>
inline void
compute_fx_xu_lambda_contract(const Fxu_Type &Hf_xu, const dU_Type &dU,
                              const Weight_Type &lam_next, Out_Type &out) {
  static_assert(dU_Type::ROWS == 1, "dU must be a (INPUT_SIZE x 1) vector");
  static_assert(Weight_Type::ROWS == 1,
                "lam_next must be a (OUTPUT_SIZE x 1) vector");
  static_assert(Out_Type::ROWS == 1,
                "out must be a (STATE_SIZE x 1) vector (ROWS == 1)");

  constexpr std::size_t STATE_SIZE = Out_Type::COLS;
  constexpr std::size_t OUTPUT_SIZE = Weight_Type::COLS;
  constexpr std::size_t INPUT_SIZE = dU_Type::COLS;

  static_assert(STATE_SIZE > 0 && INPUT_SIZE > 0,
                "STATE_SIZE, OUTPUT_SIZE and INPUT_SIZE must be positive");
  static_assert(Fxu_Type::COLS == STATE_SIZE * STATE_SIZE,
                "Hf_xu ROWS must equal OUTPUT_SIZE * STATE_SIZE");
  static_assert(Fxu_Type::ROWS == INPUT_SIZE,
                "Hf_xu COLS must equal INPUT_SIZE");
  static_assert(Out_Type::COLS == STATE_SIZE, "out COLS must equal STATE_SIZE");

  FxuLambdaContract::Conditional<
      Fxu_Type, dU_Type, Weight_Type, Out_Type, OUTPUT_SIZE, STATE_SIZE,
      INPUT_SIZE, (OUTPUT_SIZE - 1), (INPUT_SIZE > 0)>::compute(Hf_xu, dU,
                                                                lam_next, out);
}

/* fu xx lambda contract */

namespace FuxxLambdaContract {

// J-accumulation: recursively accumulate over j (STATE_SIZE dimension)
template <typename Fuxx_Type, typename dX_Type, typename Value_Type,
          std::size_t STATE_SIZE, std::size_t INPUT_SIZE, std::size_t I,
          std::size_t K, std::size_t J_idx>
struct AccumulateJ {
  static void compute(const Fuxx_Type &Hf_ux, const dX_Type &dX,
                      Value_Type &acc) {
    acc += Hf_ux.template get<I * INPUT_SIZE + K, J_idx>() *
           dX.template get<J_idx, 0>();
    AccumulateJ<Fuxx_Type, dX_Type, Value_Type, STATE_SIZE, INPUT_SIZE, I, K,
                (J_idx - 1)>::compute(Hf_ux, dX, acc);
  }
};

// J-accumulation termination (J_idx == 0)
template <typename Fuxx_Type, typename dX_Type, typename Value_Type,
          std::size_t STATE_SIZE, std::size_t INPUT_SIZE, std::size_t I,
          std::size_t K>
struct AccumulateJ<Fuxx_Type, dX_Type, Value_Type, STATE_SIZE, INPUT_SIZE, I, K,
                   0> {
  static void compute(const Fuxx_Type &Hf_ux, const dX_Type &dX,
                      Value_Type &acc) {
    acc +=
        Hf_ux.template get<I * INPUT_SIZE + K, 0>() * dX.template get<0, 0>();
  }
};

// Column recursion over k (INPUT_SIZE): computes contribution to out(k,0)
template <typename Fuxx_Type, typename dX_Type, typename Weight_Type,
          typename Out_Type, std::size_t STATE_SIZE, std::size_t INPUT_SIZE,
          std::size_t I, std::size_t K_idx>
struct Column {
  static void compute(const Fuxx_Type &Hf_ux, const dX_Type &dX,
                      const Weight_Type &lam_next, Out_Type &out) {
    using Value = typename Out_Type::Value_Type;
    Value acc = static_cast<Value>(0);
    AccumulateJ<Fuxx_Type, dX_Type, Value, STATE_SIZE, INPUT_SIZE, I, K_idx,
                (STATE_SIZE - 1)>::compute(Hf_ux, dX, acc);

    const auto w = lam_next.template get<I, 0>();
    const auto updated = out.template get<K_idx, 0>() + w * acc;
    out.template set<K_idx, 0>(updated);

    Column<Fuxx_Type, dX_Type, Weight_Type, Out_Type, STATE_SIZE, INPUT_SIZE, I,
           (K_idx - 1)>::compute(Hf_ux, dX, lam_next, out);
  }
};

// Column recursion termination for k == 0
template <typename Fuxx_Type, typename dX_Type, typename Weight_Type,
          typename Out_Type, std::size_t STATE_SIZE, std::size_t INPUT_SIZE,
          std::size_t I>
struct Column<Fuxx_Type, dX_Type, Weight_Type, Out_Type, STATE_SIZE, INPUT_SIZE,
              I, 0> {
  static void compute(const Fuxx_Type &Hf_ux, const dX_Type &dX,
                      const Weight_Type &lam_next, Out_Type &out) {
    using Value = typename Out_Type::Value_Type;
    Value acc = static_cast<Value>(0);
    AccumulateJ<Fuxx_Type, dX_Type, Value, STATE_SIZE, INPUT_SIZE, I, 0,
                (STATE_SIZE - 1)>::compute(Hf_ux, dX, acc);

    const auto w = lam_next.template get<I, 0>();
    const auto updated = out.template get<0, 0>() + w * acc;
    out.template set<0, 0>(updated);
  }
};

// Row recursion over i (STATE_SIZE): iterates the outer state index
template <typename Fuxx_Type, typename dX_Type, typename Weight_Type,
          typename Out_Type, std::size_t STATE_SIZE, std::size_t INPUT_SIZE,
          std::size_t I_idx>
struct Row {
  static void compute(const Fuxx_Type &Hf_ux, const dX_Type &dX,
                      const Weight_Type &lam_next, Out_Type &out) {
    Column<Fuxx_Type, dX_Type, Weight_Type, Out_Type, STATE_SIZE, INPUT_SIZE,
           I_idx, (INPUT_SIZE - 1)>::compute(Hf_ux, dX, lam_next, out);
    Row<Fuxx_Type, dX_Type, Weight_Type, Out_Type, STATE_SIZE, INPUT_SIZE,
        (I_idx - 1)>::compute(Hf_ux, dX, lam_next, out);
  }
};

// Row recursion termination for i == 0
template <typename Fuxx_Type, typename dX_Type, typename Weight_Type,
          typename Out_Type, std::size_t STATE_SIZE, std::size_t INPUT_SIZE>
struct Row<Fuxx_Type, dX_Type, Weight_Type, Out_Type, STATE_SIZE, INPUT_SIZE,
           0> {
  static void compute(const Fuxx_Type &Hf_ux, const dX_Type &dX,
                      const Weight_Type &lam_next, Out_Type &out) {
    Column<Fuxx_Type, dX_Type, Weight_Type, Out_Type, STATE_SIZE, INPUT_SIZE, 0,
           (INPUT_SIZE - 1)>::compute(Hf_ux, dX, lam_next, out);
  }
};

} // namespace FuxxLambdaContract

// Public wrapper to run the unrolled recursion.
// Out_Type must be pre-initialized (e.g., zero) before calling, since this
// performs accumulation with "+=" semantics.
template <typename Fuxx_Type, typename dX_Type, typename Weight_Type,
          typename Out_Type>
inline void
compute_fu_xx_lambda_contract(const Fuxx_Type &Hf_ux, const dX_Type &dX,
                              const Weight_Type &lam_next, Out_Type &out) {
  static_assert(dX_Type::ROWS == 1, "dX must be a (STATE_SIZE x 1) vector");
  static_assert(Weight_Type::ROWS == 1,
                "lam_next must be a (STATE_SIZE x 1) vector");
  static_assert(Out_Type::ROWS == 1,
                "out must be a (INPUT_SIZE x 1) vector (ROWS == 1)");

  constexpr std::size_t STATE_SIZE = dX_Type::COLS;
  constexpr std::size_t INPUT_SIZE = Out_Type::COLS;

  static_assert(STATE_SIZE > 0 && INPUT_SIZE > 0,
                "STATE_SIZE and INPUT_SIZE must be positive");
  static_assert(Fuxx_Type::COLS == STATE_SIZE * INPUT_SIZE,
                "Hf_ux ROWS must equal STATE_SIZE * INPUT_SIZE");
  static_assert(Fuxx_Type::ROWS == STATE_SIZE,
                "Hf_ux COLS must equal STATE_SIZE");
  static_assert(Weight_Type::COLS == STATE_SIZE,
                "lam_next COLS must equal STATE_SIZE");
  static_assert(Out_Type::COLS == INPUT_SIZE, "out COLS must equal INPUT_SIZE");

  FuxxLambdaContract::Row<Fuxx_Type, dX_Type, Weight_Type, Out_Type, STATE_SIZE,
                          INPUT_SIZE, (STATE_SIZE - 1)>::compute(Hf_ux, dX,
                                                                 lam_next, out);
}

/* hxx lambda contract */

namespace HxxLambdaContract {

// K-accumulation: recursively accumulate over k (STATE_SIZE dimension)
template <typename Hxx_Type, typename dX_Type, typename Value_Type,
          std::size_t OUTPUT_SIZE, std::size_t STATE_SIZE, std::size_t I,
          std::size_t J, std::size_t K_idx>
struct AccumulateK {
  static void compute(const Hxx_Type &Hh_xx, const dX_Type &dX,
                      Value_Type &acc) {
    acc += Hh_xx.template get<I * STATE_SIZE + J, K_idx>() *
           dX.template get<K_idx, 0>();
    AccumulateK<Hxx_Type, dX_Type, Value_Type, OUTPUT_SIZE, STATE_SIZE, I, J,
                (K_idx - 1)>::compute(Hh_xx, dX, acc);
  }
};

// K-accumulation termination (K_idx == 0)
template <typename Hxx_Type, typename dX_Type, typename Value_Type,
          std::size_t OUTPUT_SIZE, std::size_t STATE_SIZE, std::size_t I,
          std::size_t J>
struct AccumulateK<Hxx_Type, dX_Type, Value_Type, OUTPUT_SIZE, STATE_SIZE, I, J,
                   0> {
  static void compute(const Hxx_Type &Hh_xx, const dX_Type &dX,
                      Value_Type &acc) {
    acc +=
        Hh_xx.template get<I * STATE_SIZE + J, 0>() * dX.template get<0, 0>();
  }
};

// Column recursion over j (STATE_SIZE): computes contribution to out(j,0)
template <typename Hxx_Type, typename dX_Type, typename Weight_Type,
          typename Out_Type, std::size_t OUTPUT_SIZE, std::size_t STATE_SIZE,
          std::size_t I, std::size_t J_idx>
struct Column {
  static void compute(const Hxx_Type &Hh_xx, const dX_Type &dX,
                      const Weight_Type &weight, Out_Type &out) {
    using Value = typename Out_Type::Value_Type;
    Value acc = static_cast<Value>(0);
    AccumulateK<Hxx_Type, dX_Type, Value, OUTPUT_SIZE, STATE_SIZE, I, J_idx,
                (STATE_SIZE - 1)>::compute(Hh_xx, dX, acc);

    const auto w = weight.template get<I, 0>();
    const auto updated = out.template get<J_idx, 0>() + w * acc;
    out.template set<J_idx, 0>(updated);

    Column<Hxx_Type, dX_Type, Weight_Type, Out_Type, OUTPUT_SIZE, STATE_SIZE, I,
           (J_idx - 1)>::compute(Hh_xx, dX, weight, out);
  }
};

// Column recursion termination for j == 0
template <typename Hxx_Type, typename dX_Type, typename Weight_Type,
          typename Out_Type, std::size_t OUTPUT_SIZE, std::size_t STATE_SIZE,
          std::size_t I>
struct Column<Hxx_Type, dX_Type, Weight_Type, Out_Type, OUTPUT_SIZE, STATE_SIZE,
              I, 0> {
  static void compute(const Hxx_Type &Hh_xx, const dX_Type &dX,
                      const Weight_Type &weight, Out_Type &out) {
    using Value = typename Out_Type::Value_Type;
    Value acc = static_cast<Value>(0);
    AccumulateK<Hxx_Type, dX_Type, Value, OUTPUT_SIZE, STATE_SIZE, I, 0,
                (STATE_SIZE - 1)>::compute(Hh_xx, dX, acc);

    const auto w = weight.template get<I, 0>();
    const auto updated = out.template get<0, 0>() + w * acc;
    out.template set<0, 0>(updated);
  }
};

// Row recursion over i (OUTPUT_SIZE): iterates outputs
template <typename Hxx_Type, typename dX_Type, typename Weight_Type,
          typename Out_Type, std::size_t OUTPUT_SIZE, std::size_t STATE_SIZE,
          std::size_t I_idx>
struct Row {
  static void compute(const Hxx_Type &Hh_xx, const dX_Type &dX,
                      const Weight_Type &weight, Out_Type &out) {
    Column<Hxx_Type, dX_Type, Weight_Type, Out_Type, OUTPUT_SIZE, STATE_SIZE,
           I_idx, (STATE_SIZE - 1)>::compute(Hh_xx, dX, weight, out);
    Row<Hxx_Type, dX_Type, Weight_Type, Out_Type, OUTPUT_SIZE, STATE_SIZE,
        (I_idx - 1)>::compute(Hh_xx, dX, weight, out);
  }
};

// Row recursion termination for i == 0
template <typename Hxx_Type, typename dX_Type, typename Weight_Type,
          typename Out_Type, std::size_t OUTPUT_SIZE, std::size_t STATE_SIZE>
struct Row<Hxx_Type, dX_Type, Weight_Type, Out_Type, OUTPUT_SIZE, STATE_SIZE,
           0> {
  static void compute(const Hxx_Type &Hh_xx, const dX_Type &dX,
                      const Weight_Type &weight, Out_Type &out) {
    Column<Hxx_Type, dX_Type, Weight_Type, Out_Type, OUTPUT_SIZE, STATE_SIZE, 0,
           (STATE_SIZE - 1)>::compute(Hh_xx, dX, weight, out);
  }
};

} // namespace HxxLambdaContract

// Public wrapper to run the unrolled recursion.
// Out_Type must be pre-initialized (e.g., zero) before calling, since this
// performs accumulation with "+=" semantics.
template <typename Hxx_Type, typename dX_Type, typename Weight_Type,
          typename Out_Type>
inline void
compute_hxx_lambda_contract(const Hxx_Type &Hh_xx, const dX_Type &dX,
                            const Weight_Type &weight, Out_Type &out) {
  static_assert(dX_Type::ROWS == 1, "dX must be a (STATE_SIZE x 1) vector");
  static_assert(Weight_Type::ROWS == 1,
                "weight must be a (OUTPUT_SIZE x 1) vector");
  static_assert(Out_Type::ROWS == 1,
                "out must be a (STATE_SIZE x 1) vector (ROWS == 1)");

  constexpr std::size_t STATE_SIZE = dX_Type::COLS;
  constexpr std::size_t OUTPUT_SIZE = Weight_Type::COLS;

  static_assert(STATE_SIZE > 0 && OUTPUT_SIZE > 0,
                "STATE_SIZE and OUTPUT_SIZE must be positive");
  static_assert(Hxx_Type::ROWS == STATE_SIZE,
                "Hh_xx ROWS must equal STATE_SIZE");
  static_assert(Hxx_Type::COLS == OUTPUT_SIZE * STATE_SIZE,
                "Hh_xx COLS must equal OUTPUT_SIZE * STATE_SIZE");
  static_assert(Out_Type::COLS == STATE_SIZE, "out COLS must equal STATE_SIZE");

  HxxLambdaContract::Row<Hxx_Type, dX_Type, Weight_Type, Out_Type, OUTPUT_SIZE,
                         STATE_SIZE, (OUTPUT_SIZE - 1)>::compute(Hh_xx, dX,
                                                                 weight, out);
}

} // namespace MatrixOperation

} // namespace PythonOptimization

#endif // __PYTHON_OPTIMIZATION_SQP_MATRIX_OPERATION_HPP__
