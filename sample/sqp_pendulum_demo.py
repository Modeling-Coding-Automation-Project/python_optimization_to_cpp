"""
File: sqp_pendulum_demo.py

This script demonstrates nonlinear model predictive control (NMPC) for
 a pendulum-like system with a nonlinear actuator using Sequential Quadratic Programming (SQP).

Main Features:
--------------
- Defines a symbolic plant model for a pendulum with nonlinear actuator dynamics.
- Sets up NMPC problem parameters, including state, input, output dimensions,
 prediction horizon, and cost weights.
- Specifies input bounds and reference trajectory.
- Utilizes SQP_CostMatrices_NMPC to construct cost and constraint matrices for NMPC.
- Uses SQP_ActiveSet_PCG_PLS as the SQP solver for optimal control input sequence.
- Generates C++ header files for the cost matrices via SQP_MatrixUtilityDeploy for deployment.
- Prints the names of generated C++ files, optimized cost, and optimal input sequence.

Usage:
------
Run the script to solve the NMPC problem for the pendulum system and generate C++ code for deployment.
"""
import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(
    os.getcwd(), 'external_libraries', 'MCAP_python_control'))
sys.path.append(os.path.join(
    os.getcwd(), 'external_libraries', 'MCAP_python_optimization'))
sys.path.append(os.path.join(
    os.getcwd(), 'external_libraries', 'python_control_to_cpp'))

import numpy as np
import sympy as sp
from dataclasses import dataclass

from external_libraries.MCAP_python_optimization.optimization_utility.sqp_matrix_utility import SQP_CostMatrices_NMPC
from external_libraries.MCAP_python_optimization.python_optimization.sqp_active_set_pcg_pls import SQP_ActiveSet_PCG_PLS

from optimization_utility.sqp_matrix_utility_deploy import SQP_MatrixUtilityDeploy


def create_plant_model():
    theta, omega, u0, dt, a, b, c, d = sp.symbols(
        'theta omega u0 dt a b c d', real=True)

    theta_next = theta + dt * omega
    omega_dot = -a * sp.sin(theta) - b * omega + c * \
        sp.cos(theta) * u0 + d * (u0 ** 2)
    omega_next = omega + dt * omega_dot

    f = sp.Matrix([theta_next, omega_next])
    h = sp.Matrix([[theta]])

    x_syms = sp.Matrix([[theta], [omega]])
    u_syms = sp.Matrix([[u0]])

    return f, h, x_syms, u_syms


# --- Nonlinear NMPC Problem (Pendulum-like with nonlinear actuator) ---
nx = 2   # [theta, omega]
nu = 1   # scalar input
ny = 1   # scalar output (theta)
N = 20   # Prediction Horizon

dt = 0.05

# dynamics params
a = 9.81     # gravity/l over I scaling
b = 0.3      # damping
c = 1.2      # state-dependent control effectiveness: cos(theta)*u
d = 0.10     # actuator nonlinearity: u^2


@dataclass
class Parameters:
    a: float = a
    b: float = b
    c: float = c
    d: float = d
    dt: float = dt


state_space_parameters = Parameters()

# cost weights
Qx = np.diag([2.5, 0.5])
Qy = np.diag([2.5])
R = np.diag([0.05])
Px = Qx.copy()
Py = Qy.copy()

# input bounds
u_min = np.array([[-2.0]])
u_max = np.array([[2.0]])

# reference
reference = np.array([[0.0]])
reference_trajectory = np.tile(reference, (1, N + 1))

# Create symbolic plant model
f, h, x_syms, u_syms = create_plant_model()

sqp_cost_matrices = SQP_CostMatrices_NMPC(
    x_syms=x_syms,
    u_syms=u_syms,
    state_equation_vector=f,
    measurement_equation_vector=h,
    Np=N,
    Qx=Qx,
    Qy=Qy,
    R=R,
    U_min=u_min,
    U_max=u_max,
)


# --- Example Execution ---
sqp_cost_matrices.state_space_parameters = state_space_parameters
sqp_cost_matrices.reference_trajectory = reference_trajectory


# initial state
X_initial = np.array([[np.pi / 4.0], [0.0]])
U_horizon_initial = np.zeros((nu, N))

solver = SQP_ActiveSet_PCG_PLS(
    U_size=(nu, N)
)
solver.set_solver_max_iteration(30)

# You can create cpp header which can easily define SQP_CostMatrices_NMPC as C++ code
deployed_file_names = SQP_MatrixUtilityDeploy.generate_cpp_code(
    cost_matrices=sqp_cost_matrices)
print(deployed_file_names)

U_opt = solver.solve(
    U_horizon_initial=U_horizon_initial,
    cost_and_gradient_function=sqp_cost_matrices.compute_cost_and_gradient,
    cost_function=sqp_cost_matrices.compute_cost,
    hvp_function=sqp_cost_matrices.hvp_analytic,
    X_initial=X_initial,
    U_min_matrix=sqp_cost_matrices.U_min_matrix,
    U_max_matrix=sqp_cost_matrices.U_max_matrix,
)

print("Optimized cost:", solver.J_opt)
print("Optimal input sequence:\n", U_opt)
