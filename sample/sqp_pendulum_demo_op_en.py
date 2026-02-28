"""
File: sqp_pendulum_demo_op_en.py

This script demonstrates nonlinear model predictive control (NMPC) for
 a pendulum-like system with a nonlinear actuator using PANOC/ALM optimization engine.

Main Features:
--------------
- Defines a symbolic plant model for a pendulum with nonlinear actuator dynamics.
- Sets up NMPC problem parameters, including state, input, output dimensions,
 prediction horizon, and cost weights.
- Specifies input bounds and reference trajectory.
- Utilizes OptimizationEngine_CostMatrices to construct cost and constraint matrices for NMPC.
- Uses ALM_PM_Optimizer (with PANOC inner solver) for optimal control input sequence.
- Generates C++ header files for the cost matrices via OptimizationEngine_MatrixUtilityDeploy
 for deployment.
- Prints the names of generated C++ files, optimized cost, and optimal input sequence.

Usage:
------
Run the script to solve the NMPC problem for the pendulum system and generate C++ code for deployment.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(os.path.join(
    os.getcwd(), 'external_libraries', 'MCAP_python_control'))
sys.path.append(os.path.join(
    os.getcwd(), 'external_libraries', 'MCAP_python_optimization'))
sys.path.append(os.path.join(
    os.getcwd(), 'external_libraries', 'python_control_to_cpp'))

import numpy as np
import sympy as sp
from dataclasses import dataclass

from external_libraries.MCAP_python_optimization.optimization_utility.optimization_engine_matrix_utility import OptimizationEngine_CostMatrices
from external_libraries.MCAP_python_optimization.python_optimization.alm_pm_optimizer import (
    ALM_Factory,
    ALM_Problem,
    ALM_Cache,
    ALM_PM_Optimizer,
    BoxProjectionOperator,
)
from external_libraries.MCAP_python_optimization.python_optimization.panoc import PANOC_Cache

from optimization_utility.optimization_engine_matrix_utility_deploy import OptimizationEngine_MatrixUtilityDeploy


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

oe_cost_matrices = OptimizationEngine_CostMatrices(
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
oe_cost_matrices.state_space_parameters = state_space_parameters
oe_cost_matrices.reference_trajectory = reference_trajectory


# initial state
X_initial = np.array([[np.pi / 4.0], [0.0]])
oe_cost_matrices.X_initial = X_initial

U_horizon_initial = np.zeros((nu * N, 1))

# Set up ALM/PANOC solver
# Build U_min / U_max as flat vectors for PANOC box constraints
U_min_flat = oe_cost_matrices.U_min_matrix.reshape((-1, 1))
U_max_flat = oe_cost_matrices.U_max_matrix.reshape((-1, 1))

# Build ALM factory (cost, gradient, output mapping for ALM)
alm_factory = ALM_Factory(
    f=oe_cost_matrices.compute_cost,
    df=oe_cost_matrices.compute_gradient,
    mapping_f1=oe_cost_matrices.compute_output_mapping,
    jacobian_f1_trans=oe_cost_matrices.compute_output_jacobian_trans,
    set_c_project=BoxProjectionOperator(
        lower=oe_cost_matrices.Y_min_matrix.reshape((-1, 1)),
        upper=oe_cost_matrices.Y_max_matrix.reshape((-1, 1)),
    ).project,
    n1=ny * (N + 1),
)

# Build ALM problem
alm_problem = ALM_Problem(
    parametric_cost=alm_factory.psi,
    parametric_gradient=alm_factory.d_psi,
    u_min=U_min_flat,
    u_max=U_max_flat,
    mapping_f1=oe_cost_matrices.compute_output_mapping,
    set_c_project=BoxProjectionOperator(
        lower=oe_cost_matrices.Y_min_matrix.reshape((-1, 1)),
        upper=oe_cost_matrices.Y_max_matrix.reshape((-1, 1)),
    ).project,
    n1=ny * (N + 1),
)

# Build caches
panoc_cache = PANOC_Cache(
    problem_size=nu * N,
    tolerance=1e-4,
    lbfgs_memory=5,
)

alm_cache = ALM_Cache(
    panoc_cache=panoc_cache,
    n1=ny * (N + 1),
)

# Build ALM/PM optimizer
solver = ALM_PM_Optimizer(
    alm_cache=alm_cache,
    alm_problem=alm_problem,
)
solver.set_solver_max_iteration(
    outer_max_iterations=30,
    inner_max_iterations=500,
)

# You can create cpp header which can easily define
# OptimizationEngine_CostMatrices as C++ code
deployed_file_names = OptimizationEngine_MatrixUtilityDeploy.generate_cpp_code(
    cost_matrices=oe_cost_matrices)
print(deployed_file_names)

solver.solve(u=U_horizon_initial)

U_opt = U_horizon_initial.reshape((nu, N))

print("Optimized cost:", solver.solver_status.cost)
print("Optimal input sequence:\n", U_opt)
