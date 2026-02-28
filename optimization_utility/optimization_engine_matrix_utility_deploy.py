"""
File: optimization_engine_matrix_utility_deploy.py

This module provides utilities for deploying PANOC/ALM optimization engine
matrix-related code from Python to C++ for NMPC
(Nonlinear Model Predictive Control) applications.
It automates the generation of C++ header files for cost matrices,
state and measurement functions, Jacobians, and constraint limits,
based on Python data structures and code.

Unlike sqp_matrix_utility_deploy.py, this module does not generate
Hessian code, as the PANOC/ALM algorithm only requires first-order
gradient information.

Usage:
------
This module is intended to be used as part of a Python-to-C++ code
generation pipeline for NMPC applications,
where Python models and constraints are automatically
translated into efficient C++ code for deployment.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))


import inspect
import numpy as np
from dataclasses import fields, is_dataclass

from optimization_utility.common_optimization_deploy import MinMaxCodeGenerator
from optimization_utility.common_optimization_deploy import get_active_array

from optimization_utility.sqp_matrix_utility_deploy import create_sparse_matrix_code
from optimization_utility.sqp_matrix_utility_deploy import create_and_write_parameter_class_code
from optimization_utility.sqp_matrix_utility_deploy import create_and_write_state_function_code
from optimization_utility.sqp_matrix_utility_deploy import create_and_write_measurement_function_code
from optimization_utility.sqp_matrix_utility_deploy import create_and_write_state_measurement_jacobian_code

from external_libraries.MCAP_python_optimization.optimization_utility.optimization_engine_matrix_utility import OptimizationEngine_CostMatrices
from external_libraries.python_numpy_to_cpp.python_numpy.numpy_deploy import NumpyDeploy
from external_libraries.python_numpy_to_cpp.python_numpy.numpy_deploy import python_to_cpp_types
from external_libraries.MCAP_python_control.python_control.control_deploy import ControlDeploy
from external_libraries.MCAP_python_control.python_control.control_deploy import FunctionExtractor
from external_libraries.python_control_to_cpp.python_control.kalman_filter_deploy import FunctionToCppVisitor


class OptimizationEngine_MatrixUtilityDeploy:
    """
    Utility class for deploying PANOC/ALM optimization engine
      cost matrices and related functions from Python to C++ code.

    This class provides a static method to generate C++ header files
      that encapsulate the cost matrices, system functions, Jacobians,
        and constraints required for NMPC
          (Nonlinear Model Predictive Control) optimization.
            The generated code is tailored for integration with a C++ optimization framework.

    Unlike SQP_MatrixUtilityDeploy, this class does not generate
    Hessian-related code, as the PANOC/ALM algorithm only requires
    first-order gradient information.

    Methods
    -------
    generate_cpp_code(cost_matrices: OptimizationEngine_CostMatrices, file_name: str = None)
      -> list[str]:
        Generates C++ header files for the provided cost matrices and associated functions.

        Parameters
        ----------
        cost_matrices : OptimizationEngine_CostMatrices
            An object containing all required matrices, functions,
              and parameters for NMPC optimization.
        file_name : str, optional
            Custom base name for the generated C++ files. If not provided,
              the caller's file name is used.

        Returns
        -------
        deployed_file_names : list[str]
            List of generated C++ header file names.

        Functionality
        -------------
        - Inspects the caller's context to determine variable
          and file names for code generation.
        - Generates C++ code for:
            - Parameter class
            - State and measurement functions
            - Jacobians (state and measurement)
            - Input and output constraints (min/max)
        - Assembles all generated components into a single C++ header file
          with appropriate includes and namespace.
        - Writes the generated code to disk and returns the list of file names.

    Notes
    -----
    - Relies on several helper functions and classes
      (e.g., ControlDeploy, NumpyDeploy, MinMaxCodeGenerator) for code generation.
    - Assumes that the cost_matrices object provides all necessary attributes and code file names.
    - Designed for automated deployment of NMPC problem definitions from Python to C++.
    """

    def __init__(self):
        pass

    @staticmethod
    def generate_cpp_code(
        cost_matrices: OptimizationEngine_CostMatrices,
        file_name: str = None
    ):
        """
        Generates C++ code for deploying PANOC/ALM optimization engine
          cost matrices and related functions for NMPC.

        This static method takes an OptimizationEngine_CostMatrices object
          and generates C++ header files
        for all required matrix types, function objects,
          and parameter classes used in nonlinear
        model predictive control (NMPC).
          The generated code includes definitions for cost matrices,
        state and measurement functions, Jacobians,
        and input/output constraints.

        Args:
            cost_matrices (OptimizationEngine_CostMatrices): The cost matrices
              and associated function/code file names
                required for NMPC deployment.
            file_name (str, optional): The base name for the generated
              C++ header file. If None, the caller's
                file name is used.

        Returns:
            List[str]: A list of deployed C++ file names generated during the process.

        Raises:
            ValueError: If the data type of the cost matrices is not supported.

        Notes:
            - The method inspects the caller's frame to determine the variable name
              and file name for code generation.
            - It generates code for parameter classes, state/measurement functions,
              Jacobians, and limits.
            - The generated C++ code uses a namespace based on the caller's file
              and variable name.
            - The method writes the generated code to files and returns
              the list of file names.
        """
        deployed_file_names = []

        data_type = cost_matrices.Qx[0, 0].dtype.name
        ControlDeploy.restrict_data_type(data_type)

        type_name = NumpyDeploy.check_dtype(cost_matrices.Qx)

        # %% inspect arguments
        # Get the caller's frame
        frame = inspect.currentframe().f_back
        # Get the caller's local variables
        caller_locals = frame.f_locals
        # Find the variable name that matches the matrix_in value
        variable_name = None
        for name, value in caller_locals.items():
            if value is cost_matrices:
                variable_name = name
                break
        # Get the caller's file name
        if file_name is None:
            caller_file_full_path = frame.f_code.co_filename
            caller_file_name = os.path.basename(caller_file_full_path)
            caller_file_name_no_extension = os.path.splitext(caller_file_name)[
                0]
        else:
            caller_file_name_no_extension = file_name

        code_file_name = caller_file_name_no_extension + "_" + variable_name
        code_file_name_ext = code_file_name + ".hpp"

        # %% generate functions code
        # parameter class code
        parameter_class_file_name_no_extension = \
            f"{caller_file_name_no_extension}_parameter"

        parameter_class_cpp_file_name = \
            create_and_write_parameter_class_code(
                cost_matrices.state_space_parameters,
                type_name,
                parameter_class_file_name_no_extension)
        deployed_file_names.append(parameter_class_cpp_file_name)

        # state equation function code
        state_function_file_name_no_extension = \
            cost_matrices.state_function_code_file_name.split(".")[0]

        state_function_cpp_file_name = \
            create_and_write_state_function_code(
                state_function_file_name_no_extension)
        deployed_file_names.append(state_function_cpp_file_name)

        # measurement equation function code
        measurement_function_file_name_no_extension = \
            cost_matrices.measurement_function_code_file_name.split(".")[0]

        measurement_function_cpp_file_name = \
            create_and_write_measurement_function_code(
                measurement_function_file_name_no_extension)
        deployed_file_names.append(measurement_function_cpp_file_name)

        # state jacobian x function code
        state_jacobian_x_file_name_no_extension = \
            cost_matrices.state_jacobian_x_code_file_name.split(".")[0]

        state_jacobian_x_cpp_file_name, state_jacobian_x_SparseAvailable_list = \
            create_and_write_state_measurement_jacobian_code(
                state_jacobian_x_file_name_no_extension,
                "State_Jacobian_x_Type",
                type_name)
        deployed_file_names.append(state_jacobian_x_cpp_file_name)

        # state jacobian u function code
        state_jacobian_u_file_name_no_extension = \
            cost_matrices.state_jacobian_u_code_file_name.split(".")[0]

        state_jacobian_u_cpp_file_name, state_jacobian_u_SparseAvailable_list = \
            create_and_write_state_measurement_jacobian_code(
                state_jacobian_u_file_name_no_extension,
                "State_Jacobian_u_Type",
                type_name)
        deployed_file_names.append(state_jacobian_u_cpp_file_name)

        # measurement jacobian x function code
        measurement_jacobian_x_file_name_no_extension = \
            cost_matrices.measurement_jacobian_x_code_file_name.split(".")[0]

        measurement_jacobian_x_cpp_file_name, measurement_jacobian_x_SparseAvailable_list = \
            create_and_write_state_measurement_jacobian_code(
                measurement_jacobian_x_file_name_no_extension,
                "Measurement_Jacobian_x_Type",
                type_name)
        deployed_file_names.append(measurement_jacobian_x_cpp_file_name)

        # %% create limits code
        U_size = cost_matrices.nu
        Y_size = cost_matrices.ny

        # U_min
        min_max_array = cost_matrices.U_min_matrix[:, 0].reshape(-1, 1)
        U_min_active_array = get_active_array(min_max_array)

        U_min_code_generator = MinMaxCodeGenerator(
            min_max_array=min_max_array,
            min_max_name="U_min",
            size=U_size
        )
        U_min_code_generator.generate_active_set(
            is_active_array=U_min_active_array
        )

        # U_max
        min_max_array = cost_matrices.U_max_matrix[:, 0].reshape(-1, 1)
        U_max_active_array = get_active_array(min_max_array)

        U_max_code_generator = MinMaxCodeGenerator(
            min_max_array=min_max_array,
            min_max_name="U_max",
            size=U_size
        )
        U_max_code_generator.generate_active_set(
            is_active_array=U_max_active_array
        )

        # Y_min
        min_max_array = cost_matrices.Y_min_matrix[:, 0].reshape(-1, 1)
        Y_min_active_array = get_active_array(min_max_array)

        Y_min_code_generator = MinMaxCodeGenerator(
            min_max_array=min_max_array,
            min_max_name="Y_min",
            size=Y_size
        )
        Y_min_code_generator.generate_active_set(
            is_active_array=Y_min_active_array
        )

        # Y_max
        min_max_array = cost_matrices.Y_max_matrix[:, 0].reshape(-1, 1)
        Y_max_active_array = get_active_array(min_max_array)

        Y_max_code_generator = MinMaxCodeGenerator(
            min_max_array=min_max_array,
            min_max_name="Y_max",
            size=Y_size
        )
        Y_max_code_generator.generate_active_set(
            is_active_array=Y_max_active_array
        )

        # Limits code
        U_min_file_name, U_min_file_name_no_extension = \
            U_min_code_generator.create_limits_code(
                data_type=data_type,
                variable_name=variable_name,
                caller_file_name_no_extension=caller_file_name_no_extension
            )
        deployed_file_names.append(U_min_file_name)

        U_max_file_name, U_max_file_name_no_extension = \
            U_max_code_generator.create_limits_code(
                data_type=data_type,
                variable_name=variable_name,
                caller_file_name_no_extension=caller_file_name_no_extension
            )
        deployed_file_names.append(U_max_file_name)

        Y_min_file_name, Y_min_file_name_no_extension = \
            Y_min_code_generator.create_limits_code(
                data_type=data_type,
                variable_name=variable_name,
                caller_file_name_no_extension=caller_file_name_no_extension
            )
        deployed_file_names.append(Y_min_file_name)

        Y_max_file_name, Y_max_file_name_no_extension = \
            Y_max_code_generator.create_limits_code(
                data_type=data_type,
                variable_name=variable_name,
                caller_file_name_no_extension=caller_file_name_no_extension
            )
        deployed_file_names.append(Y_max_file_name)

        # %% create cpp code
        code_text = ""

        file_header_macro_name = "__" + code_file_name.upper() + "_HPP__"

        code_text += "#ifndef " + file_header_macro_name + "\n"
        code_text += "#define " + file_header_macro_name + "\n\n"

        code_text += f"#include \"{parameter_class_cpp_file_name}\"\n"
        code_text += f"#include \"{state_function_cpp_file_name}\"\n"
        code_text += f"#include \"{measurement_function_cpp_file_name}\"\n"
        code_text += f"#include \"{state_jacobian_x_cpp_file_name}\"\n"
        code_text += f"#include \"{state_jacobian_u_cpp_file_name}\"\n"
        code_text += f"#include \"{measurement_jacobian_x_cpp_file_name}\"\n\n"

        code_text += f"#include \"{U_min_file_name}\"\n"
        code_text += f"#include \"{U_max_file_name}\"\n"
        code_text += f"#include \"{Y_min_file_name}\"\n"
        code_text += f"#include \"{Y_max_file_name}\"\n\n"

        code_text += "#include \"python_optimization.hpp\"\n\n"

        namespace_name = code_file_name

        code_text += "namespace " + namespace_name + " {\n\n"

        code_text += "using namespace PythonNumpy;\n"
        code_text += "using namespace PythonControl;\n"
        code_text += "using namespace PythonOptimization;\n\n"

        code_text += f"constexpr std::size_t NP = {cost_matrices.Np};\n\n"

        code_text += f"constexpr std::size_t INPUT_SIZE = {cost_matrices.nu};\n"
        code_text += f"constexpr std::size_t STATE_SIZE = {cost_matrices.nx};\n"
        code_text += f"constexpr std::size_t OUTPUT_SIZE = {cost_matrices.ny};\n\n"

        code_text += f"using X_Type = StateSpaceState_Type<{type_name}, STATE_SIZE>;\n"
        code_text += f"using U_Type = StateSpaceInput_Type<{type_name}, INPUT_SIZE>;\n"
        code_text += f"using Y_Type = StateSpaceOutput_Type<{type_name}, OUTPUT_SIZE>;\n\n"

        code_text += f"using Parameter_Type = {parameter_class_file_name_no_extension}::Parameter;\n\n"

        code_text += f"using State_Jacobian_X_Matrix_Type = {state_jacobian_x_file_name_no_extension}" + \
            "::State_Jacobian_x_Type;\n"
        code_text += f"using State_Jacobian_U_Matrix_Type = {state_jacobian_u_file_name_no_extension}" + \
            "::State_Jacobian_u_Type;\n"
        code_text += f"using Measurement_Jacobian_X_Matrix_Type = {measurement_jacobian_x_file_name_no_extension}" + \
            "::Measurement_Jacobian_x_Type;\n\n"

        code_text += f"using Qx_Type = DiagMatrix_Type<{type_name}, STATE_SIZE>;\n"
        code_text += f"using R_Type = DiagMatrix_Type<{type_name}, INPUT_SIZE>;\n"
        code_text += f"using Qy_Type = DiagMatrix_Type<{type_name}, OUTPUT_SIZE>;\n\n"

        code_text += f"using U_Min_Type = {U_min_file_name_no_extension}::type;\n"
        code_text += f"using U_Max_Type = {U_max_file_name_no_extension}::type;\n"
        code_text += f"using Y_Min_Type = {Y_min_file_name_no_extension}::type;\n"
        code_text += f"using Y_Max_Type = {Y_max_file_name_no_extension}::type;\n\n"

        code_text += f"using Reference_Trajectory_Type = DenseMatrix_Type<{type_name}, OUTPUT_SIZE, (NP + 1)>;\n\n"

        code_text += f"using type = OptimizationEngine_CostMatrices_Type<{type_name}, NP, Parameter_Type,\n" + \
            "    U_Min_Type, U_Max_Type, Y_Min_Type, Y_Max_Type,\n" + \
            "    State_Jacobian_X_Matrix_Type,\n" + \
            "    State_Jacobian_U_Matrix_Type,\n" + \
            "    Measurement_Jacobian_X_Matrix_Type>;\n\n"

        code_text += "inline auto make() -> type {\n\n"

        # limits
        code_text = U_min_code_generator.write_limits_code(
            code_text, type_name)
        code_text = U_max_code_generator.write_limits_code(
            code_text, type_name)

        code_text = Y_min_code_generator.write_limits_code(
            code_text, type_name)
        code_text = Y_max_code_generator.write_limits_code(
            code_text, type_name)

        code_text += "  Qx_Type Qx = make_DiagMatrix<STATE_SIZE>(\n"
        for i in range(cost_matrices.nx):
            code_text += f"      static_cast<{type_name}>({cost_matrices.Qx[i, i]})"
            if i != cost_matrices.nx - 1:
                code_text += ",\n"
            else:
                code_text += ");\n\n"

        code_text += "  R_Type R = make_DiagMatrix<INPUT_SIZE>(\n"
        for i in range(cost_matrices.nu):
            code_text += f"      static_cast<{type_name}>({cost_matrices.R[i, i]})"
            if i != cost_matrices.nu - 1:
                code_text += ",\n"
            else:
                code_text += ");\n\n"

        code_text += "  Qy_Type Qy = make_DiagMatrix<OUTPUT_SIZE>(\n"
        for i in range(cost_matrices.ny):
            code_text += f"      static_cast<{type_name}>({cost_matrices.Qy[i, i]})"
            if i != cost_matrices.ny - 1:
                code_text += ",\n"
            else:
                code_text += ");\n\n"

        code_text += "  Reference_Trajectory_Type reference_trajectory;\n\n"

        code_text += "    type cost_matrices =\n" + \
            f"        make_OptimizationEngine_CostMatrices<{type_name}, NP, Parameter_Type,\n" + \
            "            U_Min_Type, U_Max_Type, Y_Min_Type, Y_Max_Type,\n" + \
            "            State_Jacobian_X_Matrix_Type,\n" + \
            "            State_Jacobian_U_Matrix_Type,\n" + \
            "            Measurement_Jacobian_X_Matrix_Type>(\n" + \
            "                Qx, R, Qy, U_min, U_max, Y_min, Y_max);\n\n"

        code_text += "    PythonOptimization::StateFunction_Object<X_Type, U_Type, Parameter_Type> state_function =\n" + \
            f"        {state_function_file_name_no_extension}::Function<" + \
            "X_Type, U_Type, Parameter_Type>::function;\n\n"

        code_text += "    PythonOptimization::MeasurementFunction_Object<Y_Type, X_Type, U_Type, Parameter_Type> measurement_function =\n" + \
            f"        {measurement_function_file_name_no_extension}::Function<" + \
            "X_Type, U_Type, Parameter_Type, Y_Type>::function;\n\n"

        code_text += "    PythonOptimization::StateFunctionJacobian_X_Object<\n" + \
            "        State_Jacobian_X_Matrix_Type, X_Type, U_Type, Parameter_Type> state_jacobian_x_function =\n" + \
            f"        {state_jacobian_x_file_name_no_extension}::Function<" + \
            "X_Type, U_Type, Parameter_Type>::function;\n\n"

        code_text += "    PythonOptimization::StateFunctionJacobian_U_Object<\n" + \
            "        State_Jacobian_U_Matrix_Type, X_Type, U_Type, Parameter_Type> state_jacobian_u_function =\n" + \
            f"        {state_jacobian_u_file_name_no_extension}::Function<" + \
            "X_Type, U_Type, Parameter_Type>::function;\n\n"

        code_text += "    PythonOptimization::MeasurementFunctionJacobian_X_Object<\n" + \
            "        Measurement_Jacobian_X_Matrix_Type, X_Type, U_Type, Parameter_Type> measurement_jacobian_x_function =\n" + \
            f"        {measurement_jacobian_x_file_name_no_extension}::Function<" + \
            "X_Type, U_Type, Parameter_Type>::function;\n\n"

        code_text += "    cost_matrices.set_function_objects(\n"
        code_text += "        state_function,\n"
        code_text += "        measurement_function,\n"
        code_text += "        state_jacobian_x_function,\n"
        code_text += "        state_jacobian_u_function,\n"
        code_text += "        measurement_jacobian_x_function\n"
        code_text += "    );\n\n"

        code_text += "    return cost_matrices;\n\n"

        code_text += "}\n\n"

        code_text += "} // namespace " + namespace_name + "\n\n"

        code_text += "#endif // " + file_header_macro_name + "\n"

        code_file_name_ext = ControlDeploy.write_to_file(
            code_text, code_file_name_ext)

        deployed_file_names.append(code_file_name_ext)

        return deployed_file_names
