import os
import sys
sys.path.append(os.getcwd())

import re
import inspect
import ast
import astor
import numpy as np
import sympy as sp
from dataclasses import dataclass, fields, is_dataclass

from external_libraries.MCAP_python_optimization.optimization_utility.sqp_matrix_utility import SQP_CostMatrices_NMPC
from external_libraries.python_numpy_to_cpp.python_numpy.numpy_deploy import NumpyDeploy
from external_libraries.python_numpy_to_cpp.python_numpy.numpy_deploy import python_to_cpp_types
from external_libraries.MCAP_python_control.python_control.control_deploy import ControlDeploy
from external_libraries.MCAP_python_control.python_control.control_deploy import ExpressionDeploy
from external_libraries.MCAP_python_control.python_control.control_deploy import FunctionExtractor
from external_libraries.MCAP_python_control.python_control.control_deploy import IntegerPowerReplacer
from external_libraries.python_control_to_cpp.python_control.kalman_filter_deploy import FunctionToCppVisitor


def create_and_write_parameter_class_code(
        parameter_object,
        value_type_name: str,
        file_name_without_extension: str
):

    if not is_dataclass(parameter_object):
        raise TypeError("parameter_object must be a dataclass instance")

    file_path = ControlDeploy.find_file(
        f"{file_name_without_extension}.py", os.getcwd())

    try:
        value_type_name = python_to_cpp_types[value_type_name]
    except KeyError:
        pass

    code_text = ""
    header_macro_text = "__" + file_name_without_extension.upper() + "_HPP__"

    code_text += f"#ifndef {header_macro_text}\n"
    code_text += f"#define {header_macro_text}\n\n"

    code_text += f"namespace {file_name_without_extension} {{\n\n"

    code_text += "class Parameter {\n"
    code_text += "public:\n"

    name_value_pairs = [(f.name, getattr(parameter_object, f.name))
                        for f in fields(parameter_object)]
    for _, name_value in enumerate(name_value_pairs):
        code_text += f"  {value_type_name} {name_value[0]} = static_cast<{value_type_name}>({name_value[1]});\n"

    code_text += "};\n\n"

    code_text += f"}} // namespace {file_name_without_extension}\n\n"

    code_text += f"#endif // {header_macro_text}\n"

    saved_file_name = ControlDeploy.write_to_file(
        code_text, f"{file_name_without_extension}.hpp")

    return saved_file_name


def create_and_write_state_function_code(function_name: str):

    file_path = ControlDeploy.find_file(
        f"{function_name}.py", os.getcwd())

    extractor = FunctionExtractor(file_path)
    functions = extractor.extract()
    state_function_code = []

    for _, code in functions.items():
        converter = FunctionToCppVisitor("X_Type")

        state_function_code.append(converter.convert(code))

    # generate code text
    code_text = ""
    header_macro_text = "__" + function_name.upper() + "_HPP__"

    code_text += f"#ifndef {header_macro_text}\n"
    code_text += f"#define {header_macro_text}\n\n"

    code_text += "#include \"python_math.hpp\"\n\n"

    code_text += f"namespace {function_name} {{\n\n"

    code_text += "using namespace PythonMath;\n\n"

    code_text += "template <typename X_Type, typename U_Type, typename Parameter_Type>\n"
    code_text += "class Function {\n"
    code_text += "public:\n"

    # sympy_function
    code_text += f"static "
    code_text += state_function_code[0]
    code_text += "\n"

    # function
    code_text += f"static "
    code_text += state_function_code[1]
    code_text += "\n"

    code_text += "};\n\n"

    code_text += f"}} // namespace {function_name}\n\n"

    code_text += f"#endif // {header_macro_text}\n"

    saved_file_name = ControlDeploy.write_to_file(
        code_text, f"{function_name}.hpp")

    return saved_file_name


def create_and_write_measurement_function_code(function_name: str):

    file_path = ControlDeploy.find_file(
        f"{function_name}.py", os.getcwd())

    extractor = FunctionExtractor(file_path)
    functions = extractor.extract()
    state_function_code = []

    for _, code in functions.items():
        converter = FunctionToCppVisitor("Y_Type")

        state_function_code.append(converter.convert(code))

    # generate code text
    code_text = ""
    header_macro_text = "__" + function_name.upper() + "_HPP__"

    code_text += f"#ifndef {header_macro_text}\n"
    code_text += f"#define {header_macro_text}\n\n"

    code_text += "#include \"python_math.hpp\"\n\n"

    code_text += f"namespace {function_name} {{\n\n"

    code_text += "using namespace PythonMath;\n\n"

    code_text += "template <typename X_Type, typename U_Type, " + \
        "typename Parameter_Type, typename Y_Type>\n"
    code_text += "class Function {\n"
    code_text += "public:\n"

    # sympy_function
    code_text += f"static "
    code_text += state_function_code[0]
    code_text += "\n"

    # function
    code_text += f"static "
    code_text += state_function_code[1]
    code_text += "\n"

    code_text += "};\n\n"

    code_text += f"}} // namespace {function_name}\n\n"

    code_text += f"#endif // {header_macro_text}\n"

    saved_file_name = ControlDeploy.write_to_file(
        code_text, f"{function_name}.hpp")

    return saved_file_name


def create_and_write_state_measurement_jacobian_code(
        function_name: str,
        output_type: str
):

    file_path = ControlDeploy.find_file(
        f"{function_name}.py", os.getcwd())

    extractor = FunctionExtractor(file_path)
    functions = extractor.extract()
    state_function_code = []
    SparseAvailable_list = []

    for _, code in functions.items():
        converter = FunctionToCppVisitor(output_type)

        state_function_code.append(converter.convert(code))
        SparseAvailable_list.append(converter.SparseAvailable)

    SparseAvailable_list = [
        x for x in SparseAvailable_list if x is not None]

    # generate code text
    code_text = ""
    header_macro_text = "__" + function_name.upper() + "_HPP__"

    code_text += f"#ifndef {header_macro_text}\n"
    code_text += f"#define {header_macro_text}\n\n"

    code_text += "#include \"python_math.hpp\"\n\n"

    code_text += f"namespace {function_name} {{\n\n"

    code_text += "using namespace PythonMath;\n\n"

    code_text += "template <typename X_Type, typename U_Type, " + \
        " typename Parameter_Type, typename " + output_type + ">\n"
    code_text += "class Function {\n"
    code_text += "public:\n"

    # sympy_function
    code_text += f"static "
    code_text += state_function_code[0]
    code_text += "\n"

    # function
    code_text += f"static "
    code_text += state_function_code[1]
    code_text += "\n"

    code_text += "};\n\n"

    code_text += f"}} // namespace {function_name}\n\n"

    code_text += f"#endif // {header_macro_text}\n"

    saved_file_name = ControlDeploy.write_to_file(
        code_text, f"{function_name}.hpp")

    return saved_file_name, SparseAvailable_list


def create_and_write_state_measurement_hessian_code(
        function_name: str,
        output_type: str
):

    file_path = ControlDeploy.find_file(
        f"{function_name}.py", os.getcwd())

    extractor = FunctionExtractor(file_path)
    functions = extractor.extract()
    state_function_code = []
    SparseAvailable_list = []

    for _, code in functions.items():
        converter = FunctionToCppVisitor(output_type)

        state_function_code.append(converter.convert(code))
        SparseAvailable_list.append(converter.SparseAvailable)

    SparseAvailable_list = [
        x for x in SparseAvailable_list if x is not None]

    # generate code text
    code_text = ""
    header_macro_text = "__" + function_name.upper() + "_HPP__"

    code_text += f"#ifndef {header_macro_text}\n"
    code_text += f"#define {header_macro_text}\n\n"

    code_text += "#include \"python_math.hpp\"\n\n"

    code_text += f"namespace {function_name} {{\n\n"

    code_text += "using namespace PythonMath;\n\n"

    code_text += "template <typename X_Type, typename U_Type, " + \
        " typename Parameter_Type, typename " + output_type + ">\n"
    code_text += "class Function {\n"
    code_text += "public:\n"

    # sympy_function
    code_text += f"static "
    code_text += state_function_code[0]
    code_text += "\n"

    # function
    code_text += f"static "
    code_text += state_function_code[1]
    code_text += "\n"

    code_text += "};\n\n"

    code_text += f"}} // namespace {function_name}\n\n"

    code_text += f"#endif // {header_macro_text}\n"

    saved_file_name = ControlDeploy.write_to_file(
        code_text, f"{function_name}.hpp")

    return saved_file_name, SparseAvailable_list


class SQP_MatrixUtilityDeploy:

    def __init__(self):
        pass

    @staticmethod
    def generate_cpp_code(
        cost_matrices: SQP_CostMatrices_NMPC,
        file_name: str = None
    ):

        deployed_file_names = []

        ControlDeploy.restrict_data_type(cost_matrices.Qx[0, 0].dtype.name)

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
            caller_file_name_without_ext = os.path.splitext(caller_file_name)[
                0]
        else:
            caller_file_name_without_ext = file_name

        # %% generate functions code
        # parameter class code
        parameter_class_file_name_without_ext = \
            f"{caller_file_name_without_ext}_parameter"

        parameter_class_cpp_file_name = \
            create_and_write_parameter_class_code(
                cost_matrices.state_space_parameters,
                type_name,
                parameter_class_file_name_without_ext)

        # state equation function code
        state_function_file_name_without_ext = \
            cost_matrices.state_function_code_file_name.split(".")[0]

        state_function_cpp_file_name = \
            create_and_write_state_function_code(
                state_function_file_name_without_ext)

        # measurement equation function code
        measurement_function_file_name_without_ext = \
            cost_matrices.measurement_function_code_file_name.split(".")[0]

        measurement_function_cpp_file_name = \
            create_and_write_measurement_function_code(
                measurement_function_file_name_without_ext)

        # state jacobian x function code
        state_jacobian_x_file_name_without_ext = \
            cost_matrices.state_jacobian_x_code_file_name.split(".")[0]

        state_jacobian_x_cpp_file_name, state_jacobian_x_SparseAvailable_list = \
            create_and_write_state_measurement_jacobian_code(
                state_jacobian_x_file_name_without_ext,
                "State_Jacobian_x_Type")

        # state jacobian u function code
        state_jacobian_u_file_name_without_ext = \
            cost_matrices.state_jacobian_u_code_file_name.split(".")[0]

        state_jacobian_u_cpp_file_name, state_jacobian_u_SparseAvailable_list = \
            create_and_write_state_measurement_jacobian_code(
                state_jacobian_u_file_name_without_ext,
                "State_Jacobian_u_Type")

        # measurement jacobian x function code
        measurement_jacobian_x_file_name_without_ext = \
            cost_matrices.measurement_jacobian_x_code_file_name.split(".")[0]

        measurement_jacobian_x_cpp_file_name, measurement_jacobian_x_SparseAvailable_list = \
            create_and_write_state_measurement_jacobian_code(
                measurement_jacobian_x_file_name_without_ext,
                "Measurement_Jacobian_x_Type")

        # state hessian xx function code
        state_hessian_xx_file_name_without_ext = \
            cost_matrices.hf_xx_code_file_name.split(".")[0]

        state_hessian_xx_cpp_file_name, state_hessian_xx_SparseAvailable_list = \
            create_and_write_state_measurement_hessian_code(
                state_hessian_xx_file_name_without_ext,
                "State_Hessian_xx_Type")

        pass
