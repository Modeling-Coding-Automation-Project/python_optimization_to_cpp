import os
import sys
sys.path.append(os.getcwd())

import re
import inspect
import ast
import astor
import numpy as np
import sympy as sp

from external_libraries.MCAP_python_optimization.optimization_utility.sqp_matrix_utility import SQP_CostMatrices_NMPC
from external_libraries.python_numpy_to_cpp.python_numpy.numpy_deploy import NumpyDeploy
from external_libraries.MCAP_python_control.python_control.control_deploy import ControlDeploy
from external_libraries.MCAP_python_control.python_control.control_deploy import ExpressionDeploy
from external_libraries.MCAP_python_control.python_control.control_deploy import FunctionExtractor
from external_libraries.MCAP_python_control.python_control.control_deploy import IntegerPowerReplacer
from external_libraries.python_control_to_cpp.python_control.kalman_filter_deploy import FunctionToCppVisitor


def create_and_write_state_space_function_code(function_name, return_type):

    function_file_path = ControlDeploy.find_file(
        f"{function_name}.py", os.getcwd())
    state_function_U_size = ExpressionDeploy.get_input_size_from_function_code(
        function_file_path)

    extractor = FunctionExtractor(function_file_path)
    functions = extractor.extract()
    state_function_code = []
    SparseAvailable_list = []

    for name, code in functions.items():
        converter = FunctionToCppVisitor(return_type)

        state_function_code.append(converter.convert(code))
        SparseAvailable_list.append(converter.SparseAvailable)

    SparseAvailable_list = [
        x for x in SparseAvailable_list if x is not None]

    # generate code text
    code_text = ""
    header_macro_text = "__" + function_name.upper() + "_HPP__"

    code_text += f"#ifndef {header_macro_text}\n"
    code_text += f"#define {header_macro_text}\n\n"

    code_text += "#include \"python_math.hpp\"\n"
    code_text += "#include \"python_numpy.hpp\"\n\n"

    code_text += f"namespace {function_name} {{\n\n"

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

    return saved_file_name, state_function_U_size, SparseAvailable_list


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
        # state equation function code
        state_function_file_name_without_ext = \
            cost_matrices.state_function_code_file_name.split(".")[0]

        state_function_cpp_file_name, state_function_U_size, _ = \
            create_and_write_state_space_function_code(
                state_function_file_name_without_ext, "X_Type")
