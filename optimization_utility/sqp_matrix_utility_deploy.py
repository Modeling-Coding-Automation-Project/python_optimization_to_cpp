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

from optimization_utility.common_optimization_deploy import MinMaxCodeGenerator
from optimization_utility.common_optimization_deploy import get_active_array

from external_libraries.MCAP_python_optimization.optimization_utility.sqp_matrix_utility import SQP_CostMatrices_NMPC
from external_libraries.python_numpy_to_cpp.python_numpy.numpy_deploy import NumpyDeploy
from external_libraries.python_numpy_to_cpp.python_numpy.numpy_deploy import python_to_cpp_types
from external_libraries.MCAP_python_control.python_control.control_deploy import ControlDeploy
from external_libraries.MCAP_python_control.python_control.control_deploy import ExpressionDeploy
from external_libraries.MCAP_python_control.python_control.control_deploy import FunctionExtractor
from external_libraries.MCAP_python_control.python_control.control_deploy import IntegerPowerReplacer
from external_libraries.python_control_to_cpp.python_control.kalman_filter_deploy import FunctionToCppVisitor


def create_sparse_matrix_code(
        sparse_available_list: np.ndarray,
        type_name: str,
        matrix_name: str
):
    code_text = ""

    code_text = "using namespace PythonNumpy;\n\n"

    sparse_available_name = matrix_name + "_SparseAvailable"
    code_text += "using " + sparse_available_name + \
        " = SparseAvailable<\n"

    for i in range(sparse_available_list.shape[0]):
        code_text += "    ColumnAvailable<"
        for j in range(sparse_available_list.shape[1]):

            if True == sparse_available_list[i, j]:
                code_text += "true"
            else:
                code_text += "false"
            if j != sparse_available_list.shape[1] - 1:
                code_text += ", "
        if i == sparse_available_list.shape[0] - 1:
            code_text += ">\n"
        else:
            code_text += ">,\n"

    code_text += ">;\n\n"

    code_text += "using " + matrix_name + " = SparseMatrix_Type<" + \
        type_name + ", " + \
        sparse_available_name + ">;\n\n"

    return code_text


def create_and_write_parameter_class_code(
        parameter_object,
        value_type_name: str,
        file_name_no_extension: str
):

    if not is_dataclass(parameter_object):
        raise TypeError("parameter_object must be a dataclass instance")

    file_path = ControlDeploy.find_file(
        f"{file_name_no_extension}.py", os.getcwd())

    try:
        value_type_name = python_to_cpp_types[value_type_name]
    except KeyError:
        pass

    code_text = ""
    header_macro_text = "__" + file_name_no_extension.upper() + "_HPP__"

    code_text += f"#ifndef {header_macro_text}\n"
    code_text += f"#define {header_macro_text}\n\n"

    code_text += f"namespace {file_name_no_extension} {{\n\n"

    code_text += "class Parameter {\n"
    code_text += "public:\n"

    name_value_pairs = [(f.name, getattr(parameter_object, f.name))
                        for f in fields(parameter_object)]
    for _, name_value in enumerate(name_value_pairs):
        code_text += f"  {value_type_name} {name_value[0]} = static_cast<{value_type_name}>({name_value[1]});\n"

    code_text += "};\n\n"

    code_text += f"}} // namespace {file_name_no_extension}\n\n"

    code_text += f"#endif // {header_macro_text}\n"

    saved_file_name = ControlDeploy.write_to_file(
        code_text, f"{file_name_no_extension}.hpp")

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
        output_type: str,
        type_name: str,
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

    code_text += "#include \"python_math.hpp\"\n"
    code_text += "#include \"python_numpy.hpp\"\n\n"

    code_text += f"namespace {function_name} {{\n\n"

    code_text += "using namespace PythonMath;\n"

    code_text += create_sparse_matrix_code(
        SparseAvailable_list[0], type_name, output_type)

    code_text += "template <typename X_Type, typename U_Type, " + \
        " typename Parameter_Type>\n"
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
        output_type: str,
        type_name: str,
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

    code_text += "#include \"python_math.hpp\"\n"
    code_text += "#include \"python_numpy.hpp\"\n\n"

    code_text += f"namespace {function_name} {{\n\n"

    code_text += "using namespace PythonMath;\n"

    code_text += create_sparse_matrix_code(
        SparseAvailable_list[0], type_name, output_type)

    code_text += "template <typename X_Type, typename U_Type, " + \
        " typename Parameter_Type>\n"
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

        # state equation function code
        state_function_file_name_no_extension = \
            cost_matrices.state_function_code_file_name.split(".")[0]

        state_function_cpp_file_name = \
            create_and_write_state_function_code(
                state_function_file_name_no_extension)

        # measurement equation function code
        measurement_function_file_name_no_extension = \
            cost_matrices.measurement_function_code_file_name.split(".")[0]

        measurement_function_cpp_file_name = \
            create_and_write_measurement_function_code(
                measurement_function_file_name_no_extension)

        # state jacobian x function code
        state_jacobian_x_file_name_no_extension = \
            cost_matrices.state_jacobian_x_code_file_name.split(".")[0]

        state_jacobian_x_cpp_file_name, state_jacobian_x_SparseAvailable_list = \
            create_and_write_state_measurement_jacobian_code(
                state_jacobian_x_file_name_no_extension,
                "State_Jacobian_x_Type",
                type_name)

        # state jacobian u function code
        state_jacobian_u_file_name_no_extension = \
            cost_matrices.state_jacobian_u_code_file_name.split(".")[0]

        state_jacobian_u_cpp_file_name, state_jacobian_u_SparseAvailable_list = \
            create_and_write_state_measurement_jacobian_code(
                state_jacobian_u_file_name_no_extension,
                "State_Jacobian_u_Type",
                type_name)

        # measurement jacobian x function code
        measurement_jacobian_x_file_name_no_extension = \
            cost_matrices.measurement_jacobian_x_code_file_name.split(".")[0]

        measurement_jacobian_x_cpp_file_name, measurement_jacobian_x_SparseAvailable_list = \
            create_and_write_state_measurement_jacobian_code(
                measurement_jacobian_x_file_name_no_extension,
                "Measurement_Jacobian_x_Type",
                type_name)

        # state hessian xx function code
        state_hessian_xx_file_name_no_extension = \
            cost_matrices.state_hessian_xx_code_file_name.split(".")[0]

        state_hessian_xx_cpp_file_name, state_hessian_xx_SparseAvailable_list = \
            create_and_write_state_measurement_hessian_code(
                state_hessian_xx_file_name_no_extension,
                "State_Hessian_xx_Type",
                type_name)

        # state hessian xu function code
        state_hessian_xu_file_name_no_extension = \
            cost_matrices.state_hessian_xu_code_file_name.split(".")[0]

        state_hessian_xu_cpp_file_name, state_hessian_xu_SparseAvailable_list = \
            create_and_write_state_measurement_hessian_code(
                state_hessian_xu_file_name_no_extension,
                "State_Hessian_xu_Type",
                type_name)

        # state hessian ux function code
        state_hessian_ux_file_name_no_extension = \
            cost_matrices.state_hessian_ux_code_file_name.split(".")[0]

        state_hessian_ux_cpp_file_name, state_hessian_ux_SparseAvailable_list = \
            create_and_write_state_measurement_hessian_code(
                state_hessian_ux_file_name_no_extension,
                "State_Hessian_ux_Type",
                type_name)

        # state hessian uu function code
        state_hessian_uu_file_name_no_extension = \
            cost_matrices.state_hessian_uu_code_file_name.split(".")[0]

        state_hessian_uu_cpp_file_name, state_hessian_uu_SparseAvailable_list = \
            create_and_write_state_measurement_hessian_code(
                state_hessian_uu_file_name_no_extension,
                "State_Hessian_uu_Type",
                type_name)

        # measurement hessian xx function code
        measurement_hessian_xx_file_name_no_extension = \
            cost_matrices.measurement_hessian_xx_code_file_name.split(".")[0]

        measurement_hessian_xx_cpp_file_name, measurement_hessian_xx_SparseAvailable_list = \
            create_and_write_state_measurement_hessian_code(
                measurement_hessian_xx_file_name_no_extension,
                "Measurement_Hessian_xx_Type",
                type_name)

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
        code_text += f"#include \"{measurement_jacobian_x_cpp_file_name}\"\n"
        code_text += f"#include \"{state_hessian_xx_cpp_file_name}\"\n"
        code_text += f"#include \"{state_hessian_xu_cpp_file_name}\"\n"
        code_text += f"#include \"{state_hessian_ux_cpp_file_name}\"\n"
        code_text += f"#include \"{state_hessian_uu_cpp_file_name}\"\n"
        code_text += f"#include \"{measurement_hessian_xx_cpp_file_name}\"\n\n"

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

        code_text += f"using State_Hessian_XX_Matrix_Type = {state_hessian_xx_file_name_no_extension}" + \
            "::State_Hessian_xx_Type;\n"
        code_text += f"using State_Hessian_XU_Matrix_Type = {state_hessian_xu_file_name_no_extension}" + \
            "::State_Hessian_xu_Type;\n"
        code_text += f"using State_Hessian_UX_Matrix_Type = {state_hessian_ux_file_name_no_extension}" + \
            "::State_Hessian_ux_Type;\n"
        code_text += f"using State_Hessian_UU_Matrix_Type = {state_hessian_uu_file_name_no_extension}" + \
            "::State_Hessian_uu_Type;\n"
        code_text += f"using Measurement_Hessian_XX_Matrix_Type = {measurement_hessian_xx_file_name_no_extension}" + \
            "::Measurement_Hessian_xx_Type;\n\n"

        code_text += "PythonOptimization::StateFunction_Object<X_Type, U_Type, Parameter_Type> state_function =\n" + \
            f"    {state_function_file_name_no_extension}::Function<" + \
            "X_Type, U_Type, Parameter_Type>::function;\n\n"

        code_text += "} // namespace " + namespace_name + "\n\n"

        code_text += "#endif // " + file_header_macro_name + "\n"

        code_file_name_ext = ControlDeploy.write_to_file(
            code_text, code_file_name_ext)

        deployed_file_names.append(code_file_name_ext)

        return deployed_file_names
