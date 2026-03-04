"""
File: common_optimization_deploy.py

Description: This file contains utility functions and classes for generating
C++ code for optimization problems, particularly focusing on min/max limits
and their active sets. It includes functionality to create C++ code snippets
for these limits, which can be used in Model Predictive Control (MPC) deployments.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import sympy as sp
import copy
from dataclasses import fields, is_dataclass

from external_libraries.python_numpy_to_cpp.python_numpy.numpy_deploy import NumpyDeploy
from external_libraries.python_numpy_to_cpp.python_numpy.numpy_deploy import python_to_cpp_types
from external_libraries.MCAP_python_control.python_control.control_deploy import ControlDeploy
from external_libraries.MCAP_python_control.python_control.control_deploy import FunctionExtractor
from external_libraries.python_control_to_cpp.python_control.kalman_filter_deploy import FunctionToCppVisitor

VALUE_IS_ZERO_TOL = 1e-30


def get_active_array(
        min_max_array: np.ndarray) -> np.ndarray:
    """
    Generates a boolean array indicating which elements in the input array are finite.

    Parameters
    ----------
    min_max_array : np.ndarray
        A 2D NumPy array containing numerical values.

    Returns
    -------
    np.ndarray
        A boolean array of the same shape as `min_max_array`,
          where each element is True if the corresponding element in `min_max_array`
            is finite, and False otherwise.
    """
    is_active_array = np.zeros_like(min_max_array, dtype=bool)

    if min_max_array is not None:
        for i in range(min_max_array.shape[0]):
            for j in range(min_max_array.shape[1]):
                if np.isfinite(min_max_array[i, j]):
                    is_active_array[i, j] = True

    return is_active_array


class MinMaxCodeGenerator:
    """
    A class to generate code for min/max limits and their active sets,
    typically used in Model Predictive Control (MPC) deployments.
    Attributes:
        values (np.ndarray): Array of min/max values.
        size (int): Number of elements in the min/max array.
        active_set (np.ndarray or None): Boolean array indicating active limits.
        min_max_name (str): Name identifier for the min/max variable.
        file_name_no_extension (str or None): File name (without extension) for generated code.
    Methods:
        __init__(min_max_array, min_max_name, size):
            Initializes the generator with min/max values and configuration.
        generate_active_set(is_active_function=None, is_active_array=None):
            Generates the active set array using a function or a boolean array.
        create_limits_code(data_type, variable_name, caller_file_name_no_extension):
            Generates C++ code for the limits and returns the file name and its base name.
        write_limits_code(code_text, type_name):
            Appends code for setting active limits to the provided code text.
    """

    def __init__(
            self,
            min_max_array: np.ndarray,
            min_max_name: str,
            size: int
    ):
        if min_max_array is not None:
            self.values = min_max_array
            self.size = self.values.shape[0]
            self.active_set = None
        else:
            self.size = size
            self.values = np.ones((self.size, 1))
            self.active_set = np.zeros((self.size, 1), dtype=bool)

        self.min_max_name = min_max_name
        self.file_name_no_extension = None

    def generate_active_set(
        self,
        is_active_function: callable = None,
        is_active_array: np.ndarray = None
    ):
        """
        Generates and returns the active set for the current object.

        The active set is a boolean numpy array of shape (self.size, 1)
          indicating which elements are active.
        The active set can be determined either by a provided function or by
          a provided boolean array.

        Args:
            is_active_function (callable, optional): A function that takes an index
              (int) and returns True if the element is active, False otherwise.
            is_active_array (np.ndarray, optional): A boolean numpy array of length
              self.size indicating active elements.

        Raises:
            ValueError: If neither is_active_function nor is_active_array is provided.

        Returns:
            np.ndarray: A boolean numpy array of shape (self.size, 1)
              representing the active set.
        """
        if is_active_function is None and is_active_array is None:
            raise ValueError(
                "Either is_active_function or is_active_array must be provided")

        if self.active_set is None:
            self.active_set = np.zeros((self.size, 1), dtype=bool)

            for i in range(self.size):
                if is_active_function is not None and is_active_function(i):
                    self.active_set[i, 0] = True

                elif is_active_array is not None and is_active_array[i]:
                    self.active_set[i, 0] = True

        return self.active_set

    def create_limits_code(
            self,
            data_type,
            variable_name: str,
            caller_file_name_no_extension: str
    ):
        """
        Generates C++ code for the active set limits and returns
          the generated file name and its name without extension.

        Args:
            data_type: The desired NumPy data type for the active set array.
            variable_name (str): The base name for the variable to store the active set.
            caller_file_name_no_extension (str): The file name
              (without extension) to use for the generated C++ code.

        Returns:
            tuple:
                - file_name (str): The name of the generated C++ code file.
                - file_name_no_extension (str): The file name without its extension.

        Side Effects:
            - Sets self.file_name_no_extension to the file name without extension.

        Notes:
            - Uses `exec` and `eval` to dynamically create variables
              and call code generation functions.
            - Relies on `self.active_set` and `self.min_max_name` attributes.
        """
        active_set = np.array(self.active_set, dtype=data_type).reshape(-1, 1)

        locals_map = {
            f"{variable_name}_{self.min_max_name}": copy.deepcopy(active_set),
            "caller_file_name_no_extension": caller_file_name_no_extension
        }
        file_name = eval(
            f"NumpyDeploy.generate_matrix_cpp_code(matrix_in={variable_name}_" +
            self.min_max_name + ", file_name=caller_file_name_no_extension)",
            globals(),
            locals_map
        )

        self.file_name_no_extension = file_name.split(".")[0]

        return file_name, self.file_name_no_extension

    def write_limits_code(
            self,
            code_text: str,
            type_name: str
    ):
        """
        Appends C++ code for setting limit values to the provided code text.

        This method generates code that creates an instance of a min-max limits object,
        and, if an active set is defined and non-zero, sets specific limit values using
        template-based setter calls for each active index.

        Args:
            code_text (str): The initial code text to append to.
            type_name (str): The C++ type name to use for static casting limit values.

        Returns:
            str: The updated code text with generated limit-setting code appended.
        """
        code_text += f"  auto {self.min_max_name} = {self.file_name_no_extension}::make();\n\n"
        if self.active_set is not None and \
                np.linalg.norm(self.active_set) > VALUE_IS_ZERO_TOL:
            for i in range(len(self.active_set)):
                if self.active_set[i]:
                    code_text += f"  {self.min_max_name}.template set<{i}, 0>("
                    code_text += \
                        f"static_cast<{type_name}>({self.values[i, 0]})"
                    code_text += ");\n"
            code_text += "\n"

        return code_text


def create_sparse_matrix_code(
        sparse_available_list: np.ndarray,
        type_name: str,
        matrix_name: str
):
    """
    Generates C++ code for defining a sparse matrix type using template metaprogramming.

    The function creates C++ type aliases for a sparse matrix, where the sparsity pattern is
    specified by a boolean numpy array. The generated code uses custom template types
    (e.g., SparseAvailable, ColumnAvailable, SparseMatrix_Type) and includes the necessary
    namespace declaration.

    Args:
        sparse_available_list (np.ndarray): A 2D numpy array of booleans indicating the
            sparsity pattern of the matrix. Each element represents whether the corresponding
            entry in the matrix is available (True) or not (False).
        type_name (str): The C++ type name for the matrix elements (e.g., "double").
        matrix_name (str): The desired name for the generated sparse matrix type.

    Returns:
        str: A string containing the generated C++ code for the sparse matrix type definition.
    """
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
    """
    Generates C++ header code for a parameter class based on a Python dataclass instance,
    and writes it to a .hpp file.

    The generated C++ class will have public member variables corresponding to the fields
    of the provided dataclass, with their values initialized using static_cast to the specified
    C++ type. The class is wrapped in a namespace named after the file name (without extension),
    and include guards are added.

    Args:
        parameter_object: An instance of a Python dataclass containing parameter names and values.
        value_type_name (str): The name of the value type (Python type) to be mapped to a C++ type.
        file_name_no_extension (str): The base name for the output file
          and namespace (without extension).

    Returns:
        str: The path to the saved .hpp file containing the generated C++ code.

    Raises:
        TypeError: If `parameter_object` is not a dataclass instance.
    """
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
    """
    Generates and writes C++ header code for a state function based on extracted Python functions.

    This function locates a Python file corresponding to the given function name, extracts its functions,
    converts them to C++ code using a visitor pattern, and writes the resulting code to a C++ header file.
    The generated header file includes necessary macros, namespace declarations, and class definitions
    with static member functions representing the converted Python functions.

    Args:
        function_name (str): The base name of the Python file and the C++ namespace/class to generate.

    Returns:
        str: The path to the saved C++ header file.
    """

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
    """
    Generates C++ header code for measurement functions based on Python source code and writes it to a file.

    This function locates the Python file corresponding to the given function name, extracts its functions,
    converts them to C++ code using a visitor pattern, and generates a C++ header file with appropriate
    macros, includes, namespace, and class structure. The generated header file contains static methods
    for the measurement functions and is saved to disk.

    Args:
        function_name (str): The name of the measurement function (used to locate the Python file and
                             as the namespace and header file name).

    Returns:
        str: The path to the saved C++ header file.
    """
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
    """
    Generates C++ header code for state measurement Jacobian functions from Python source code,
    writes the generated code to a file, and returns the file name and sparse matrix availability.

    This function locates the Python file containing the specified function, extracts its code,
    converts the relevant functions to C++ using a visitor pattern, and generates a C++ header
    file with appropriate namespace, includes, and class structure. It also handles sparse matrix
    code generation if applicable.

    Args:
        function_name (str): The name of the function to process and generate code for.
        output_type (str): The output type to be used in the generated C++ code.
        type_name (str): The type name for matrix code generation.

    Returns:
        Tuple[str, List[Any]]: A tuple containing the name of the saved C++ header file and a list
        indicating the availability of sparse matrix representations for the extracted functions.
    """
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
