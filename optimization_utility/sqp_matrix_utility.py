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
