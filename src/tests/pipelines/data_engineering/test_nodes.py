# Copyright 2021 QuantumBlack Visual Analytics Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND
# NONINFRINGEMENT. IN NO EVENT WILL THE LICENSOR OR OTHER CONTRIBUTORS
# BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF, OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# The QuantumBlack Visual Analytics Limited ("QuantumBlack") name and logo
# (either separately or in combination, "QuantumBlack Trademarks") are
# trademarks of QuantumBlack. The License does not grant you any right or
# license to the QuantumBlack Trademarks. You may not use the QuantumBlack
# Trademarks or any confusingly similar mark as a trademark for your product,
# or use the QuantumBlack Trademarks in any other manner that might cause
# confusion in the marketplace, including but not limited to in advertising,
# on websites, or on software.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This module contains an example test.

Tests should be placed in ``src/tests``, in modules that mirror your
project's structure, and in files named test_*.py. They are simply functions
named ``test_*`` which test a unit of logic.

To run the tests, run ``kedro test``.
"""
from pathlib import Path

import pandas as pd
import pytest

from src.kedro_iris_ml_ops.pipelines.data_engineering.nodes import (
    predict_input_validation,
)


@pytest.fixture
def expected_columns():
    # Define the expected order of columns
    expected_columns = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    return expected_columns


@pytest.fixture
def expected_df():
    """
    Expected input into prediction pipeline
    """
    # Define data
    prediction_input_dict = {
        "sepal_length": [5.1, 5.0, 4.5, 4.8, 6.2, 1],
        "sepal_width": [3.2, 3.3, 3.5, 3.7, 3.2, 1],
        "petal_length": [1.3, 1.3, 1.4, 1.4, 5.2, 1],
        "petal_width": [0.2, 0.2, 0.2, 0.2, 2.0, 1],
    }

    # Create a Pandas data frame
    prediction_input_df = pd.DataFrame(prediction_input_dict)

    return prediction_input_df


@pytest.fixture
def unexpected_df(expected_df, expected_columns):
    """
    unexpected_df is a pandas dataframe with the column order reversed
    """
    prediction_input_df = expected_df

    unexpected_columns = expected_columns.copy()
    unexpected_columns.reverse()

    prediction_input_df = prediction_input_df[unexpected_columns]

    return prediction_input_df


# Test ordering of columns by function predict_input_validation():
class TestPredictInputValidationNode:
    def test_predict_input_validation_on_expected_df(
        self, expected_df, expected_columns
    ):

        test_output_df = predict_input_validation(expected_df)
        actual_columns = test_output_df.columns

        # Assert the order of columns are identical
        assert len(actual_columns) == len(expected_columns)
        assert all([a == b for a, b in zip(actual_columns, expected_columns)])

    def test_predict_input_validation_on_unexpected_df(
        self, unexpected_df, expected_columns
    ):

        test_output_df = predict_input_validation(unexpected_df)
        actual_columns = test_output_df.columns

        # Assert the order of columns are identical
        assert len(actual_columns) == len(expected_columns)
        assert all([a == b for a, b in zip(actual_columns, expected_columns)])
