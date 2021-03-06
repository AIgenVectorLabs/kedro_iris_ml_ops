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

"""Example code for the nodes in the example pipeline. This code is meant
just for illustrating basic Kedro features.

Delete this when you start working on your own Kedro project.
"""

from kedro.pipeline import Pipeline, node

from .nodes import split_data, predict_input_validation, train_standard_scaler, apply_standard_scaler


def create_train_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                split_data,
                ["iris_data", "params:test_data_ratio"],
                dict(
                    train_x="train_x",
                    train_y="train_y",
                    test_x="test_x",
                    test_y="test_y",
                ),
                name="split",
            ),

            node(
                train_standard_scaler,
                "train_x",
                "scaler"
            ),

            node(
                apply_standard_scaler,
                dict(data="train_x", scaler="scaler"),
                "scaled_train_x"
            ),

            node(
                apply_standard_scaler,
                dict(data="test_x", scaler="scaler"),
                "scaled_test_x"
            ),
        ]
    )


def create_predict_pipeline(**kwargs):
    return Pipeline(
        [
            node(predict_input_validation, "params:prediction_input", "predict_df"),
            node(
                apply_standard_scaler,
                dict(data="predict_df", scaler="scaler"),
                "scaled_predict_df"
            ),
        ]
    )
