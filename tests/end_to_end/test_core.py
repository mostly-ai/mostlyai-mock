# Copyright 2025 MOSTLY AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from unittest.mock import patch

import litellm
import pandas as pd

from mostlyai import mock

litellm_completion = litellm.completion


def test_single_table():
    def litellm_completion_with_mock_response(*args, **kwargs):
        mock_response = {
            "rows": {
                "guest_id": 1,
                "nationality": "US",
                "name": "John Doe",
                "gender": "male",
                "age": 25,
                "date_of_birth": "1990-01-01",
                "checkin_time": "2025-05-01 10:00:00",
                "is_vip": True,
                "price_per_night": 100.0,
                "room_number": 101,
            }
        }
        return litellm_completion(*args, **kwargs, mock_response=json.dumps(mock_response))

    tables = {
        "guests": {
            "prompt": "Guests of an Alpine ski hotel in Austria",
            "columns": {
                "guest_id": {"prompt": "the unique id of the guest", "dtype": "integer"},
                "nationality": {"prompt": "2-letter code for the nationality", "dtype": "string"},
                "name": {"prompt": "first name and last name of the guest", "dtype": "string"},
                "gender": {"dtype": "category", "values": ["male", "female"]},
                "age": {"prompt": "age in years; min: 18, max: 80; avg: 25", "dtype": "integer"},
                "date_of_birth": {"prompt": "date of birth", "dtype": "date"},
                "checkin_time": {"prompt": "the check in timestamp of the guest; may 2025", "dtype": "datetime"},
                "is_vip": {"prompt": "is the guest a VIP", "dtype": "boolean"},
                "price_per_night": {"prompt": "price paid per night, in EUR", "dtype": "float"},
                "room_number": {
                    "prompt": "room number",
                    "dtype": "integer",
                    "values": [101, 102, 103, 201, 202, 203, 204],
                },
            },
            "primary_key": "guest_id",
        }
    }
    with patch("mostlyai.mock.core.litellm.completion", side_effect=litellm_completion_with_mock_response):
        df = mock.sample(tables=tables, sample_size=5, model="openai/gpt-4.1-nano")
        assert df.shape == (5, 10)
        assert df.dtypes.to_dict() == {
            "guest_id": "int64[pyarrow]",
            "nationality": "string[pyarrow]",
            "name": "string[pyarrow]",
            "gender": pd.CategoricalDtype(categories=["male", "female"]),
            "age": "int64[pyarrow]",
            "date_of_birth": "datetime64[ns]",
            "checkin_time": "datetime64[ns]",
            "is_vip": "boolean[pyarrow]",
            "price_per_night": "float64[pyarrow]",
            "room_number": "int64[pyarrow]",
        }
