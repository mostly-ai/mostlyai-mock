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

from unittest.mock import patch

import litellm
import pandas as pd
import pytest

from mostlyai import mock

litellm_completion = litellm.completion


def test_single_table():
    def litellm_completion_with_mock_response(*args, **kwargs):
        # remove unsupported params for mock
        kwargs.pop("reasoning_effort", None)
        mock_response = '{"rows": [{"guest_id": 1, "nationality": "US", "name": "John Doe", "gender": "male", "age": 25, "date_of_birth": "1990-01-01", "checkin_time": "2025-05-01 10:00:00", "is_vip": true, "price_per_night": 100.0, "room_number": 101}]}'
        return litellm_completion(*args, **kwargs, mock_response=mock_response)

    tables = {
        "guests": {
            "prompt": "Guests of an Alpine ski hotel in Austria",
            "columns": {
                "guest_id": {"prompt": "the unique id of the guest", "dtype": "string"},
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
    with patch("mostlyai.mock.core.litellm.acompletion", side_effect=litellm_completion_with_mock_response):
        df = mock.sample(tables=tables, sample_size=5)
        assert df.shape == (5, 10)
        assert df.dtypes.to_dict() == {
            "guest_id": "string[pyarrow]",
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


def test_retries():
    def litellm_completion_with_mock_response(*args, **kwargs):
        # remove unsupported params for mock
        kwargs.pop("reasoning_effort", None)
        mock_response = '{"rows": [{"name": "John Doe"}]}'
        return litellm_completion(*args, **kwargs, mock_response=mock_response)

    tables = {
        "guests": {
            "columns": {
                "name": {"dtype": "string"},
                "age": {"dtype": "integer"},
            }
        }
    }
    with patch("mostlyai.mock.core.litellm.acompletion", side_effect=litellm_completion_with_mock_response):
        with pytest.raises(RuntimeError) as e:
            mock.sample(tables=tables, sample_size=30)
        assert "Too many malformed batches were generated" in str(e.value)


def test_existing_data():
    # all columns are present in the existing data => no LLM calls should be made
    tables = {
        "guests": {
            "columns": {
                "name": {"dtype": "string"},
                "age": {"dtype": "integer"},
            }
        }
    }
    with patch("mostlyai.mock.core.litellm.acompletion") as mock_acompletion:
        existing_guests = pd.DataFrame({"name": ["John Doe"], "age": [25]})
        df = mock.sample(tables=tables, existing_data={"guests": existing_guests})
        pd.testing.assert_frame_equal(df, existing_guests, check_dtype=False)
        mock_acompletion.assert_not_called()


def test_auto_increment_with_foreign_keys():
    # test auto-increment integer PKs with self-referencing FK: 2 workers, 4 user rows, 4 task rows
    # note: LLM generates string PKs which are then remapped to integers in post-processing
    batch = 0
    n_workers = 2

    def litellm_completion_with_mock_response(*args, **kwargs):
        nonlocal batch
        batch += 1
        # remove unsupported params for mock
        kwargs.pop("reasoning_effort", None)
        if batch <= n_workers:
            # mock users: each worker produces 2 rows
            mock_response = f'{{"rows": [{{"id": "B{batch}-1", "name": "Alice", "manager_id": null}}, {{"id": "B{batch}-2", "name": "Bob", "manager_id": "B{batch}-1"}}]}}'
        else:
            # mock tasks: return 1 row per batch, referencing users sequentially
            row_idx = batch - n_workers  # 1, 2, 3, 4
            # map row_idx to user string: 1->B1-1, 2->B1-2, 3->B2-1, 4->B2-2
            batch_num = ((row_idx - 1) // 2) + 1
            user_num = ((row_idx - 1) % 2) + 1
            mock_response = (
                f'{{"rows": [{{"id": "B{row_idx}-1", "name": "Training", "user_id": "B{batch_num}-{user_num}"}}]}}'
            )
        return litellm_completion(*args, **kwargs, mock_response=mock_response)

    tables = {
        "users": {
            "columns": {
                "id": {"dtype": "integer"},
                "name": {"dtype": "string"},
                "manager_id": {"dtype": "integer"},
            },
            "primary_key": "id",
            "foreign_keys": [{"column": "manager_id", "referenced_table": "users", "prompt": "manager of the user"}],
        },
        "tasks": {
            "columns": {
                "id": {"dtype": "integer"},
                "name": {"dtype": "string"},
                "user_id": {"dtype": "integer"},
            },
            "primary_key": "id",
            "foreign_keys": [{"column": "user_id", "referenced_table": "users", "prompt": "user of the task"}],
        },
    }

    with patch("mostlyai.mock.core.litellm.acompletion", side_effect=litellm_completion_with_mock_response):
        result = mock.sample(tables=tables, sample_size=4, n_workers=n_workers)
        # check expected structure: auto-increment ids 1-4 for users (remapped from strings)
        # manager_ids should also be remapped (B1-1->1, B2-1->3, etc.)
        # use pd.testing.assert_frame_equal for better dtype handling
        pd.testing.assert_frame_equal(
            result["users"],
            pd.DataFrame(
                {
                    "id": [1, 2, 3, 4],
                    "name": ["Alice", "Bob", "Alice", "Bob"],
                    "manager_id": [None, 1.0, None, 3.0],
                }
            ),
            check_dtype=False,
        )
        # tasks reference users sequentially: T1->B1-1 (id=1), T2->B1-2 (id=2), T3->B2-1 (id=3), T4->B2-2 (id=4)
        pd.testing.assert_frame_equal(
            result["tasks"],
            pd.DataFrame(
                {
                    "id": [1, 2, 3, 4],
                    "name": ["Training"] * 4,
                    "user_id": [1, 2, 3, 4],
                }
            ),
            check_dtype=False,
        )
