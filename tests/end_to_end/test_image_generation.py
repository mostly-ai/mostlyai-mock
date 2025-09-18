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

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest

from mostlyai import mock
from mostlyai.mock.core import DType


def test_image_column_validation():
    """Test that image columns are properly validated."""
    # Valid image column configuration
    tables = {
        "users": {
            "columns": {
                "user_id": {"dtype": "string"},
                "name": {"dtype": "string"},
                "profile_photo": {
                    "prompt": "A professional headshot photo", 
                    "dtype": "image",
                    "image_model": "openai/dall-e-3",
                    "image_size": "512x512"
                }
            },
            "primary_key": "user_id"
        }
    }
    
    # Should not raise validation errors
    config = mock.core.MockConfig(tables)
    assert "profile_photo" in config.root["users"].columns
    assert config.root["users"].columns["profile_photo"].dtype == DType.IMAGE


def test_image_column_validation_missing_prompt():
    """Test that image columns without prompts raise validation errors."""
    tables = {
        "users": {
            "columns": {
                "profile_photo": {
                    "dtype": "image"
                }
            }
        }
    }
    
    with pytest.raises(ValueError, match="Image columns must have a prompt"):
        mock.core.MockConfig(tables)


def test_image_column_validation_invalid_size():
    """Test that image columns with invalid size format raise validation errors."""
    tables = {
        "users": {
            "columns": {
                "profile_photo": {
                    "prompt": "A photo", 
                    "dtype": "image",
                    "image_size": "invalid_size"
                }
            }
        }
    }
    
    with pytest.raises(ValueError, match="image_size must be in format 'widthxheight'"):
        mock.core.MockConfig(tables)


@pytest.mark.asyncio
async def test_image_generation_integration():
    """Test the end-to-end image generation functionality."""
    
    def mock_litellm_completion(*args, **kwargs):
        # Mock text data generation
        mock_response = '{"rows": [{"user_id": "user_1", "name": "John Doe", "age": 25}]}'
        return mock.litellm.completion(*args, **kwargs, mock_response=mock_response)
    
    async def mock_image_generation(*args, **kwargs):
        # Mock image generation - return fake image data
        return b"fake_image_data"
    
    tables = {
        "users": {
            "prompt": "Users of a social media platform",
            "columns": {
                "user_id": {"dtype": "string"},
                "name": {"dtype": "string"},
                "age": {"dtype": "integer"},
                "profile_photo": {
                    "prompt": "A professional headshot photo of the user", 
                    "dtype": "image",
                    "image_model": "openai/dall-e-3",
                    "image_size": "512x512"
                }
            },
            "primary_key": "user_id"
        }
    }
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Patch the necessary functions
        with patch("mostlyai.mock.core.litellm.acompletion", side_effect=mock_litellm_completion), \
             patch("mostlyai.mock.image_generator.ImageGenerator._generate_image_content", 
                   side_effect=mock_image_generation):
            
            df = await mock.core._sample_common(
                tables=tables, 
                sample_size=2
            )
            
            # Verify the DataFrame structure
            assert len(df) == 2
            assert list(df.columns) == ["user_id", "name", "age", "profile_photo"]
            
            # Check data types
            assert df.dtypes["user_id"] == "string[pyarrow]"
            assert df.dtypes["name"] == "string[pyarrow]" 
            assert df.dtypes["age"] == "int64[pyarrow]"
            assert df.dtypes["profile_photo"] == "string[pyarrow]"
            
            # Verify that image file paths were generated
            for _, row in df.iterrows():
                profile_photo_path = row["profile_photo"]
                if profile_photo_path is not None:  # Check if image generation succeeded
                    assert isinstance(profile_photo_path, str)
                    assert profile_photo_path.endswith(".png")


def test_image_columns_without_dependencies():
    """Test that image columns without image generation dependencies show warning."""
    tables = {
        "users": {
            "columns": {
                "name": {"dtype": "string"},
                "profile_photo": {
                    "prompt": "A photo", 
                    "dtype": "image"
                }
            }
        }
    }
    
    # Mock the image generation as unavailable
    with patch("mostlyai.mock.core._IMAGE_GENERATION_AVAILABLE", False), \
         patch("mostlyai.mock.core.litellm.acompletion") as mock_completion:
        
        # Setup mock for text generation
        mock_completion.return_value = AsyncMock()
        mock_completion.return_value.__aiter__ = AsyncMock(return_value=iter([]))
        
        # This should work but show a warning
        df = mock.sample(tables=tables, sample_size=1)
        
        # Image column should be present but with None values
        assert "profile_photo" in df.columns
        assert df["profile_photo"].isna().all()


def test_mixed_columns_with_images():
    """Test tables with mix of regular and image columns."""
    def mock_litellm_completion(*args, **kwargs):
        mock_response = '{"rows": [{"guest_id": "G001", "name": "Alice Smith"}]}'
        return mock.litellm.completion(*args, **kwargs, mock_response=mock_response)
    
    tables = {
        "guests": {
            "prompt": "Hotel guests",
            "columns": {
                "guest_id": {"dtype": "string"},
                "name": {"dtype": "string"},
                "photo": {
                    "prompt": "Guest photo for ID purposes", 
                    "dtype": "image"
                },
                "room_photo": {
                    "prompt": "Photo of the guest's room", 
                    "dtype": "image"
                }
            },
            "primary_key": "guest_id"
        }
    }
    
    with patch("mostlyai.mock.core.litellm.acompletion", side_effect=mock_litellm_completion), \
         patch("mostlyai.mock.image_generator.ImageGenerator._generate_image_content", 
               return_value=b"mock_image"):
        
        df = mock.sample(tables=tables, sample_size=1)
        
        # Verify all columns are present in correct order
        assert list(df.columns) == ["guest_id", "name", "photo", "room_photo"]
        assert len(df) == 1
        
        # Check that text columns have proper data
        assert df.iloc[0]["guest_id"] == "G001"
        assert df.iloc[0]["name"] == "Alice Smith"


def test_only_image_columns():
    """Test tables with only image columns."""
    tables = {
        "art_gallery": {
            "columns": {
                "artwork": {
                    "prompt": "A beautiful painting", 
                    "dtype": "image"
                },
                "frame": {
                    "prompt": "An elegant picture frame", 
                    "dtype": "image"
                }
            }
        }
    }
    
    with patch("mostlyai.mock.image_generator.ImageGenerator._generate_image_content", 
               return_value=b"mock_image"):
        
        df = mock.sample(tables=tables, sample_size=2)
        
        assert list(df.columns) == ["artwork", "frame"]
        assert len(df) == 2