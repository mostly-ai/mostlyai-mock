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

import asyncio
import base64
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import httpx
import litellm
import pandas as pd

from mostlyai.mock.core import ColumnConfig, DType


class ImageGenerator:
    """Handles generation of synthetic images using various AI image generation services."""
    
    def __init__(self, base_output_dir: str = "generated_images"):
        self.base_output_dir = base_output_dir
        
    def _create_output_directory(self, table_name: str) -> Path:
        """Create a timestamped output directory for the table."""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = Path(self.base_output_dir) / f"{table_name}_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    
    def _build_image_prompt(self, column_config: ColumnConfig, row_data: Dict[str, Any]) -> str:
        """Build a comprehensive image generation prompt using column config and row context."""
        base_prompt = column_config.prompt or "Generate an image"
        
        # Add context from other columns in the row to make images more relevant
        context_parts = []
        for col_name, value in row_data.items():
            if value is not None and str(value).strip():
                # Skip the image column itself and any other image columns
                if col_name != column_config and not str(value).startswith(("/", "generated_images")):
                    context_parts.append(f"{col_name}: {value}")
        
        if context_parts:
            context_str = ", ".join(context_parts[:5])  # Limit to first 5 context items
            enhanced_prompt = f"{base_prompt}. Context: {context_str}"
        else:
            enhanced_prompt = base_prompt
            
        return enhanced_prompt
    
    async def _generate_with_openai_dalle(
        self, 
        prompt: str, 
        model: str = "dall-e-3", 
        size: str = "1024x1024"
    ) -> bytes:
        """Generate image using OpenAI DALL-E."""
        try:
            response = await litellm.aimage_generation(
                model=f"openai/{model}",
                prompt=prompt,
                size=size,
                quality="standard",
                n=1
            )
            
            image_url = response.data[0].url
            
            # Download the image
            async with httpx.AsyncClient() as client:
                image_response = await client.get(image_url)
                image_response.raise_for_status()
                return image_response.content
                
        except Exception as e:
            raise RuntimeError(f"Failed to generate image with OpenAI DALL-E: {str(e)}")
    
    async def _generate_with_stability_ai(
        self,
        prompt: str,
        model: str = "stable-diffusion-xl-1024-v1-0",
        size: str = "1024x1024"
    ) -> bytes:
        """Generate image using Stability AI."""
        try:
            # Parse size
            width, height = map(int, size.split("x"))
            
            response = await litellm.aimage_generation(
                model=f"stability-ai/{model}",
                prompt=prompt,
                width=width,
                height=height,
                n=1
            )
            
            # Stability AI returns base64 encoded images
            if hasattr(response.data[0], 'b64_json'):
                image_data = base64.b64decode(response.data[0].b64_json)
                return image_data
            else:
                # Fallback to URL if b64_json not available
                image_url = response.data[0].url
                async with httpx.AsyncClient() as client:
                    image_response = await client.get(image_url)
                    image_response.raise_for_status()
                    return image_response.content
                    
        except Exception as e:
            raise RuntimeError(f"Failed to generate image with Stability AI: {str(e)}")
    
    async def _generate_image_content(
        self, 
        prompt: str, 
        image_model: str, 
        image_size: str
    ) -> bytes:
        """Generate image content using the specified model."""
        if image_model.startswith("openai/"):
            model_name = image_model.split("/")[-1]
            return await self._generate_with_openai_dalle(prompt, model_name, image_size)
        elif image_model.startswith("stability-ai/"):
            model_name = image_model.split("/")[-1]
            return await self._generate_with_stability_ai(prompt, model_name, image_size)
        else:
            # Fallback to OpenAI DALL-E
            return await self._generate_with_openai_dalle(prompt, "dall-e-3", image_size)
    
    def _save_image(
        self, 
        image_content: bytes, 
        output_dir: Path, 
        filename_base: str,
        image_format: str = "png"
    ) -> str:
        """Save image content to file and return the file path."""
        filename = f"{filename_base}.{image_format}"
        filepath = output_dir / filename
        
        with open(filepath, "wb") as f:
            f.write(image_content)
            
        return str(filepath)
    
    async def generate_images_for_dataframe(
        self, 
        df: pd.DataFrame,
        table_name: str, 
        image_columns: Dict[str, ColumnConfig],
        primary_key: str | None = None
    ) -> pd.DataFrame:
        """Generate images for all image columns in a DataFrame."""
        if not image_columns:
            return df
            
        df_copy = df.copy()
        output_dir = self._create_output_directory(table_name)
        
        # Generate images for each row
        for idx, row in df_copy.iterrows():
            row_data = row.to_dict()
            
            # Generate an identifier for this row
            if primary_key and primary_key in row_data:
                row_id = str(row_data[primary_key])
            else:
                row_id = f"row_{idx}"
                
            # Process each image column
            for col_name, column_config in image_columns.items():
                try:
                    # Build image prompt with context
                    image_prompt = self._build_image_prompt(column_config, row_data)
                    
                    # Generate image
                    image_content = await self._generate_image_content(
                        prompt=image_prompt,
                        image_model=column_config.image_model,
                        image_size=column_config.image_size
                    )
                    
                    # Save image with descriptive filename
                    filename_base = f"{row_id}_{col_name}"
                    filepath = self._save_image(image_content, output_dir, filename_base)
                    
                    # Store file path in DataFrame
                    df_copy.at[idx, col_name] = filepath
                    
                except Exception as e:
                    print(f"Warning: Failed to generate image for {col_name} in row {idx}: {e}")
                    # Store error placeholder or None
                    df_copy.at[idx, col_name] = None
                    
                # Add small delay to avoid rate limiting
                await asyncio.sleep(0.1)
        
        return df_copy