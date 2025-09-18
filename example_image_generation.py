#!/usr/bin/env python3
"""
Example demonstrating the image generation functionality in mostlyai-mock.

This script shows how to generate synthetic data that includes both structured 
data and AI-generated images linked to mock entities.
"""

import os
from mostlyai import mock

# Set your API keys (you'll need these for the LLM and image generation)
# os.environ["OPENAI_API_KEY"] = "your-openai-api-key-here"

def example_single_table_with_images():
    """Example: Generate a table of hotel guests with profile photos."""
    print("=== Single Table with Images Example ===")
    
    tables = {
        "guests": {
            "prompt": "Guests of an Alpine ski hotel in Austria",
            "columns": {
                "guest_id": {
                    "prompt": "the unique id of the guest", 
                    "dtype": "string"
                },
                "name": {
                    "prompt": "first name and last name of the guest", 
                    "dtype": "string"
                },
                "nationality": {
                    "prompt": "2-letter code for the nationality", 
                    "dtype": "string"
                },
                "age": {
                    "prompt": "age in years; min: 18, max: 80; avg: 35", 
                    "dtype": "integer"
                },
                "profile_photo": {
                    "prompt": "A professional headshot photo of the guest suitable for hotel ID purposes",
                    "dtype": "image",
                    "image_model": "openai/dall-e-3",
                    "image_size": "512x512",
                    "output_dir": "generated_images/guests"
                },
                "room_photo": {
                    "prompt": "A photo of the guest's luxury hotel room with Alpine mountain views",
                    "dtype": "image",
                    "image_model": "openai/dall-e-3",
                    "image_size": "1024x1024",
                    "output_dir": "generated_images/rooms"
                }
            },
            "primary_key": "guest_id",
        }
    }
    
    try:
        df = mock.sample(
            tables=tables,
            sample_size=3,
            model="openai/gpt-4o-mini"  # Use a fast model for text generation
        )
        
        print("Generated data:")
        print(df)
        print("\nColumns and types:")
        print(df.dtypes)
        
        # Show image file paths
        print("\nGenerated image files:")
        for idx, row in df.iterrows():
            print(f"Guest {row['name']} ({row['guest_id']}):")
            print(f"  Profile photo: {row['profile_photo']}")
            print(f"  Room photo: {row['room_photo']}")
            
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have set your OPENAI_API_KEY environment variable")


def example_multi_table_with_images():
    """Example: Generate related tables with images (products and reviews)."""
    print("\n=== Multi-Table with Images Example ===")
    
    tables = {
        "products": {
            "prompt": "Products in an online electronics store",
            "columns": {
                "product_id": {
                    "prompt": "unique product identifier", 
                    "dtype": "string"
                },
                "name": {
                    "prompt": "product name", 
                    "dtype": "string"
                },
                "category": {
                    "dtype": "category",
                    "values": ["laptop", "smartphone", "tablet", "headphones", "camera"]
                },
                "price": {
                    "prompt": "price in USD", 
                    "dtype": "float"
                },
                "product_image": {
                    "prompt": "A high-quality product photo suitable for e-commerce, clean white background",
                    "dtype": "image",
                    "image_model": "openai/dall-e-3",
                    "image_size": "1024x1024"
                }
            },
            "primary_key": "product_id",
        },
        "reviews": {
            "prompt": "Customer reviews for products",
            "columns": {
                "review_id": {
                    "prompt": "unique review identifier", 
                    "dtype": "string"
                },
                "product_id": {
                    "prompt": "the product being reviewed", 
                    "dtype": "string"
                },
                "reviewer_name": {
                    "prompt": "name of the person writing the review", 
                    "dtype": "string"
                },
                "rating": {
                    "prompt": "star rating from 1 to 5", 
                    "dtype": "integer",
                    "values": [1, 2, 3, 4, 5]
                },
                "review_text": {
                    "prompt": "detailed review text", 
                    "dtype": "string"
                },
                "reviewer_photo": {
                    "prompt": "Profile photo of the reviewer, diverse and realistic",
                    "dtype": "image",
                    "image_model": "openai/dall-e-3",
                    "image_size": "256x256"
                }
            },
            "primary_key": "review_id",
            "foreign_keys": [
                {
                    "column": "product_id",
                    "referenced_table": "products",
                    "prompt": "each product has 2-3 reviews"
                }
            ],
        }
    }
    
    try:
        data = mock.sample(
            tables=tables,
            sample_size=2,  # 2 products
            model="openai/gpt-4o-mini"
        )
        
        print("Generated products:")
        print(data["products"][["product_id", "name", "category", "price"]])
        print("\nProduct image paths:")
        for _, row in data["products"].iterrows():
            print(f"  {row['name']}: {row['product_image']}")
        
        print("\nGenerated reviews:")
        print(data["reviews"][["review_id", "product_id", "reviewer_name", "rating"]])
        print("\nReviewer photo paths:")
        for _, row in data["reviews"].iterrows():
            print(f"  {row['reviewer_name']}: {row['reviewer_photo']}")
            
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have set your OPENAI_API_KEY environment variable")


def example_image_generation_options():
    """Example: Demonstrate different image generation options."""
    print("\n=== Image Generation Options Example ===")
    
    tables = {
        "artwork": {
            "prompt": "Artwork in a modern art gallery",
            "columns": {
                "artwork_id": {
                    "dtype": "string"
                },
                "title": {
                    "prompt": "creative artwork title", 
                    "dtype": "string"
                },
                "style": {
                    "dtype": "category",
                    "values": ["abstract", "portrait", "landscape", "still_life"]
                },
                # Different image models and sizes
                "artwork_image": {
                    "prompt": "The actual artwork - a beautiful and creative piece",
                    "dtype": "image",
                    "image_model": "openai/dall-e-3",
                    "image_size": "1024x1024"
                },
                "thumbnail": {
                    "prompt": "A smaller thumbnail version of the artwork for gallery catalog",
                    "dtype": "image", 
                    "image_model": "openai/dall-e-3",
                    "image_size": "256x256"
                }
            },
            "primary_key": "artwork_id",
        }
    }
    
    try:
        df = mock.sample(
            tables=tables,
            sample_size=2,
            model="openai/gpt-4o-mini"
        )
        
        print("Generated artwork data:")
        print(df[["artwork_id", "title", "style"]])
        
        print("\nImage files generated:")
        for _, row in df.iterrows():
            print(f"Artwork '{row['title']}':")
            print(f"  Full size: {row['artwork_image']}")
            print(f"  Thumbnail: {row['thumbnail']}")
            
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have set your OPENAI_API_KEY environment variable")


if __name__ == "__main__":
    print("🎨 MostlyAI Mock - Image Generation Examples")
    print("=" * 50)
    
    # Check if API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY environment variable is not set.")
        print("   You'll need to set this to run the examples successfully.")
        print("   Example: export OPENAI_API_KEY='your-api-key-here'")
        print()
    
    # Run examples
    example_single_table_with_images()
    example_multi_table_with_images() 
    example_image_generation_options()
    
    print("\n✅ Examples completed!")
    print("\nNote: Generated images are saved in the 'generated_images/' directory")
    print("with organized folder structure and descriptive filenames.")