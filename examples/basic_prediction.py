import asyncio
import time
from pathlib import Path

from src.core.model import Model, ModelError
from src.preprocessing.image_processor import ImageProcessor


async def main():
    # Initialize components
    image_processor = ImageProcessor(min_width=320, min_height=220)
    model = Model(model_name="llama3.2-vision")

    # Example image path - you can change this to your image path
    image_path = Path("examples/sample_data/image.jpg")

    try:
        # Step 1: Validate the image
        print(f"\nProcessing image: {image_path}")
        validated_image_path = image_processor(image_path)
        print("✓ Image validation successful")

        # Step 2: Get predictions
        print("\nGenerating predictions...")
        start_time = time.time()

        result = await model.predict(validated_image_path, image_id=image_path.stem)

        processing_time = time.time() - start_time

        # Step 3: Print results
        print("\nPrediction Results:")
        print("-" * 50)
        print(f"Driving Action: {result.prediction.action.value}")
        print(f"Confidence: {result.prediction.confidence:.2f}")
        print("\nScene Context:")
        print(f"Weather: {result.scene_context.weather.value}")
        print(f"Time of Day: {result.scene_context.time_of_day}")
        print(f"Road Type: {result.scene_context.road_type}")
        print("\nDirection:")
        print(f"Angle: {result.direction.angle}°")
        print(f"Action: {result.direction.type.value}")
        print(f"Confidence: {result.direction.confidence:.2f}")
        print("-" * 50)
        print(f"\nTotal processing time: {processing_time:.2f} seconds")

    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
    except ModelError as e:
        print(f"Error: Model prediction failed - {e}")
    except Exception as e:
        print(f"Error: An unexpected error occurred - {e}")


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
