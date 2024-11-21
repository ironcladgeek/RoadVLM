# examples/basic_prediction.py

import asyncio
import time
from datetime import datetime
from pathlib import Path

from PIL import Image

from src.core.model import Model
from src.core.scene_analyzer import SceneAnalyzer
from src.preprocessing.image_processor import ImageProcessor


def format_results(
    image_path: Path,
    action_result,
    objects,
    scene_context,
    analysis_time: float,
    prediction_time: float,
) -> str:
    """Format analysis results as text."""
    lines = [
        f"Analysis Results for {image_path.name}",
        "=" * 50,
        "\nDriving Action:",
        "-" * 50,
        f"Action: {action_result.prediction.action.value}",
        f"Confidence: {action_result.prediction.confidence:.2f}",
        "\nDetected Objects:",
        "-" * 50,
    ]

    if not objects:
        lines.append("No objects detected")
    else:
        for obj in objects:
            lines.append(
                f"- Type: {obj.type.value:<12} Confidence: {obj.confidence:.2f}"
            )
            lines.append(
                f"  Location: ({obj.bbox.x_min}, {obj.bbox.y_min}, {obj.bbox.x_max}, {obj.bbox.y_max})"
            )

    lines.extend(
        [
            "-" * 50,
            "\nScene Context:",
            "-" * 50,
            f"Weather: {scene_context.weather.value}",
            f"Time of Day: {scene_context.time_of_day}",
            f"Road Type: {scene_context.road_type}",
            "\nPerformance Summary:",
            f"Scene Analysis: {analysis_time:.2f}s",
            f"Action Prediction: {prediction_time:.2f}s",
            f"Total Processing: {analysis_time + prediction_time:.2f}s",
            "\n",
        ]
    )

    return "\n".join(lines)


async def process_single_image(
    image_path: Path,
    output_dir: Path,
    image_processor: ImageProcessor,
    model: Model,
    scene_analyzer: SceneAnalyzer,
    timestamp: str,
) -> None:
    """Process a single image and save results."""
    try:
        # Create output filename base
        output_base = output_dir / f"{image_path.stem}_{timestamp}"
        output_image = output_base.with_suffix(".png")
        output_results = output_base.with_suffix(".txt")

        # Validate image
        print(f"\nProcessing image: {image_path}")
        validated_image_path = image_processor(image_path)
        print("✓ Image validation successful")

        # Get image size
        with Image.open(validated_image_path) as img:
            image_size = img.size

        # Run scene analysis
        print("\nAnalyzing scene...")
        analysis_start = time.time()

        objects, scene_context = await scene_analyzer.analyze(
            validated_image_path, image_size=image_size
        )

        # Generate visualization
        SceneAnalyzer.visualize(
            image_path=image_path, objects=objects, output_path=output_image
        )

        analysis_time = time.time() - analysis_start
        print(f"✓ Scene analysis completed in {analysis_time:.2f} seconds")

        # Get driving prediction
        print("\nGenerating driving predictions...")
        prediction_start = time.time()

        result = await model.predict(validated_image_path, image_id=image_path.stem)
        prediction_time = time.time() - prediction_start

        print(f"✓ Prediction completed in {prediction_time:.2f} seconds")

        # Format and save results
        results_text = format_results(
            image_path, result, objects, scene_context, analysis_time, prediction_time
        )

        output_results.write_text(results_text)

        print(f"✓ Results saved to: {output_results}")
        print(f"✓ Annotated image saved to: {output_image}")

    except Exception as e:
        print(f"Error processing {image_path.name}: {str(e)}")
        # Write error to a separate error log
        error_log = output_dir / f"errors_{timestamp}.log"
        with error_log.open("a") as f:
            f.write(f"{image_path.name}: {str(e)}\n")


async def main():
    # Get input and output paths
    input_dir = Path("examples/sample_data/input")
    output_dir = Path("examples/sample_data/output")

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Initialize components
    image_processor = ImageProcessor(min_width=320, min_height=220)
    model = Model(model_name="llama3.2-vision")
    scene_analyzer = SceneAnalyzer(model_name="llama3.2-vision", debug=True)

    try:
        # Get all image files
        image_files = [
            f
            for f in input_dir.iterdir()
            if f.suffix.lower() in [".png", ".jpg", ".jpeg"]
        ]

        if not image_files:
            print(f"No images found in {input_dir}")
            return

        print(f"Found {len(image_files)} images to process")

        # Process each image
        for image_path in image_files:
            await process_single_image(
                image_path,
                output_dir,
                image_processor,
                model,
                scene_analyzer,
                timestamp,
            )

        # Write summary
        summary_path = output_dir / f"summary_{timestamp}.txt"
        summary = [
            "Processing Summary",
            f"Timestamp: {timestamp}",
            f"Total images processed: {len(image_files)}",
            f"Input directory: {input_dir}",
            f"Output directory: {output_dir}",
        ]
        summary_path.write_text("\n".join(summary))

        print("\nProcessing complete!")
        print(f"Results saved to: {output_dir}")

    except Exception as e:
        print(f"Error: An unexpected error occurred - {e}")


if __name__ == "__main__":
    asyncio.run(main())
