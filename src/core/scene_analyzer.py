# src/core/scene_analyzer.py

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import ollama
from PIL import Image, ImageDraw

from src.utils.data_types import (
    BoundingBox,
    DetectedObject,
    ObjectType,
    SceneContext,
    WeatherCondition,
)


class SceneAnalysisError(Exception):
    """Base exception for scene analysis errors."""

    pass


class SceneAnalyzer:
    """Analyzes driving scenes for objects and context."""

    def __init__(self, model_name: str = "llama3.2-vision", debug: bool = False):
        self.model_name = model_name
        self.debug = debug
        self._prompt_template = self._get_scene_prompt()

    def _get_scene_prompt(self) -> str:
        """Get the prompt for scene analysis."""
        return """Analyze this driving scene image and respond with ONLY a JSON object in exactly this format.

Focus on key elements:
1. Individual Vehicles:
   - Identify EACH vehicle separately with its own bounding box
   - Do not group nearby vehicles together
   - Include cars, trucks, buses separately

2. Traffic Controls:
   - Each traffic light should have its own separate bounding box
   - Each traffic sign should be detected individually
   - Do not combine multiple traffic lights or signs

3. Road Environment:
   - Identify the road conditions
   - Note weather conditions
   - Describe time of day

Required JSON Format:
{
    "objects": [
        {
            "type": "vehicle|traffic_light|traffic_sign",
            "bbox": [x1, y1, x2, y2],  // EXACT normalized coordinates (0-1), must be precise
            "confidence": 0.9           // detection confidence
        }
    ],
    "context": {
        "weather": "clear|cloudy|rainy|foggy|snowy",
        "time": "day|night|dawn|dusk",
        "road": "brief road description"
    }
}

Important Rules:
1. Each object MUST have its own separate bounding box
2. Be extremely precise with bounding box coordinates
3. Only include actually visible objects
4. Use normalized coordinates (0-1) for bbox
5. Coordinates must be exact - no rounding
6. Each detection should have its own confidence score"""

    def _parse_response(self, response: Dict) -> Dict:
        """Parse the model response."""
        try:
            content = response["message"]["content"].strip()

            if self.debug:
                print("\nDebug: Raw model response:")
                print("-" * 50)
                print(content)
                print("-" * 50)

            # Parse JSON content
            data = json.loads(content)

            if "objects" not in data:
                data["objects"] = []
            if "context" not in data:
                raise SceneAnalysisError("Missing required 'context' field")

            if self.debug:
                print("\nDebug: Parsed JSON data:")
                print(json.dumps(data, indent=2))

            return data

        except Exception as e:
            raise SceneAnalysisError(f"Failed to parse response: {str(e)}") from e

    def _validate_bbox(self, bbox: List[float]) -> bool:
        """Validate bounding box coordinates."""
        if len(bbox) != 4:
            return False

        # Ensure coordinates are normalized and properly ordered
        if not all(0 <= x <= 1 for x in bbox):
            return False
        if not (bbox[0] < bbox[2] and bbox[1] < bbox[3]):
            return False

        # Validate minimum and maximum size
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        if width < 0.01 or height < 0.01:  # Minimum size check
            return False
        if width > 0.9 or height > 0.9:  # Maximum size check (avoid full image boxes)
            return False

        return True

    def _parse_objects(self, objects_data: List[Dict]) -> List[DetectedObject]:
        """Parse detected objects."""
        detected_objects = []

        for obj in objects_data:
            try:
                if self.debug:
                    print(f"\nDebug: Processing object: {obj}")

                # Validate coordinates
                if not self._validate_bbox(obj["bbox"]):
                    if self.debug:
                        print(f"Invalid bbox: {obj['bbox']}")
                    continue

                # Create object
                detected_obj = DetectedObject(
                    type=ObjectType(obj["type"]),
                    bbox=BoundingBox(
                        x_min=int(obj["bbox"][0] * 1000),
                        y_min=int(obj["bbox"][1] * 1000),
                        x_max=int(obj["bbox"][2] * 1000),
                        y_max=int(obj["bbox"][3] * 1000),
                    ),
                    confidence=float(obj["confidence"]),
                )

                detected_objects.append(detected_obj)

            except (KeyError, ValueError) as e:
                if self.debug:
                    print(f"Failed to parse object: {e}")
                continue

        return detected_objects

    def _parse_context(self, context_data: Dict) -> SceneContext:
        """Parse scene context."""
        return SceneContext(
            weather=WeatherCondition(context_data["weather"].lower()),
            time_of_day=context_data["time"].lower(),
            road_type=context_data["road"].strip(),
        )

    async def analyze(
        self, image_path: Union[str, Path], image_size: Optional[Tuple[int, int]] = None
    ) -> Tuple[List[DetectedObject], SceneContext]:
        """Analyze a driving scene image."""
        try:
            # Get model prediction with low temperature for more focused results
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": self._prompt_template,
                        "images": [str(image_path)],
                    }
                ],
                format="json",
                options={
                    "temperature": 0.5,  # Low temperature for more consistent results
                    "seed": 42,  # Fixed seed for reproducibility
                    "top_p": 0.5,  # More focused sampling
                    "num_predict": 1024,  # Ensure enough tokens for detailed response
                },
            )

            # Parse response
            data = self._parse_response(response)
            objects = self._parse_objects(data["objects"])
            context = self._parse_context(data["context"])

            # Scale bounding boxes if image size provided
            if image_size:
                width, height = image_size
                for obj in objects:
                    bbox = obj.bbox
                    obj.bbox = BoundingBox(
                        x_min=int(bbox.x_min * width / 1000),
                        y_min=int(bbox.y_min * height / 1000),
                        x_max=int(bbox.x_max * width / 1000),
                        y_max=int(bbox.y_max * height / 1000),
                    )

            return objects, context

        except Exception as e:
            if self.debug:
                import traceback

                print("\nDebug: Error traceback:")
                print(traceback.format_exc())
            raise SceneAnalysisError(f"Analysis failed: {str(e)}") from e

    @staticmethod
    def visualize(
        image_path: Union[str, Path],
        objects: List[DetectedObject],
        output_path: Optional[Path] = None,
    ) -> None:
        """Visualize detected objects on the image."""
        try:
            # Load image
            image = Image.open(image_path)
            draw = ImageDraw.Draw(image)

            # Color scheme for different object types
            colors = {
                ObjectType.VEHICLE: "#FF0000",  # Red
                ObjectType.TRAFFIC_LIGHT: "#00FF00",  # Green
                ObjectType.TRAFFIC_SIGN: "#0000FF",  # Blue
                ObjectType.BUS: "#FFA500",  # Orange
                ObjectType.CAR: "#FF69B4",  # Pink
            }

            # Draw boxes and labels
            for obj in objects:
                color = colors.get(obj.type, "#FFFFFF")

                # Draw bounding box
                draw.rectangle(obj.bbox.as_tuple, outline=color, width=2)

                # Draw label with confidence
                label = f"{obj.type.value}: {obj.confidence:.2f}"
                label_pos = (obj.bbox.x_min, max(0, obj.bbox.y_min - 20))

                # Draw label background
                text_bbox = draw.textbbox(label_pos, label)
                draw.rectangle(text_bbox, fill=color)

                # Draw label text
                draw.text(label_pos, label, fill="#FFFFFF")

            # Save or show
            if output_path:
                image.save(output_path)
            else:
                image.show()

        except Exception as e:
            print(f"Visualization failed: {e}")
