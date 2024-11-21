# src/core/model.py
import re
from pathlib import Path
from typing import Dict, Optional, Union

import ollama
from pydantic import ValidationError

from src.utils.data_types import (
    ActionType,
    Prediction,
    RoadVLMOutput,
    SceneContext,
    WeatherCondition,
)
from src.utils.prompts import (
    get_action_values,
    get_prompts,
    get_time_values,
    get_weather_values,
)


class ModelError(Exception):
    """Base exception for model-related errors."""

    pass


class ResponseParsingError(ModelError):
    """Raised when response parsing fails."""

    pass


class Model:
    """Handles interaction with the Ollama vision-language model."""

    def __init__(
        self,
        model_name: str = "llama3.2-vision",
    ):
        """Initialize the model."""
        self.model_name = model_name
        self._prompts = get_prompts()

    def _create_message(self, image_path: Union[str, Path], prompt: str) -> Dict:
        """Create a message for the model with image and prompt."""
        return {
            "role": "user",
            "content": prompt,
            "images": [str(image_path)],
        }

    def _parse_response(self, response: Dict) -> tuple[Prediction, SceneContext]:
        """Parse the complete model response."""
        content = response["message"]["content"].strip()
        print("\nModel response:")
        print(content)

        try:
            # Split content into lines and remove empty lines
            lines = [line.strip() for line in content.split("\n") if line.strip()]

            if len(lines) != 4:
                raise ResponseParsingError(
                    f"Expected 4 lines of output, got {len(lines)}. Response:\n{content}"
                )

            # Parse action line
            action_match = re.match(
                r"^Action:\s*(\w+),\s*Confidence:\s*(0?\.\d+|1\.0|1)$", lines[0]
            )
            if not action_match:
                raise ResponseParsingError(
                    f"Invalid action format. Got: '{lines[0]}'\n"
                    f"Allowed actions are: {get_action_values()}"
                )

            # Parse weather line
            weather_match = re.match(r"^Weather:\s*(\w+)$", lines[1])
            if not weather_match:
                raise ResponseParsingError(
                    f"Invalid weather format. Got: '{lines[1]}'\n"
                    f"Allowed weather values are: {get_weather_values()}"
                )

            # Parse time line
            time_match = re.match(r"^Time:\s*(\w+)$", lines[2])
            if not time_match:
                raise ResponseParsingError(
                    f"Invalid time format. Got: '{lines[2]}'\n"
                    f"Allowed time values are: {get_time_values()}"
                )

            # Parse road line
            road_match = re.match(r"^Road:\s*(.+?)$", lines[3])
            if not road_match:
                raise ResponseParsingError(f"Invalid road format. Got: '{lines[3]}'")

            # Validate enum values
            try:
                action = ActionType(action_match.group(1))
                weather = WeatherCondition(weather_match.group(1).lower())
                time = time_match.group(1).lower()
                if time not in ["day", "night", "dawn", "dusk"]:
                    raise ValueError(f"Invalid time value: {time}")
            except ValueError as e:
                raise ResponseParsingError(f"Invalid enum value: {str(e)}") from e

            # Create objects
            prediction = Prediction(
                action=action,
                confidence=float(action_match.group(2)),
            )

            scene_context = SceneContext(
                weather=weather,
                time_of_day=time,
                road_type=road_match.group(1).strip(),
            )

            return prediction, scene_context

        except (AttributeError, IndexError) as e:
            raise ResponseParsingError(
                f"Failed to parse response components: {str(e)}\nResponse:\n{content}"
            ) from e

    async def predict(
        self, image_path: Union[str, Path], image_id: Optional[str] = None
    ) -> RoadVLMOutput:
        """Generate predictions for a driving scene image."""
        try:
            # Get complete scene analysis in one call
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    self._create_message(image_path, self._prompts["scene_analysis"])
                ],
            )

            prediction, scene_context = self._parse_response(response)

            # Combine all predictions
            output = RoadVLMOutput(
                prediction=prediction,
                scene_context=scene_context,
                image_id=image_id,
            )

            return output

        except ValidationError as e:
            raise ModelError(f"Failed to validate model output: {e}") from e
        except ResponseParsingError as e:
            raise ModelError(str(e)) from e
        except Exception as e:
            raise ModelError(f"Model prediction failed: {e}") from e

    def __call__(
        self, image_path: Union[str, Path], image_id: Optional[str] = None
    ) -> RoadVLMOutput:
        """Convenience method to call predict."""
        return self.predict(image_path, image_id)
