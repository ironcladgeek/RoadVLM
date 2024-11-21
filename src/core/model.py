import json
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
from src.utils.prompts import get_prompts


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
            # Parse JSON content
            data = json.loads(content)

            # Validate enum values
            try:
                action = ActionType(data["Action"])
                weather = WeatherCondition(data["Weather"].lower())
                time = data["Time"].lower()
                if time not in ["day", "night", "dawn", "dusk"]:
                    raise ValueError(f"Invalid time value: {time}")
            except ValueError as e:
                raise ResponseParsingError(f"Invalid enum value: {str(e)}") from e

            # Create objects
            prediction = Prediction(
                action=action,
                confidence=float(data["Confidence"]),
            )

            scene_context = SceneContext(
                weather=weather,
                time_of_day=time,
                road_type=data["Road"].strip(),
            )

            return prediction, scene_context

        except (json.JSONDecodeError, KeyError, TypeError) as e:
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
                format="json",
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
