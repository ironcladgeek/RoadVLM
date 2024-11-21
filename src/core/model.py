import re
from pathlib import Path
from typing import Dict, Optional, Union

import ollama
from pydantic import ValidationError

from src.utils.data_types import (
    ActionType,
    Direction,
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
        temperature: float = 0.1,
        timeout: float = 30.0,
    ):
        """Initialize the model.

        Args:
            model_name: Name of the Ollama model to use.
            temperature: Sampling temperature for model outputs (0.0 to 1.0).
            timeout: Maximum time (seconds) to wait for model response.
        """
        self.model_name = model_name
        self.temperature = temperature
        self.timeout = timeout

        self._prompts = get_prompts()

    def _create_message(self, image_path: Union[str, Path], prompt: str) -> Dict:
        """Create a message for the model with image and prompt."""
        return {
            "role": "user",
            "content": prompt,
            "images": [str(image_path)],
        }

    def _parse_action_response(self, response: Dict) -> Prediction:
        """Parse action prediction from model response."""
        content = response["message"]["content"]

        # Extract action and confidence using regex
        action_match = re.search(r"Action:\s*(\w+)", content)
        confidence_match = re.search(r"Confidence:\s*(0?\.\d+|1\.0|1)", content)

        if not action_match or not confidence_match:
            raise ResponseParsingError(f"Failed to parse action response: {content}")

        action = action_match.group(1)
        confidence = float(confidence_match.group(1))

        return Prediction(
            action=ActionType(action),
            confidence=confidence,
        )

    def _parse_scene_context(self, response: Dict) -> SceneContext:
        """Parse scene context from model response."""
        content = response["message"]["content"]

        # Extract weather, time, and road type using regex
        weather_match = re.search(r"Weather:\s*(\w+)", content)
        time_match = re.search(r"Time:\s*(\w+)", content)
        road_match = re.search(r"Road:\s*(.+?)(?=\s*$|\s*[A-Z])", content)

        if not all([weather_match, time_match, road_match]):
            raise ResponseParsingError(f"Failed to parse scene context: {content}")

        return SceneContext(
            weather=WeatherCondition(weather_match.group(1).lower()),
            time_of_day=time_match.group(1).lower(),
            road_type=road_match.group(1).strip(),
        )

    def _parse_direction(self, response: Dict) -> Direction:
        """Parse direction prediction from model response."""
        content = response["message"]["content"]

        # Extract angle, action type, and confidence using regex
        angle_match = re.search(r"Angle:\s*(\d+)", content)
        action_match = re.search(r"Action:\s*(\w+)", content)
        confidence_match = re.search(r"Confidence:\s*(0?\.\d+|1\.0|1)", content)

        if not all([angle_match, action_match, confidence_match]):
            raise ResponseParsingError(f"Failed to parse direction: {content}")

        return Direction(
            angle=float(angle_match.group(1)),
            type=ActionType(action_match.group(1)),
            confidence=float(confidence_match.group(1)),
        )

    async def predict(
        self, image_path: Union[str, Path], image_id: Optional[str] = None
    ) -> RoadVLMOutput:
        """Generate predictions for a driving scene image."""
        try:
            # Get driving action prediction
            action_response = ollama.chat(
                model=self.model_name,
                messages=[
                    self._create_message(image_path, self._prompts["driving_action"])
                ],
            )
            prediction = self._parse_action_response(action_response)

            # Get scene context
            context_response = ollama.chat(
                model=self.model_name,
                messages=[
                    self._create_message(image_path, self._prompts["scene_context"])
                ],
            )
            scene_context = self._parse_scene_context(context_response)

            # Get direction prediction
            direction_response = ollama.chat(
                model=self.model_name,
                messages=[self._create_message(image_path, self._prompts["direction"])],
            )
            direction = self._parse_direction(direction_response)

            # Combine all predictions
            output = RoadVLMOutput(
                prediction=prediction,
                direction=direction,
                scene_context=scene_context,
                image_id=image_id,
                objects=[],  # Empty for now, will be handled by scene analyzer
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
