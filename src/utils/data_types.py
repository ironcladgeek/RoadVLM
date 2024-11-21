from enum import Enum
from typing import List, Optional, Tuple

from pydantic import BaseModel, Field


class ActionType(str, Enum):
    """Available driving actions that can be predicted."""

    STOP = "STOP"
    CONTINUE = "CONTINUE"
    TURN_LEFT = "TURN_LEFT"
    TURN_RIGHT = "TURN_RIGHT"
    SLOW_DOWN = "SLOW_DOWN"


class ObjectType(str, Enum):
    """Types of objects that can be detected in the scene."""

    VEHICLE = "vehicle"
    PEDESTRIAN = "pedestrian"
    TRAFFIC_LIGHT = "traffic_light"
    TRAFFIC_SIGN = "traffic_sign"


class TrafficLightState(str, Enum):
    """Possible states for traffic lights."""

    RED = "red"
    YELLOW = "yellow"
    GREEN = "green"


class WeatherCondition(str, Enum):
    """Weather conditions that can be detected in the scene."""

    CLEAR = "clear"
    RAINY = "rainy"
    SNOWY = "snowy"
    FOGGY = "foggy"
    CLOUDY = "cloudy"


class BoundingBox(BaseModel):
    """Represents a bounding box for detected objects."""

    x_min: int = Field(..., description="Left coordinate")
    y_min: int = Field(..., description="Top coordinate")
    x_max: int = Field(..., description="Right coordinate")
    y_max: int = Field(..., description="Bottom coordinate")

    @property
    def as_tuple(self) -> Tuple[int, int, int, int]:
        """Return bounding box as tuple (x_min, y_min, x_max, y_max)."""
        return (self.x_min, self.y_min, self.x_max, self.y_max)


class DetectedObject(BaseModel):
    """Represents a detected object in the scene."""

    type: ObjectType
    bbox: BoundingBox
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Detection confidence score"
    )
    state: Optional[TrafficLightState] = Field(
        None, description="State for traffic lights"
    )


class Prediction(BaseModel):
    """Main prediction output for a driving action."""

    action: ActionType
    confidence: float = Field(..., ge=0.0, le=1.0)


class SceneContext(BaseModel):
    """Represents the context of the driving scene."""

    weather: WeatherCondition
    time_of_day: str = Field(..., pattern="^(day|night|dawn|dusk)$")
    road_type: str = Field(
        ..., description="Type of road (intersection, highway, etc.)"
    )


class RoadVLMOutput(BaseModel):
    """Complete output of the RoadVLM system."""

    prediction: Prediction
    objects: List[DetectedObject] = Field(default_factory=list)
    scene_context: SceneContext
    image_id: Optional[str] = Field(
        None, description="Identifier for the processed image"
    )
    processing_time: Optional[float] = Field(
        None, description="Processing time in seconds"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "prediction": {
                    "action": "CONTINUE",
                    "confidence": 0.89,
                },
                "objects": [
                    {
                        "type": "vehicle",
                        "bbox": {
                            "x_min": 100,
                            "y_min": 200,
                            "x_max": 300,
                            "y_max": 400,
                        },
                        "confidence": 0.95,
                    }
                ],
                "scene_context": {
                    "weather": "clear",
                    "time_of_day": "day",
                    "road_type": "intersection",
                },
            }
        }
