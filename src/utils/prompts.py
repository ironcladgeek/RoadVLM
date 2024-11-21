from typing import Dict

from src.utils.data_types import ActionType, WeatherCondition


def get_action_values() -> str:
    """Get formatted string of possible action values."""
    return ", ".join([action.value for action in ActionType])


def get_weather_values() -> str:
    """Get formatted string of possible weather values."""
    return ", ".join([weather.value for weather in WeatherCondition])


def get_time_values() -> str:
    """Get formatted string of possible time of day values."""
    return "day, night, dawn, dusk"


PROMPTS: Dict[str, str] = {
    "driving_action": (
        "You are a driving assistant. Based on this image, what should be the next "
        "driving action? Choose EXACTLY ONE action from the following options: {actions}. "
        "Also rate your confidence from 0 to 1. "
        "Response format - Action: [ACTION], Confidence: [0-1]"
    ),
    "scene_context": (
        "Analyze the scene conditions. "
        "Weather must be one of: {weather}. "
        "Time of day must be one of: {time_of_day}. "
        "Also describe the road type (intersection, highway, etc.). "
        "Response format - Weather: [WEATHER], Time: [TIME], Road: [DESCRIPTION]"
    ),
    "direction": (
        "Based on the driving scene, what direction should the vehicle move? "
        "Provide the angle in degrees (0-360), the action type ({actions}), "
        "and your confidence (0-1). "
        "Response format - Angle: [0-360], Action: [ACTION], Confidence: [0-1]"
    ),
}

def get_prompts() -> Dict[str, str]:
    """Get the available model prompts."""
    return {
            key: prompt.format(
                actions=get_action_values(),
                weather=get_weather_values(),
                time_of_day=get_time_values(),
            )
            for key, prompt in PROMPTS.items()
        }