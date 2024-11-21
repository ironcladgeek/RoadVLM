# src/utils/prompts.py

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


def get_prompts() -> Dict[str, str]:
    """Get the available model prompts."""
    return {
        "scene_analysis": (
            "Analyze this driving scene and respond using EXACTLY the following format with EXACTLY "
            "these allowed values. Do not use any other values.\n\n"
            f"Allowed ACTION values: {get_action_values()}\n"
            f"Allowed WEATHER values: {get_weather_values()}\n"
            f"Allowed TIME values: {get_time_values()}\n\n"
            "Required format (in JSON):\n"
            "{\n"
            "  \"Action\": \"[EXACT ACTION VALUE]\",\n"
            "  \"Confidence\": [NUMBER 0-1],\n"
            "  \"Weather\": \"[EXACT WEATHER VALUE]\",\n"
            "  \"Time\": \"[EXACT TIME VALUE]\",\n"
            "  \"Road\": \"[BRIEF DESCRIPTION]\"\n"
            "}"
        )
    }