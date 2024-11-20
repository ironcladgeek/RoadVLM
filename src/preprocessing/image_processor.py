from pathlib import Path
from typing import Union

from PIL import Image, UnidentifiedImageError


class ImageProcessingError(Exception):
    """Base exception for image processing errors."""

    pass


class ImageProcessor:
    """Handles image validation and basic preprocessing for the RoadVLM model."""

    def __init__(self, min_width: int = 320, min_height: int = 240):
        """Initialize the image processor.

        Args:
            min_width: Minimum acceptable image width.
            min_height: Minimum acceptable image height.
        """
        self.min_width = min_width
        self.min_height = min_height
        self._supported_formats = {".jpg", ".jpeg", ".png"}

    def validate_image(self, image_path: Union[str, Path]) -> Path:
        """Validate image file existence, format, and dimensions.

        Args:
            image_path: Path to the image file.

        Returns:
            Path: Validated image path.

        Raises:
            ImageProcessingError: If validation fails.
        """
        image_path = Path(image_path)

        # Check if file exists
        if not image_path.exists():
            raise ImageProcessingError(f"Image file not found: {image_path}")

        # Check format
        if image_path.suffix.lower() not in self._supported_formats:
            raise ImageProcessingError(
                f"Unsupported image format: {image_path.suffix}. "
                f"Supported formats: {', '.join(self._supported_formats)}"
            )

        try:
            # Verify image can be opened and check dimensions
            with Image.open(image_path) as img:
                width, height = img.size
                if width < self.min_width or height < self.min_height:
                    raise ImageProcessingError(
                        f"Image dimensions ({width}x{height}) are smaller than "
                        f"minimum required ({self.min_width}x{self.min_height})"
                    )

                # Ensure it's in RGB format
                if img.mode not in ("RGB", "RGBA"):
                    raise ImageProcessingError(
                        f"Unsupported image mode: {img.mode}. Must be RGB or RGBA."
                    )

            return image_path

        except UnidentifiedImageError as e:
            raise ImageProcessingError(
                f"Invalid or corrupted image file: {image_path}"
            ) from e
        except Exception as e:
            raise ImageProcessingError(f"Failed to validate image: {e}") from e

    def __call__(self, image_path: Union[str, Path]) -> Path:
        """Process and validate an image file.

        Args:
            image_path: Path to the image file.

        Returns:
            Path: Validated image path.
        """
        return self.validate_image(image_path)
