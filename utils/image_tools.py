"""
Image utility functions for post-processing logos.
Includes cropping, padding, and sharpening to clean up AI-generated logos.
"""

from PIL import Image, ImageFilter

def post_process_logo(img: Image.Image) -> Image.Image:
    """
    Perform lightweight post-processing on a generated logo image.
    Steps:
    1. Convert to RGBA (handles transparency properly)
    2. Crop whitespace or transparent border
    3. Center on a square transparent canvas
    4. Apply mild sharpening for crispness
    """
    img = img.convert("RGBA")

    # Crop transparent or white borders
    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)

    # Create square canvas
    size = max(img.width, img.height)
    canvas = Image.new("RGBA", (size, size), (255, 255, 255, 0))
    x = (size - img.width) // 2
    y = (size - img.height) // 2
    canvas.paste(img, (x, y), img)

    # Slight sharpen for clarity
    canvas = canvas.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=3))

    return canvas
