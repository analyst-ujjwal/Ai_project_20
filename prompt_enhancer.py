"""
Small prompt enhancer that expands user text into a richer prompt geared for logos.
This is intentionally simple and rule-based, but easily replaceable by an LLM call.
"""
import json
from typing import Optional

# Load style templates
try:
    with open("style_config.json", "r") as f:
        STYLE_CONFIG = json.load(f)
except Exception:
    STYLE_CONFIG = {
        "minimal": {
            "suffix": ", minimal flat vector logo, clean lines, solid shapes, transparent background"
        },
        "geometric": {
            "suffix": ", bold geometric emblem, symmetrical, negative space, vector style"
        },
        "gradient": {
            "suffix": ", modern logo with subtle gradient, soft highlights, polished"
        },
        "vintage": {
            "suffix": ", vintage badge logo, textured, muted color palette, hand-crafted"
        }
    }


def enhance_prompt(user_prompt: str, style: Optional[str] = "minimal") -> str:
    """
    Combine user prompt with style suffix and some generic logo-focused keywords.
    This makes prompts more descriptive for diffusion-based logo generation.
    """
    base = user_prompt.strip()
    suffix = STYLE_CONFIG.get(style, {}).get(
        "suffix", ", minimal flat vector logo, clean lines"
    )

    # Add diffusion-friendly tokens for better logo clarity and composition
    clarity_tokens = ", vector, centered composition, transparent background, high contrast, clean design"

    enhanced = f"{base}{suffix}{clarity_tokens}"
    return enhanced


# Example test (only runs if executed directly)
if __name__ == "__main__":
    print(enhance_prompt("modern coffee brand logo", "vintage"))
