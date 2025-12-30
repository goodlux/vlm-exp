#!/usr/bin/env python3
"""Test VLM for aesthetic scoring and quality assessment."""

import base64
import json
from pathlib import Path

import requests


def encode_image(image_path: Path) -> str:
    """Encode image to base64 for Ollama."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def analyze_image(image_path: Path, model: str = "llava:7b", prompt: str = "") -> dict:
    """Send image to VLM via Ollama API."""
    url = "http://localhost:11434/api/generate"

    payload = {
        "model": model,
        "prompt": prompt,
        "images": [encode_image(image_path)],
        "stream": False
    }

    response = requests.post(url, json=payload)
    return response.json()


def test_aesthetic_scoring(image_path: Path, model: str = "llava:7b"):
    """Test aesthetic quality scoring."""
    prompts = {
        "aesthetic_score": """Rate this image's aesthetic quality on a scale of 1-10, where:
1-3: Poor composition, lighting, or technical quality
4-6: Average/acceptable quality
7-8: Good quality, well-composed
9-10: Exceptional, professional quality

Provide ONLY a number followed by a brief reason (one sentence).""",

        "blur_detection": """Is this image blurry or out of focus? Answer with:
SHARP - image is in focus
SLIGHT_BLUR - minor blur, still usable
BLURRY - significantly out of focus
Followed by a brief explanation.""",

        "technical_quality": """Assess the technical quality of this image:
- Exposure (underexposed/correct/overexposed)
- Focus (sharp/soft/blurry)
- Noise (clean/moderate/noisy)
- Composition (poor/average/good/excellent)

Be concise.""",

        "subject_tags": """List 5-10 descriptive tags for this image that would be useful for searching or organizing photos.
Format as: tag1, tag2, tag3, etc.
Focus on: subjects, mood, style, setting, colors."""
    }

    print(f"\n{'='*80}")
    print(f"🖼️  {image_path.name}")
    print(f"📐 Model: {model}")
    print(f"{'='*80}\n")

    for test_name, prompt in prompts.items():
        print(f"🔍 {test_name.upper().replace('_', ' ')}")
        print("-" * 80)

        result = analyze_image(image_path, model, prompt)
        response = result.get("response", "No response")

        print(response)
        print()


def main():
    images_dir = Path("images")

    if not images_dir.exists():
        print("No images/ directory found!")
        return

    # Get all image files
    image_files = list(images_dir.glob("*.jpg")) + \
                  list(images_dir.glob("*.jpeg")) + \
                  list(images_dir.glob("*.png")) + \
                  list(images_dir.glob("*.webp"))

    if not image_files:
        print("No images found in images/")
        return

    print(f"Found {len(image_files)} images.")
    print("Testing aesthetic scoring on all images...\n")

    # Test all images with llava:7b
    for img_path in image_files:
        test_aesthetic_scoring(img_path, "llava:7b")

    # Uncomment to test with minicpm-v once pulled
    # for img_path in image_files:
    #     test_aesthetic_scoring(img_path, "minicpm-v")


if __name__ == "__main__":
    main()
