#!/usr/bin/env python3
"""Quick test script to run llava:7b on local images."""

import base64
import json
from pathlib import Path

import requests


def encode_image(image_path: Path) -> str:
    """Encode image to base64 for Ollama."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def analyze_image(image_path: Path, prompt: str = "Describe this image in detail.") -> dict:
    """Send image to llava via Ollama API."""
    url = "http://localhost:11434/api/generate"

    payload = {
        "model": "llava:7b",
        "prompt": prompt,
        "images": [encode_image(image_path)],
        "stream": False
    }

    response = requests.post(url, json=payload)
    return response.json()


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

    print(f"Found {len(image_files)} images. Running llava:7b...\n")

    for img_path in image_files:
        print(f"🖼️  {img_path.name}")
        print("-" * 80)

        result = analyze_image(img_path)
        print(result.get("response", "No response"))
        print("\n")


if __name__ == "__main__":
    main()
