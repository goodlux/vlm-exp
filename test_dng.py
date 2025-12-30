#!/usr/bin/env python3
"""Test if VLM can handle DNG files directly."""

import base64
from pathlib import Path

import requests


def encode_image(image_path: Path) -> str:
    """Encode image to base64 for Ollama."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def test_dng(image_path: Path, model: str = "llava:7b"):
    """Test DNG file with VLM."""
    url = "http://localhost:11434/api/generate"

    prompt = "Describe this image briefly."

    payload = {
        "model": model,
        "prompt": prompt,
        "images": [encode_image(image_path)],
        "stream": False
    }

    print(f"Testing: {image_path.name}")
    print(f"Size: {image_path.stat().st_size / 1024 / 1024:.2f} MB")
    print("-" * 80)

    try:
        response = requests.post(url, json=payload, timeout=60)
        result = response.json()

        if response.status_code == 200:
            print("✅ SUCCESS - VLM processed DNG directly!")
            print(result.get("response", "No response"))
        else:
            print(f"❌ FAILED - Status {response.status_code}")
            print(result)
    except Exception as e:
        print(f"❌ ERROR: {e}")

    print()


def main():
    dng_dir = Path("images/dance-dng")

    if not dng_dir.exists():
        print(f"Directory not found: {dng_dir}")
        return

    dng_files = list(dng_dir.glob("*.dng")) + list(dng_dir.glob("*.DNG"))

    if not dng_files:
        print("No DNG files found")
        return

    print(f"Found {len(dng_files)} DNG files\n")

    # Test just the first one
    print("Testing first DNG file to see if VLM can handle it...\n")
    test_dng(dng_files[0])


if __name__ == "__main__":
    main()
