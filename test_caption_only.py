#!/usr/bin/env python3
"""Simple caption-only test for speed comparison."""

import base64
import subprocess
import tempfile
import time
from pathlib import Path

import requests


def extract_preview(dng_path: Path, output_path: Path = None) -> Path:
    """Extract PreviewImage from DNG file using exiftool.

    Returns path to extracted JPEG.
    """
    if output_path is None:
        # Create temp file
        output_path = Path(tempfile.mktemp(suffix=".jpg"))

    # Use exiftool command directly
    cmd = ["exiftool", "-b", "-PreviewImage", str(dng_path)]

    result = subprocess.run(cmd, capture_output=True, check=True)

    if result.stdout:
        with open(output_path, "wb") as f:
            f.write(result.stdout)
        return output_path
    else:
        raise ValueError(f"No PreviewImage found in {dng_path}")


def encode_image(image_path: Path) -> str:
    """Encode image to base64 for Ollama."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def analyze_image(image_path: Path, model: str, prompt: str) -> dict:
    """Send image to VLM via Ollama API."""
    url = "http://localhost:11434/api/generate"

    payload = {
        "model": model,
        "prompt": prompt,
        "images": [encode_image(image_path)],
        "stream": False
    }

    response = requests.post(url, json=payload, timeout=120)
    return response.json()


def caption_image(dng_path: Path, model: str = "llava:34b"):
    """Generate caption for DNG image."""
    overall_start = time.time()

    print(f"\n{'='*80}")
    print(f"🖼️  {dng_path.name}")
    print(f"📐 Model: {model}")
    print(f"{'='*80}\n")

    # Extract preview
    extract_start = time.time()
    preview_path = extract_preview(dng_path)
    extract_time = time.time() - extract_start

    # Simple caption prompt
    prompt = "Describe this image in detail, focusing on the subject, their pose or action, clothing, and setting."

    analysis_start = time.time()
    result = analyze_image(preview_path, model, prompt)
    analysis_time = time.time() - analysis_start

    response = result.get("response", "No response")
    print(response)
    print()

    overall_time = time.time() - overall_start
    print(f"⏱️  Extraction: {extract_time:.2f}s | Analysis: {analysis_time:.2f}s | Total: {overall_time:.2f}s")

    # Cleanup temp file
    if preview_path.exists():
        preview_path.unlink()


def main():
    dng_dir = Path("images/dance-dng")

    if not dng_dir.exists():
        print(f"Directory not found: {dng_dir}")
        return

    dng_files = list(dng_dir.glob("*.dng")) + list(dng_dir.glob("*.DNG"))

    if not dng_files:
        print("No DNG files found")
        return

    # Sort alphabetically
    dng_files.sort()

    print(f"Found {len(dng_files)} DNG files (sorted alphabetically)")

    # Caption all DNG files with qwen3-vl:32b
    for dng_path in dng_files:
        try:
            caption_image(dng_path, "qwen3-vl:32b")
        except Exception as e:
            print(f"❌ Error processing {dng_path.name}: {e}\n")


if __name__ == "__main__":
    main()
