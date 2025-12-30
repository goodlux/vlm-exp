#!/usr/bin/env python3
"""Caption test2 DNGs using PreviewImage and moondream:1.8b."""

import base64
import subprocess
import tempfile
import time
from pathlib import Path

import requests


def extract_preview(dng_path: Path) -> Path:
    """Extract PreviewImage from DNG file using exiftool."""
    output_path = Path(tempfile.mktemp(suffix=".jpg"))

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


def generate_caption(image_path: Path) -> str:
    """Generate caption using moondream:1.8b via Ollama."""
    url = "http://localhost:11434/api/generate"

    prompt = "Describe this image in detail, focusing on the subject, their pose or action, clothing, and setting."

    payload = {
        "model": "moondream:1.8b",
        "prompt": prompt,
        "images": [encode_image(image_path)],
        "stream": False
    }

    response = requests.post(url, json=payload, timeout=120)
    result = response.json()
    return result.get("response", "").strip()


def main():
    test_dir = Path("images/test2")

    if not test_dir.exists():
        print(f"❌ Directory not found: {test_dir}")
        return

    dng_files = sorted(test_dir.glob("*.dng"))

    if not dng_files:
        print("❌ No DNG files found")
        return

    print(f"📸 Found {len(dng_files)} DNG files")
    print(f"🤖 Model: moondream:1.8b\n")

    for i, dng_path in enumerate(dng_files, 1):
        print(f"[{i}/{len(dng_files)}] {dng_path.name}")

        try:
            # Extract preview
            start = time.time()
            preview_path = extract_preview(dng_path)
            extract_time = time.time() - start
            preview_size = preview_path.stat().st_size / 1024

            # Generate caption
            caption_start = time.time()
            caption = generate_caption(preview_path)
            caption_time = time.time() - caption_start

            # Cleanup
            preview_path.unlink()

            total_time = time.time() - start

            print(f"  ✅ {caption}")
            print(f"  ⏱️  Extract: {extract_time:.2f}s ({preview_size:.1f}KB) | Caption: {caption_time:.2f}s | Total: {total_time:.2f}s")

        except Exception as e:
            print(f"  ❌ Error: {e}")

        print()


if __name__ == "__main__":
    main()
