#!/usr/bin/env python3
"""Test moondream3 via MLX (native) vs moondream:1.8b via Ollama."""

import subprocess
import tempfile
import time
from pathlib import Path

import moondream as md
from PIL import Image


def extract_preview(dng_path: Path, output_path: Path = None) -> Path:
    """Extract PreviewImage from DNG file using exiftool."""
    if output_path is None:
        output_path = Path(tempfile.mktemp(suffix=".jpg"))

    cmd = ["exiftool", "-b", "-PreviewImage", str(dng_path)]
    result = subprocess.run(cmd, capture_output=True, check=True)

    if result.stdout:
        with open(output_path, "wb") as f:
            f.write(result.stdout)
        return output_path
    else:
        raise ValueError(f"No PreviewImage found in {dng_path}")


def test_moondream3_mlx(image_path: Path):
    """Test moondream3 via MLX (assumes moondream-station server is running)."""
    print("🌙 Testing Moondream3 (MLX native via station server)")
    print("-" * 80)

    # Connect to moondream-station server
    print("Connecting to moondream station at http://localhost:2020/v1...")
    try:
        model = md.vl(endpoint="http://localhost:2020/v1")
    except Exception as e:
        print(f"❌ Failed to connect to moondream-station server: {e}")
        print("💡 Start the server first: moondream-station")
        return None, 0

    # Load image
    image = Image.open(image_path)

    # Generate caption
    print(f"📸 Captioning: {image_path.name}")
    start_caption = time.time()

    try:
        result = model.query(image, "Describe this image in detail, focusing on the subject, their pose or action, clothing, and setting.")

        # Debug: see what the result actually contains
        print(f"DEBUG - Result type: {type(result)}")
        if isinstance(result, dict):
            print(f"DEBUG - Result keys: {list(result.keys())}")
            print(f"DEBUG - Full result: {result}")
        else:
            print(f"DEBUG - Result: {result}")

        # Try different possible response formats
        if isinstance(result, dict):
            caption = result.get("answer") or result.get("response") or result.get("text") or str(result)
        else:
            caption = str(result)

    except Exception as e:
        print(f"❌ Query failed: {e}")
        print(f"Exception type: {type(e)}")
        import traceback
        traceback.print_exc()
        return None, 0

    caption_time = time.time() - start_caption

    print(f"Caption: {caption}")
    print(f"⏱️  Caption time: {caption_time:.2f}s\n")

    return caption, caption_time


def main():
    # Test on a few DNGs
    dng_dir = Path("images/dance-dng")

    if not dng_dir.exists():
        # Try current directory
        dng_dir = Path(".")

    dng_files = list(dng_dir.glob("*.dng"))[:3]  # Just first 3

    if not dng_files:
        print(f"No DNG files found in {dng_dir}")
        return

    print(f"Testing moondream3 MLX on {len(dng_files)} images\n")

    total_caption_time = 0

    for dng_path in dng_files:
        print(f"\n{'='*80}")
        print(f"🖼️  {dng_path.name}")
        print(f"{'='*80}\n")

        try:
            # Extract preview
            extract_start = time.time()
            preview_path = extract_preview(dng_path)
            extract_time = time.time() - extract_start
            print(f"📤 Extracted preview in {extract_time:.2f}s\n")

            try:
                # Test moondream3 MLX
                caption, caption_time = test_moondream3_mlx(preview_path)
                if caption:
                    total_caption_time += caption_time

            finally:
                # Cleanup
                if preview_path.exists():
                    preview_path.unlink()

        except ValueError as e:
            print(f"⚠️  Skipping: {e}\n")
            continue

    avg_time = total_caption_time / len(dng_files)
    print(f"\n{'='*80}")
    print(f"📊 Average caption time: {avg_time:.2f}s per image")
    print(f"📊 Total caption time: {total_caption_time:.2f}s for {len(dng_files)} images")
    print(f"\n💡 Compare to moondream:1.8b via Ollama (~1.5s per image)")


if __name__ == "__main__":
    main()
