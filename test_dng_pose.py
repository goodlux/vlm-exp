#!/usr/bin/env python3
"""Extract preview from DNG and test pose/dance detection with VLM."""

import base64
import json
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


def test_pose_detection(dng_path: Path, model: str = "llava:7b"):
    """Test pose/dance detection on DNG image with combined JSON output."""
    overall_start = time.time()

    print(f"\n{'='*80}")
    print(f"🖼️  {dng_path.name}")
    print(f"📐 Model: {model}")
    print(f"{'='*80}\n")

    # Extract preview
    extract_start = time.time()
    print("📤 Extracting preview from DNG...")
    preview_path = extract_preview(dng_path)
    extract_time = time.time() - extract_start
    preview_size = preview_path.stat().st_size / 1024
    print(f"✅ Extracted preview: {preview_size:.1f}KB ({extract_time:.2f}s)\n")

    # Combined JSON prompt with constrained options
    prompt = """Analyze this dance/movement image and provide a JSON response with the following fields:

{
  "caption": "A detailed description of what's happening in the image",
  "visible_body_parts": {
    "head": true/false,
    "arms": true/false,
    "hands": true/false,
    "torso": true/false,
    "legs": true/false,
    "feet": true/false
  },
  "direction_facing": "MUST be one of: facing_camera, profile_left, profile_right, back_to_camera, three_quarter_left, three_quarter_right",
  "pose_name": "name of the pose if identifiable (e.g., arabesque, plie, attitude, etc.), or 'unknown'",
  "body_position": "MUST be one of: standing, kneeling, sitting, lying, jumping, suspended, floor_work",
  "tags": ["tag1", "tag2", "tag3"]
}

For tags, use Danbooru/e621 style tags (e.g., "1girl", "solo", "dancing", "motion_blur", "dramatic_lighting", "stage", "performance").
Provide ONLY valid JSON, no additional text."""

    print("🔍 ANALYZING IMAGE")
    print("-" * 80)

    analysis_start = time.time()
    result = analyze_image(preview_path, model, prompt)
    analysis_time = time.time() - analysis_start

    response = result.get("response", "No response")
    print(response)
    print()

    # Try to parse as JSON
    try:
        parsed = json.loads(response)
        print("✅ Valid JSON response!")
        print(f"📊 Caption: {parsed.get('caption', 'N/A')[:80]}...")
        print(f"📊 Pose: {parsed.get('pose_name', 'N/A')}")
        print(f"📊 Tags: {', '.join(parsed.get('tags', [])[:5])}...")
    except json.JSONDecodeError:
        print("⚠️  Response is not valid JSON")

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

    print(f"Found {len(dng_files)} DNG files")

    # Test all DNG files with llava:34b
    for dng_path in dng_files:
        try:
            test_pose_detection(dng_path, "llava:34b")
        except Exception as e:
            print(f"❌ Error processing {dng_path.name}: {e}\n")


if __name__ == "__main__":
    main()
