#!/usr/bin/env python3
"""Caption DNG files with Moondream and write to custom XMP-vlm namespace."""

import base64
import subprocess
import tempfile
import time
from pathlib import Path

import requests


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


def encode_image(image_path: Path) -> str:
    """Encode image to base64 for Ollama."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def generate_caption(image_path: Path, model: str = "moondream:1.8b") -> str:
    """Generate caption using VLM via Ollama API."""
    url = "http://localhost:11434/api/generate"

    prompt = "Describe this image in detail, focusing on the subject, their pose or action, clothing, and setting."

    payload = {
        "model": model,
        "prompt": prompt,
        "images": [encode_image(image_path)],
        "stream": False
    }

    response = requests.post(url, json=payload, timeout=120)
    result = response.json()
    return result.get("response", "")


def write_caption_to_xmp(dng_path: Path, caption: str) -> bool:
    """Write caption to custom XMP-vlm:MoondreamCaption field."""

    # Create exiftool config for custom namespace
    config_content = """
%Image::ExifTool::UserDefined = (
    'Image::ExifTool::XMP::Main' => {
        vlm => {
            SubDirectory => {
                TagTable => 'Image::ExifTool::UserDefined::vlm',
            },
        },
    },
);

%Image::ExifTool::UserDefined::vlm = (
    GROUPS => { 0 => 'XMP', 1 => 'XMP-vlm', 2 => 'Image' },
    NAMESPACE => { 'vlm' => 'http://goodlux.com/vlm/1.0/' },
    WRITABLE => 'string',
    MoondreamCaption => {},
);

1;
"""

    # Write config to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.config', delete=False) as config_file:
        config_file.write(config_content)
        config_path = config_file.name

    try:
        # Build exiftool command
        cmd = [
            "exiftool",
            "-config", config_path,
            f"-XMP-vlm:MoondreamCaption={caption}",
            "-overwrite_original",
            str(dng_path)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            return True
        else:
            print(f"❌ exiftool error: {result.stderr}")
            return False

    finally:
        # Cleanup config file
        Path(config_path).unlink(missing_ok=True)


def caption_dng(dng_path: Path, model: str = "moondream:1.8b"):
    """Extract preview, generate caption, write to XMP."""
    start_time = time.time()

    print(f"\n🖼️  {dng_path.name}")
    print("-" * 80)

    # Extract preview
    extract_start = time.time()
    preview_path = extract_preview(dng_path)
    extract_time = time.time() - extract_start

    # Generate caption
    caption_start = time.time()
    caption = generate_caption(preview_path, model)
    caption_time = time.time() - caption_start

    print(f"Caption: {caption}")

    # Write to XMP
    write_start = time.time()
    success = write_caption_to_xmp(dng_path, caption)
    write_time = time.time() - write_start

    # Cleanup temp preview
    if preview_path.exists():
        preview_path.unlink()

    total_time = time.time() - start_time

    if success:
        print(f"✅ Written to XMP-vlm:MoondreamCaption")
    else:
        print(f"❌ Failed to write XMP")

    print(f"⏱️  Extract: {extract_time:.2f}s | Caption: {caption_time:.2f}s | Write: {write_time:.2f}s | Total: {total_time:.2f}s")


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

    print(f"Found {len(dng_files)} DNG files")
    print(f"Captioning with moondream:1.8b and writing to XMP-vlm:MoondreamCaption\n")

    success_count = 0
    for dng_path in dng_files:
        try:
            caption_dng(dng_path)
            success_count += 1
        except Exception as e:
            print(f"❌ Error processing {dng_path.name}: {e}\n")

    print(f"\n{'='*80}")
    print(f"✅ Successfully captioned {success_count}/{len(dng_files)} images")


if __name__ == "__main__":
    main()
