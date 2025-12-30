#!/usr/bin/env python3
"""
Caption all DNG files in a directory tree.

Usage:
    python caption_dngs.py /path/to/root/directory

Writes captions to:
- XMP-dc:Description (Lightroom visible)
- XMP-Kalliste:moondream18b (model-specific field)

Resumable via logs in logs/ directory.
"""

import argparse
import base64
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import requests


def setup_logs():
    """Create logs directory if it doesn't exist."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    return log_dir


def log_completed(log_dir: Path, filepath: str):
    """Append to completed log."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_dir / "completed.log", "a") as f:
        f.write(f"{timestamp} | {filepath}\n")


def log_error(log_dir: Path, filepath: str, error: str):
    """Append to error log."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_dir / "errors.log", "a") as f:
        f.write(f"{timestamp} | {filepath} | {error}\n")


def load_completed(log_dir: Path) -> set:
    """Load set of already-processed files."""
    completed = set()
    completed_log = log_dir / "completed.log"

    if completed_log.exists():
        with open(completed_log) as f:
            for line in f:
                # Extract filepath from: "timestamp | filepath"
                parts = line.strip().split(" | ")
                if len(parts) >= 2:
                    completed.add(parts[1])

    return completed


def find_all_dngs(root_path: Path) -> list:
    """Recursively find all DNG files, sorted alphabetically."""
    dngs = []
    for dirpath, dirnames, filenames in os.walk(root_path):
        for filename in filenames:
            if filename.lower().endswith('.dng'):
                full_path = Path(dirpath) / filename
                dngs.append(str(full_path.resolve()))

    return sorted(dngs)


def extract_preview(dng_path: Path, output_path: Path = None) -> Path:
    """Extract preview/thumbnail from DNG file using exiftool.

    Tries PreviewImage first, falls back to ThumbnailTIFF if needed.
    """
    if output_path is None:
        output_path = Path(tempfile.mktemp(suffix=".jpg"))

    # Try PreviewImage first
    cmd = ["exiftool", "-b", "-PreviewImage", str(dng_path)]
    result = subprocess.run(cmd, capture_output=True)

    if result.stdout and len(result.stdout) > 0:
        with open(output_path, "wb") as f:
            f.write(result.stdout)
        return output_path

    # Fall back to ThumbnailTIFF
    cmd = ["exiftool", "-b", "-ThumbnailTIFF", str(dng_path)]
    result = subprocess.run(cmd, capture_output=True)

    if result.stdout and len(result.stdout) > 0:
        with open(output_path, "wb") as f:
            f.write(result.stdout)
        return output_path

    raise ValueError(f"No PreviewImage or ThumbnailTIFF found")


def encode_image(image_path: Path) -> str:
    """Encode image to base64 for Ollama."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def generate_caption(image_path: Path, model: str = "moondream:1.8b") -> str:
    """Generate caption using moondream via Ollama."""
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
    return result.get("response", "").strip()


def write_caption_to_dng(dng_path: Path, caption: str) -> bool:
    """Write caption to both XMP-dc:Description and XMP-Kalliste:moondream18b."""
    # Set config file path
    config_path = Path(__file__).parent / ".ExifTool_config"

    cmd = [
        "exiftool",
        "-config", str(config_path),
        "-overwrite_original",
        f"-XMP-dc:Description={caption}",
        f"-XMP-Kalliste:moondream18b={caption}",
        str(dng_path)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        return True
    else:
        raise Exception(f"exiftool error: {result.stderr}")


def process_dng(dng_path: Path, log_dir: Path) -> dict:
    """Process a single DNG file: extract preview, generate caption, write to metadata."""
    result = {
        "file": str(dng_path),
        "success": False,
        "caption": None,
        "extract_time": 0,
        "caption_time": 0,
        "write_time": 0,
        "error": None
    }

    preview_path = None

    try:
        # Extract preview
        extract_start = time.time()
        preview_path = extract_preview(dng_path)
        result["extract_time"] = time.time() - extract_start

        # Generate caption
        caption_start = time.time()
        caption = generate_caption(preview_path)
        result["caption_time"] = time.time() - caption_start
        result["caption"] = caption

        # Write to DNG metadata
        write_start = time.time()
        write_caption_to_dng(dng_path, caption)
        result["write_time"] = time.time() - write_start

        result["success"] = True
        log_completed(log_dir, str(dng_path))

    except Exception as e:
        result["error"] = str(e)
        log_error(log_dir, str(dng_path), str(e))

    finally:
        # Cleanup temp preview
        if preview_path and preview_path.exists():
            preview_path.unlink()

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Caption all DNG files in a directory tree using moondream:1.8b"
    )
    parser.add_argument(
        "root_path",
        type=str,
        help="Root directory to scan for DNG files"
    )
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        default=True,
        help="Skip files already in completed.log (default: True)"
    )

    args = parser.parse_args()

    root_path = Path(args.root_path)

    if not root_path.exists():
        print(f"❌ Error: Path does not exist: {root_path}")
        sys.exit(1)

    if not root_path.is_dir():
        print(f"❌ Error: Path is not a directory: {root_path}")
        sys.exit(1)

    # Setup
    log_dir = setup_logs()
    print(f"📁 Root path: {root_path}")
    print(f"📝 Logs directory: {log_dir}")
    print()

    # Find all DNGs
    print("🔍 Discovering DNG files...")
    all_dngs = find_all_dngs(root_path)
    print(f"   Found {len(all_dngs)} DNG files")

    # Load completed
    completed = load_completed(log_dir)
    print(f"   Already processed: {len(completed)} files")

    # Filter to pending
    pending = [dng for dng in all_dngs if dng not in completed]
    print(f"   To process: {len(pending)} files")
    print()

    if not pending:
        print("✅ All files already processed!")
        return

    # Process
    print(f"🤖 Model: moondream:1.8b (via Ollama)")
    print(f"📝 Writing to: XMP-dc:Description, XMP-Kalliste:moondream18b")
    print()

    start_time = time.time()
    successful = 0
    failed = 0

    for i, dng_path in enumerate(pending, 1):
        rel_path = Path(dng_path).relative_to(root_path.resolve())
        print(f"[{i}/{len(pending)}] {rel_path}")

        result = process_dng(Path(dng_path), log_dir)

        if result["success"]:
            successful += 1
            caption_preview = result['caption'][:80] + "..." if len(result['caption']) > 80 else result['caption']
            print(f"  ✅ {caption_preview}")
            print(f"  ⏱️  Extract: {result['extract_time']:.2f}s | Caption: {result['caption_time']:.2f}s | Write: {result['write_time']:.2f}s")
        else:
            failed += 1
            print(f"  ❌ {result['error']}")

        print()

    # Summary
    total_time = time.time() - start_time

    print("=" * 80)
    print("📊 SUMMARY")
    print("=" * 80)
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"⏱️  Total time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
    if successful > 0:
        print(f"⏱️  Avg time per image: {total_time/successful:.2f}s")
    print()
    print(f"📝 Logs saved to: {log_dir}/")
    print(f"   Completed: {log_dir / 'completed.log'}")
    if failed > 0:
        print(f"   Errors: {log_dir / 'errors.log'}")


if __name__ == "__main__":
    main()
