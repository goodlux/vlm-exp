#!/usr/bin/env python3
"""Test moondream3 multi-skill capabilities: caption + detect + segment."""

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


def test_all_skills(image_path: Path):
    """Test caption, detect, and segment on one image."""

    print(f"\n{'='*80}")
    print(f"🖼️  Testing All Skills on: {image_path.name}")
    print(f"{'='*80}\n")

    # Connect to moondream-station
    print("Connecting to moondream station...")
    model = md.vl(endpoint="http://localhost:2020/v1")

    # Load image once
    image = Image.open(image_path)
    print(f"✅ Loaded image: {image.size}\n")

    # Test 1: Caption
    print("📝 CAPTION")
    print("-" * 80)
    start = time.time()
    caption_result = model.query(image, "Describe this image in detail.")
    caption_time = time.time() - start
    caption = caption_result.get("answer", "")
    print(f"Caption: {caption}")
    print(f"⏱️  {caption_time:.2f}s\n")

    # Test 2: Detect objects/people
    print("🔍 DETECT")
    print("-" * 80)
    start = time.time()
    try:
        detect_result = model.detect(image, "person")
        detect_time = time.time() - start

        if detect_result and "objects" in detect_result:
            objects = detect_result["objects"]
            print(f"Found {len(objects)} person(s)")
            for i, obj in enumerate(objects):
                print(f"  Person {i+1}: bbox({obj['x_min']:.3f}, {obj['y_min']:.3f}, {obj['x_max']:.3f}, {obj['y_max']:.3f})")
        else:
            print("No detections")
        print(f"⏱️  {detect_time:.2f}s\n")
    except Exception as e:
        print(f"❌ Detect failed: {e}\n")
        detect_time = 0

    # Test 3: Point to specific elements
    print("🎯 POINT")
    print("-" * 80)
    start = time.time()
    try:
        point_result = model.point(image, "Where is the person's head?")
        point_time = time.time() - start

        if point_result and "points" in point_result:
            points = point_result["points"]
            for i, pt in enumerate(points):
                print(f"  Point {i+1}: ({pt['x']:.3f}, {pt['y']:.3f})")
        print(f"⏱️  {point_time:.2f}s\n")
    except Exception as e:
        print(f"❌ Point failed: {e}\n")
        point_time = 0

    # Test 4: Segment
    print("✂️  SEGMENT")
    print("-" * 80)
    start = time.time()
    try:
        segment_result = model.segment(image, "person")
        segment_time = time.time() - start

        if segment_result and "masks" in segment_result:
            masks = segment_result["masks"]
            print(f"Generated {len(masks)} mask(s)")
            for i, mask in enumerate(masks[:1]):  # Show first mask preview
                if "path" in mask:
                    path = mask["path"]
                    print(f"  Mask {i+1} SVG path (preview): {path[:100]}...")
        elif segment_result:
            print(f"Segment result keys: {list(segment_result.keys()) if isinstance(segment_result, dict) else type(segment_result)}")
        print(f"⏱️  {segment_time:.2f}s\n")
    except Exception as e:
        print(f"❌ Segment failed: {e}\n")
        segment_time = 0

    # Summary
    total_time = caption_time + detect_time + point_time + segment_time
    print("="*80)
    print(f"📊 TOTAL TIME: {total_time:.2f}s")
    print(f"  Caption: {caption_time:.2f}s")
    print(f"  Detect:  {detect_time:.2f}s")
    print(f"  Point:   {point_time:.2f}s")
    print(f"  Segment: {segment_time:.2f}s")


def main():
    dng_dir = Path("images/dance-dng")

    if not dng_dir.exists():
        dng_dir = Path(".")

    # Get all DNGs except the broken one
    dng_files = [f for f in dng_dir.glob("*.dng") if f.name != "IMG_0737.dng"]
    dng_files.sort()

    if not dng_files:
        print("No DNG files found")
        return

    print(f"Testing moondream3 skills on {len(dng_files)} images\n")

    for i, dng_path in enumerate(dng_files, 1):
        preview_path = None
        try:
            # Extract preview
            print(f"\n[{i}/{len(dng_files)}] 📤 Extracting preview from {dng_path.name}...")
            preview_path = extract_preview(dng_path)

            # Test all skills
            test_all_skills(preview_path)

        except ValueError as e:
            print(f"⚠️  Skipping {dng_path.name}: {e}\n")
        except Exception as e:
            print(f"❌ Error on {dng_path.name}: {e}\n")
        finally:
            if preview_path and preview_path.exists():
                preview_path.unlink()


if __name__ == "__main__":
    main()
