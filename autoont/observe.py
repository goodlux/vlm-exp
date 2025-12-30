#!/usr/bin/env python3
"""
Extract observations from images using VLM.

Produces simple SPO sentences that will be used for ontology building.
"""

import json
import sys
from pathlib import Path
from typing import List, Dict
import ollama
from PIL import Image
import subprocess


def extract_preview_image(dng_path: Path) -> Image.Image:
    """Extract preview image from DNG file using exiftool."""
    try:
        result = subprocess.run(
            ["exiftool", "-b", "-PreviewImage", str(dng_path)],
            capture_output=True,
            check=True
        )
        from io import BytesIO
        return Image.open(BytesIO(result.stdout))
    except subprocess.CalledProcessError:
        # Fallback: try to open directly
        return Image.open(dng_path)


def get_vlm_description(image_path: Path, model: str = "qwen3-vl:2b") -> str:
    """
    Ask VLM to describe image in natural language.

    Returns:
        Natural language description of the image
    """
    prompt = """Describe this image in EXTREME DETAIL using natural, flowing language.

Write as if you are a photographer or art critic providing a comprehensive description.
Be exhaustive and thorough about EVERYTHING that IS PRESENT in the image.

Describe in rich detail:
- ALL objects visible (foreground, midground, background)
- Colors, textures, materials, patterns
- Sizes, shapes, conditions, states
- Spatial relationships and composition
- Actions, poses, gestures, expressions
- Lighting conditions, shadows, highlights, reflections
- Any text, symbols, or writing visible
- Weather, atmosphere, mood
- Fine details and small elements
- Context and setting

IMPORTANT: Only describe what you can actually SEE. Do NOT describe what is absent or missing.
Write in complete paragraphs with natural flow. Be detailed and thorough."""

    # Load image
    print("[DEBUG] Extracting preview image...")
    if image_path.suffix.lower() == '.dng':
        img = extract_preview_image(image_path)
        # Save temp JPG for Ollama
        temp_path = Path(f"/tmp/{image_path.stem}_preview.jpg")
        img.save(temp_path)
        image_path = temp_path
        print(f"[DEBUG] Saved preview to {temp_path}")

    # Call Ollama VLM
    print(f"[DEBUG] Calling Ollama with model {model}...")
    try:
        response = ollama.chat(
            model=model,
            messages=[{
                'role': 'user',
                'content': prompt,
                'images': [str(image_path)]
            }]
        )
        print("[DEBUG] Ollama response received")
    except Exception as e:
        print(f"\n[ERROR] Ollama call failed: {e}")
        return ""

    # Return the raw text description
    text = response.get('message', {}).get('content', '').strip()
    return text


def observe_images(image_dir: Path, output_dir: Path, model: str = "qwen3-vl:2b"):
    """
    Process all images in directory and extract natural language descriptions.

    Outputs individual .txt files for each image: imagename_NL.txt
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.dng', '.tiff', '.tif'}
    image_paths = [p for p in image_dir.rglob('*') if p.suffix.lower() in image_extensions]

    print(f"Found {len(image_paths)} images in {image_dir}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    import time

    total_start = time.time()
    processing_times = []
    total_chars = 0

    for i, img_path in enumerate(image_paths, 1):
        print(f"\n{'='*80}")
        print(f"[{i}/{len(image_paths)}] Processing: {img_path.name}")
        print('='*80)

        try:
            start_time = time.time()
            description = get_vlm_description(img_path, model=model)
            elapsed = time.time() - start_time

            if description:
                # Save to individual text file
                output_file = output_dir / f"{img_path.stem}_NL.txt"
                output_file.write_text(description)

                processing_times.append(elapsed)
                total_chars += len(description)

                # Display VLM output with timing
                print(f"\nVLM Output ({len(description)} chars, {elapsed:.1f}s):")
                print("-" * 80)
                print(description)
                print("-" * 80)
                print(f"✓ Saved to: {output_file}")
            else:
                print(f"✗ No description generated")

        except Exception as e:
            print(f"\n✗ Error: {e}")
            continue

    # Summary stats
    total_elapsed = time.time() - total_start

    print(f"\n{'='*80}")
    print("TIMING SUMMARY")
    print('='*80)
    if processing_times:
        import statistics
        print(f"Total time: {total_elapsed:.1f}s")
        print(f"Images processed: {len(processing_times)}")
        print(f"Average per image: {statistics.mean(processing_times):.1f}s")
        print(f"Fastest: {min(processing_times):.1f}s")
        print(f"Slowest: {max(processing_times):.1f}s")
        print(f"Total characters: {total_chars}")
        print(f"Average chars/image: {total_chars/len(processing_times):.0f}")
    print(f"\nDescriptions saved to: {output_dir}/")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Extract VLM natural language descriptions from images')
    parser.add_argument('--images', type=Path, required=True, help='Directory of images')
    parser.add_argument('--output', type=Path, default=Path('descriptions'), help='Output directory for text files')
    parser.add_argument('--model', default='qwen3-vl:2b', help='Ollama VLM model')

    args = parser.parse_args()

    if not args.images.exists():
        print(f"Error: Image directory not found: {args.images}")
        sys.exit(1)

    observe_images(args.images, args.output, args.model)


if __name__ == '__main__':
    main()
