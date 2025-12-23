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


def get_vlm_sentences(image_path: Path, model: str = "qwen3-vl:2b") -> List[str]:
    """
    Ask VLM to describe image using simple SPO sentences.

    Returns:
        List of simple sentences in Subject-Predicate-Object format
    """
    prompt = """Describe this image in EXTREME DETAIL using simple sentences.
Each sentence should state ONE FACT. Be as thorough and specific as possible.

IMPORTANT: Produce as many sentences as you can. Describe EVERYTHING you see. Be exhaustive.

Describe EVERYTHING:
- ALL objects present (foreground, midground, background)
- Colors, textures, materials of EACH object
- Sizes, shapes, conditions
- Spatial relationships (behind, on, near, next to, above, below, left, right)
- Actions, poses, states
- Lighting, shadows, reflections
- Any text or symbols visible
- Weather, atmosphere
- Fine details and small elements

Write one sentence per line. Be thorough and specific about every element you can see. Do not add any commentary or explanation - just describe."""

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
        return []

    # Just get the text - no parsing or manipulation
    text = response.get('message', {}).get('content', '')

    # Split into lines, that's all
    sentences = [line.strip() for line in text.split('\n') if line.strip()]

    return sentences


def observe_images(image_dir: Path, output_path: Path, model: str = "qwen3-vl:2b"):
    """
    Process all images in directory and extract observations.

    Outputs JSONL file with one observation per line:
    {"image": "path/to/image.jpg", "sentences": ["...", "..."]}
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.dng', '.tiff', '.tif'}
    image_paths = [p for p in image_dir.rglob('*') if p.suffix.lower() in image_extensions]

    print(f"Found {len(image_paths)} images in {image_dir}")

    observations = []

    import time

    total_start = time.time()
    processing_times = []

    for i, img_path in enumerate(image_paths, 1):
        print(f"\n{'='*80}")
        print(f"[{i}/{len(image_paths)}] Processing: {img_path.name}")
        print('='*80)

        try:
            start_time = time.time()
            sentences = get_vlm_sentences(img_path, model=model)
            elapsed = time.time() - start_time

            observation = {
                'image': str(img_path),
                'sentences': sentences,
                'sentence_count': len(sentences)
            }

            observations.append(observation)
            processing_times.append(elapsed)

            # Display VLM output with timing
            print(f"\nVLM Output ({len(sentences)} sentences, {elapsed:.1f}s, {len(sentences)/elapsed:.1f} sent/sec):")
            for j, sentence in enumerate(sentences, 1):
                print(f"  {j}. {sentence}")

        except Exception as e:
            print(f"\n✗ Error: {e}")
            continue

    # Write JSONL output
    with output_path.open('w') as f:
        for obs in observations:
            f.write(json.dumps(obs) + '\n')

    # Summary stats
    total_elapsed = time.time() - total_start
    total_sentences = sum(o['sentence_count'] for o in observations)

    print(f"\n{'='*80}")
    print("TIMING SUMMARY")
    print('='*80)
    if processing_times:
        import statistics
        print(f"Total time: {total_elapsed:.1f}s")
        print(f"Images processed: {len(observations)}")
        print(f"Average per image: {statistics.mean(processing_times):.1f}s")
        print(f"Fastest: {min(processing_times):.1f}s")
        print(f"Slowest: {max(processing_times):.1f}s")
        print(f"Total sentences: {total_sentences}")
        print(f"Average sentences/image: {total_sentences/len(observations):.1f}")
    print(f"\nObservations saved to: {output_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Extract VLM observations from images')
    parser.add_argument('--images', type=Path, required=True, help='Directory of images')
    parser.add_argument('--output', type=Path, default=Path('observations.jsonl'), help='Output JSONL file')
    parser.add_argument('--model', default='qwen3-vl:2b', help='Ollama VLM model')

    args = parser.parse_args()

    if not args.images.exists():
        print(f"Error: Image directory not found: {args.images}")
        sys.exit(1)

    observe_images(args.images, args.output, args.model)


if __name__ == '__main__':
    main()
