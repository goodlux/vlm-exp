#!/usr/bin/env python3
"""
AutoOnt - Automatic Ontology Generation from Images

Complete pipeline: Images → VLM observations → SPO parsing → Ontology building
"""

import sys
from pathlib import Path
import argparse

from observe import observe_images
from parse_spo import parse_observations
from build_ontology import OntologyBuilder, write_text_ontology


def autoont_pipeline(
    images_dir: Path,
    output_file: Path,
    vlm_model: str = "qwen3-vl:2b",
    keep_intermediates: bool = False
):
    """
    Run complete AutoOnt pipeline.

    Args:
        images_dir: Directory containing images
        output_file: Path for final ontology text file
        vlm_model: Ollama VLM model to use
        keep_intermediates: Keep observations.jsonl and triples.jsonl
    """
    print("=" * 80)
    print("AutoOnt - Automatic Ontology Generation")
    print("=" * 80)
    print()

    # Step 1: Extract observations from images
    print("STEP 1: Extracting VLM observations from images")
    print("-" * 80)
    observations_file = Path("observations.jsonl")
    observe_images(images_dir, observations_file, model=vlm_model)
    print()

    # Step 2: Parse sentences into SPO triples
    print("STEP 2: Parsing sentences into SPO triples")
    print("-" * 80)
    triples_file = Path("triples.jsonl")
    parse_observations(observations_file, triples_file)
    print()

    # Step 3: Build ontology from triples
    print("STEP 3: Building ontology from triples")
    print("-" * 80)
    builder = OntologyBuilder()
    builder.load_triples(triples_file)
    ontology = builder.build_ontology()
    print()

    # Step 4: Write text output
    print("STEP 4: Writing ontology to file")
    print("-" * 80)
    write_text_ontology(ontology, output_file)
    print()

    # Cleanup
    if not keep_intermediates:
        print("Cleaning up intermediate files...")
        observations_file.unlink(missing_ok=True)
        triples_file.unlink(missing_ok=True)
    else:
        print(f"Intermediate files kept:")
        print(f"  - {observations_file}")
        print(f"  - {triples_file}")

    print()
    print("=" * 80)
    print("✓ AutoOnt pipeline complete!")
    print(f"✓ Ontology saved to: {output_file}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='AutoOnt - Generate ontology automatically from images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python autoont.py ../images/test4
        """
    )

    parser.add_argument(
        'images_dir',
        type=Path,
        help='Directory containing images to process'
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.images_dir.exists():
        print(f"Error: Images directory not found: {args.images_dir}")
        sys.exit(1)

    if not args.images_dir.is_dir():
        print(f"Error: {args.images_dir} is not a directory")
        sys.exit(1)

    # Auto-generate output filename from input dir name
    output_file = Path(f"outputs/{args.images_dir.name}_ontology.txt")
    output_file.parent.mkdir(exist_ok=True)

    # Run pipeline with defaults
    try:
        autoont_pipeline(
            args.images_dir,
            output_file,
            vlm_model="qwen3-vl:8b",
            keep_intermediates=True  # Always keep intermediates
        )
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
