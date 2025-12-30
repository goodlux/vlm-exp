#!/usr/bin/env python3
"""
Extract knowledge graphs from VLM descriptions using KGGen.
"""

from kg_gen import KGGen
from pathlib import Path
import os

# Enable verbose API logging
os.environ["LITELLM_LOG"] = "DEBUG"
import litellm
litellm.set_verbose = True


def main():
    descriptions_dir = Path("descriptions")

    if not descriptions_dir.exists():
        print(f"Error: descriptions directory not found at {descriptions_dir}")
        print("Run observe.py first to generate VLM descriptions")
        return

    txt_files = list(descriptions_dir.glob("*_NL.txt"))

    if not txt_files:
        print(f"Error: No *_NL.txt files found in {descriptions_dir}")
        return

    print(f"Found {len(txt_files)} description files")
    print()

    # Initialize KGGen
    kg = KGGen()

    for i, txt_file in enumerate(txt_files, 1):
        print("=" * 80)
        print(f"[{i}/{len(txt_files)}] Processing: {txt_file.name}")
        print("=" * 80)

        # Read VLM description
        text = txt_file.read_text()

        print(f"\nInput text ({len(text)} chars):")
        print("-" * 80)
        print(text)
        print("-" * 80)

        print("\nExtracting knowledge graph with KGGen...")
        print("Model: anthropic/claude-sonnet-4-5-20250929")
        print("Clustering: enabled (FULL)")
        print("Temperature: 0.1")
        print()

        # Extract graph
        import os
        import sys

        # Enable verbose logging
        print("→ Starting KGGen extraction...")
        sys.stdout.flush()

        print("  [1/4] Extracting entities from text...")
        sys.stdout.flush()

        graph = kg.generate(
            input_data=text,
            model="anthropic/claude-sonnet-4-5-20250929",
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            cluster=True,
            temperature=0.1,
            context="Exhaustive visual scene description. Extract ALL spatial relationships, compositional structures, visual properties, and interactions between entities."
        )

        print("  [4/4] Clustering complete!")
        sys.stdout.flush()

        # Display results
        print("\n" + "=" * 80)
        print("RESULTS")
        print("=" * 80)

        print(f"\nEntities ({len(graph.entities)}):")
        for entity in sorted(graph.entities):
            print(f"  - {entity}")

        print(f"\nPredicates ({len(graph.edges)}):")
        for edge in sorted(graph.edges):
            print(f"  - {edge}")

        print(f"\nRelations ({len(graph.relations)}):")
        for subj, pred, obj in sorted(graph.relations):
            print(f"  ({subj}) --[{pred}]--> ({obj})")

        if graph.entity_clusters:
            print(f"\nEntity Clusters ({len(graph.entity_clusters)}):")
            for canonical, variants in graph.entity_clusters.items():
                if len(variants) > 1:
                    print(f"  {canonical} ← {variants}")

        if graph.edge_clusters:
            print(f"\nEdge Clusters ({len(graph.edge_clusters)}):")
            for canonical, variants in graph.edge_clusters.items():
                if len(variants) > 1:
                    print(f"  {canonical} ← {variants}")

        # Save graph
        import json
        output_file = Path(f"graphs/{txt_file.stem}_graph.json")
        output_file.parent.mkdir(exist_ok=True)

        with output_file.open('w') as f:
            graph_dict = {
                'entities': list(graph.entities),
                'edges': list(graph.edges),
                'relations': [list(r) for r in graph.relations],
            }
            if graph.entity_clusters:
                graph_dict['entity_clusters'] = {k: list(v) for k, v in graph.entity_clusters.items()}
            if graph.edge_clusters:
                graph_dict['edge_clusters'] = {k: list(v) for k, v in graph.edge_clusters.items()}
            json.dump(graph_dict, f, indent=2)

        print(f"\n✓ Graph saved to: {output_file}")

        # Visualize
        viz_file = Path(f"graphs/{txt_file.stem}_graph.html")
        KGGen.visualize(graph, str(viz_file), open_in_browser=False)
        print(f"✓ Visualization saved to: {viz_file}")
        print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
