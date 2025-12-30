#!/usr/bin/env python3
"""
Test KGGen on sample VLM description.

This script tests KGGen's ability to extract knowledge graphs from
natural language image descriptions produced by VLMs.
"""

from kg_gen import KGGen
from pathlib import Path


def test_single_description():
    """Test KGGen on a single text description."""

    # Sample VLM description (typical output from qwen3-vl)
    sample_text = """
    This image captures an ancient stone building with weathered light gray columns
    standing against a blue sky with scattered white clouds. A person in a blue shirt
    walks near the building's base, while a woman in a brown dress moves through the
    midground. The building's partially broken roof reveals its age, and some columns
    show signs of damage or are missing entirely. The ground is covered with uneven
    white stone fragments and sparse green grass growing between them. Bright sunlight
    illuminates the scene, creating distinct shadows across the ancient structure.
    """

    print("=" * 80)
    print("Testing KGGen on Sample VLM Description")
    print("=" * 80)
    print(f"\nInput text ({len(sample_text)} chars):")
    print("-" * 80)
    print(sample_text.strip())
    print("-" * 80)

    # Initialize KGGen with Ollama
    kg = KGGen()

    print("\nExtracting knowledge graph...")
    print("Model: ollama_chat/deepseek-r1:14b")
    print("Clustering: enabled")

    graph = kg.generate(
        input_data=sample_text,
        model="ollama_chat/deepseek-r1:14b",
        cluster=True,
        temperature=0.0,
        context="Visual scene description from image"
    )

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
    output_file = Path("test_graph.json")
    import json
    with output_file.open('w') as f:
        # Convert graph to dict for JSON serialization
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
    viz_file = Path("test_graph.html")
    KGGen.visualize(graph, str(viz_file), open_in_browser=False)
    print(f"✓ Visualization saved to: {viz_file}")

    return graph


def test_from_file():
    """Test KGGen on actual VLM description file."""

    descriptions_dir = Path("descriptions")

    if not descriptions_dir.exists():
        print(f"\nNo descriptions directory found at {descriptions_dir}")
        print("Run observe.py first to generate VLM descriptions")
        return None

    # Get first description file
    txt_files = list(descriptions_dir.glob("*_NL.txt"))

    if not txt_files:
        print(f"\nNo *_NL.txt files found in {descriptions_dir}")
        return None

    txt_file = txt_files[0]
    text = txt_file.read_text()

    print("\n" + "=" * 80)
    print(f"Testing KGGen on Real VLM Description: {txt_file.name}")
    print("=" * 80)
    print(f"\nInput text ({len(text)} chars):")
    print("-" * 80)
    print(text)
    print("-" * 80)

    kg = KGGen()

    print("\nExtracting knowledge graph...")
    graph = kg.generate(
        input_data=text,
        model="ollama_chat/deepseek-r1:14b",
        cluster=True,
        temperature=0.0,
        context="Visual scene description from image"
    )

    print(f"\n✓ Extracted {len(graph.relations)} relations")
    print(f"✓ Found {len(graph.entities)} entities")
    print(f"✓ Found {len(graph.edges)} predicates")

    return graph


if __name__ == "__main__":
    import sys

    # Test 1: Sample text
    print("\n" + "=" * 80)
    print("TEST 1: Sample VLM Description")
    print("=" * 80)

    try:
        graph1 = test_single_description()
        print("\n✓ Test 1 complete")
    except Exception as e:
        print(f"\n✗ Test 1 failed: {e}")
        import traceback
        traceback.print_exc()

    # Test 2: Real file (if available)
    print("\n\n" + "=" * 80)
    print("TEST 2: Real VLM Description File")
    print("=" * 80)

    try:
        graph2 = test_from_file()
        if graph2:
            print("\n✓ Test 2 complete")
        else:
            print("\n⊘ Test 2 skipped (no files available)")
    except Exception as e:
        print(f"\n✗ Test 2 failed: {e}")
        import traceback
        traceback.print_exc()
