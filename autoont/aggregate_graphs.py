#!/usr/bin/env python3
"""
Aggregate individual knowledge graphs into a unified graph.

Loads all graph JSON files and combines them using KGGen's aggregation.
"""

from kg_gen import KGGen
from kg_gen.models import Graph
from kg_gen.steps._3_deduplicate import DeduplicateMethod
from pathlib import Path
import json
import os
from datetime import datetime

# Enable verbose API logging
os.environ["LITELLM_LOG"] = "DEBUG"
import litellm
litellm.set_verbose = True


def load_graph_from_json(json_path: Path) -> Graph:
    """Load a Graph object from JSON file."""
    with json_path.open('r') as f:
        data = json.load(f)

    # Convert lists to sets and tuples
    graph_data = {
        'entities': set(data.get('entities', [])),
        'edges': set(data.get('edges', [])),
        'relations': set(tuple(r) for r in data.get('relations', [])),
    }

    # Add optional cluster data
    if 'entity_clusters' in data:
        graph_data['entity_clusters'] = {
            k: set(v) for k, v in data['entity_clusters'].items()
        }

    if 'edge_clusters' in data:
        graph_data['edge_clusters'] = {
            k: set(v) for k, v in data['edge_clusters'].items()
        }

    return Graph(**graph_data)


def main():
    graphs_dir = Path("graphs")

    if not graphs_dir.exists():
        print(f"Error: graphs directory not found at {graphs_dir}")
        print("Run extract_graphs.py first to generate individual graphs")
        return

    # Find all graph JSON files
    json_files = list(graphs_dir.glob("*_graph.json"))

    if not json_files:
        print(f"Error: No *_graph.json files found in {graphs_dir}")
        return

    print(f"Found {len(json_files)} graph files")
    print()

    # Load all graphs
    print("Loading individual graphs...")
    graphs = []
    for json_file in sorted(json_files):
        print(f"  Loading: {json_file.name}")
        graph = load_graph_from_json(json_file)
        print(f"    - {len(graph.entities)} entities, {len(graph.relations)} relations")
        graphs.append(graph)

    print()
    print("=" * 80)
    print("AGGREGATING GRAPHS")
    print("=" * 80)

    # Aggregate using KGGen (with retrieval model for LLM clustering)
    # Using Claude Sonnet 4.5 with extended thinking for high-quality clustering
    kg = KGGen(
        model="anthropic/claude-sonnet-4-5-20250929",
        retrieval_model="sentence-transformers/all-mpnet-base-v2",
        reasoning_effort="high",  # Enable extended thinking
        temperature=0.0
    )

    print("\nCombining all graphs...")
    combined_graph = kg.aggregate(graphs)

    print(f"\n✓ Combined graph created:")
    print(f"  - Total entities: {len(combined_graph.entities)}")
    print(f"  - Total edges: {len(combined_graph.edges)}")
    print(f"  - Total relations: {len(combined_graph.relations)}")

    # Cluster using FULL method (SEMHASH + LLM)
    print("\n" + "=" * 80)
    print("CLUSTERING COMBINED GRAPH")
    print("=" * 80)
    print("\nApplying FULL clustering to consolidate entities and relations...")
    print("Method: FULL (SEMHASH + LLM deduplication)")
    print("Model: anthropic/claude-sonnet-4-5-20250929 (with extended thinking)")
    print("Reasoning effort: HIGH")
    print()
    print("Stage 1: SEMHASH clustering (fast, no API calls)")
    print("Stage 2: LLM-based deduplication with reasoning (API calls below)")
    print()

    import sys
    sys.stdout.flush()

    # Counter for API calls
    call_counter = {'count': 0}

    # Patch DSPy to show detailed prompts/responses
    try:
        import dspy
        original_forward = dspy.Predict.forward

        def verbose_forward(self, **kwargs):
            call_counter['count'] += 1

            # Always show what we're being asked
            print(f"\n{'='*60}")
            print(f"API Call #{call_counter['count']}")
            print(f"{'='*60}")
            print(f"kwargs keys: {list(kwargs.keys())}")

            # Show what we're asking about
            for key, value in kwargs.items():
                if key == 'context':
                    print(f"  context: {value[:100]}..." if len(str(value)) > 100 else f"  context: {value}")
                elif isinstance(value, (str, int, float, bool)):
                    print(f"  {key}: {value}")
                elif isinstance(value, (list, set)):
                    print(f"  {key}: [{len(value)} items]")
                    if len(value) > 0 and len(value) <= 10:
                        for i, item in enumerate(list(value)[:10], 1):
                            print(f"    {i}. {item}")
                    elif len(value) > 10:
                        for i, item in enumerate(list(value)[:5], 1):
                            print(f"    {i}. {item}")
                        print(f"    ... and {len(value) - 5} more")
                else:
                    print(f"  {key}: {type(value)}")

            sys.stdout.flush()

            # Call original
            result = original_forward(self, **kwargs)

            # Show what the LLM decided
            print(f"\nResult:")
            if hasattr(result, 'representative'):
                print(f"  ✓ LLM chose representative: \"{result.representative}\"")
                if 'cluster' in kwargs and len(kwargs['cluster']) > 1:
                    merged = [item for item in kwargs['cluster'] if item != result.representative]
                    if merged:
                        print(f"  Merged: {merged} → {result.representative}")
            elif hasattr(result, 'duplicates'):
                if result.duplicates:
                    print(f"  ✓ LLM found {len(result.duplicates)} duplicates:")
                    for dup in list(result.duplicates)[:10]:
                        print(f"    - {dup}")
                    if len(result.duplicates) > 10:
                        print(f"    ... and {len(result.duplicates) - 10} more")
                else:
                    print(f"  ✗ No duplicates found")
            elif hasattr(result, '_store'):
                print(f"  _store keys: {list(result._store.keys())}")
                for key, value in result._store.items():
                    if isinstance(value, (list, set)):
                        print(f"    {key}: [{len(value)} items]")
                        if len(value) <= 5:
                            for item in value:
                                print(f"      - {item}")
                    else:
                        print(f"    {key}: {value}")
            else:
                print(f"  ⚠ Unknown result structure: {type(result)}")

            print(f"{'='*60}\n")
            sys.stdout.flush()

            return result

        dspy.Predict.forward = verbose_forward
        print("  → Verbose DSPy logging enabled")
    except Exception as e:
        print(f"  → Could not enable DSPy verbose logging: {e}")

    sys.stdout.flush()

    # Run clustering with FULL method (SEMHASH + LLM with reasoning)
    # Model already configured in KGGen constructor above
    clustered_graph = kg.cluster(
        combined_graph,
        method=DeduplicateMethod.FULL,
        context="Visual scene descriptions from multiple related images"
    )

    print("\n✓ Clustering complete!")
    print(f"  - Final entities: {len(clustered_graph.entities)}")
    print(f"  - Final edges: {len(clustered_graph.edges)}")
    print(f"  - Final relations: {len(clustered_graph.relations)}")

    if clustered_graph.entity_clusters:
        # Count meaningful clusters (more than 1 variant)
        meaningful_entity_clusters = sum(
            1 for variants in clustered_graph.entity_clusters.values()
            if len(variants) > 1
        )
        print(f"  - Entity clusters: {meaningful_entity_clusters} consolidations")

    if clustered_graph.edge_clusters:
        meaningful_edge_clusters = sum(
            1 for variants in clustered_graph.edge_clusters.values()
            if len(variants) > 1
        )
        print(f"  - Edge clusters: {meaningful_edge_clusters} consolidations")

    # Display results
    print("\n" + "=" * 80)
    print("AGGREGATED KNOWLEDGE GRAPH")
    print("=" * 80)

    print(f"\nEntities ({len(clustered_graph.entities)}):")
    for entity in sorted(clustered_graph.entities)[:50]:  # Show first 50
        print(f"  - {entity}")
    if len(clustered_graph.entities) > 50:
        print(f"  ... and {len(clustered_graph.entities) - 50} more")

    print(f"\nPredicates ({len(clustered_graph.edges)}):")
    for edge in sorted(clustered_graph.edges):
        print(f"  - {edge}")

    print(f"\nSample Relations (showing 30 of {len(clustered_graph.relations)}):")
    for i, (subj, pred, obj) in enumerate(sorted(clustered_graph.relations)[:30], 1):
        print(f"  {i:2d}. ({subj}) --[{pred}]--> ({obj})")

    if clustered_graph.entity_clusters:
        print(f"\nEntity Consolidations (examples):")
        consolidation_count = 0
        for canonical, variants in sorted(clustered_graph.entity_clusters.items()):
            if len(variants) > 1:
                print(f"  {canonical} ← {sorted(variants)}")
                consolidation_count += 1
                if consolidation_count >= 20:  # Show first 20
                    remaining = sum(1 for v in clustered_graph.entity_clusters.values() if len(v) > 1) - 20
                    if remaining > 0:
                        print(f"  ... and {remaining} more consolidations")
                    break

    if clustered_graph.edge_clusters:
        print(f"\nEdge Consolidations (examples):")
        consolidation_count = 0
        for canonical, variants in sorted(clustered_graph.edge_clusters.items()):
            if len(variants) > 1:
                print(f"  {canonical} ← {sorted(variants)}")
                consolidation_count += 1
                if consolidation_count >= 20:  # Show first 20
                    remaining = sum(1 for v in clustered_graph.edge_clusters.values() if len(v) > 1) - 20
                    if remaining > 0:
                        print(f"  ... and {remaining} more consolidations")
                    break

    # Save aggregated graph
    print("\n" + "=" * 80)
    print("SAVING")
    print("=" * 80)

    output_dir = Path("aggregated")
    output_dir.mkdir(exist_ok=True)

    # Generate timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"aggregated_{timestamp}.json"

    with output_file.open('w') as f:
        graph_dict = {
            'entities': sorted(clustered_graph.entities),
            'edges': sorted(clustered_graph.edges),
            'relations': [list(r) for r in sorted(clustered_graph.relations)],
            'metadata': {
                'source_graphs': len(graphs),
                'aggregation_method': 'kg_gen.aggregate',
                'clustering_method': 'FULL (SEMHASH + LLM with extended thinking)',
                'extraction_model': 'anthropic/claude-sonnet-4-5-20250929',
                'clustering_model': 'anthropic/claude-sonnet-4-5-20250929',
                'reasoning_effort': 'high',
                'timestamp': timestamp
            }
        }

        if clustered_graph.entity_clusters:
            graph_dict['entity_clusters'] = {
                k: sorted(v) for k, v in sorted(clustered_graph.entity_clusters.items())
            }

        if clustered_graph.edge_clusters:
            graph_dict['edge_clusters'] = {
                k: sorted(v) for k, v in sorted(clustered_graph.edge_clusters.items())
            }

        json.dump(graph_dict, f, indent=2)

    print(f"\n✓ Graph saved to: {output_file}")

    # Visualize
    viz_file = output_dir / f"aggregated_{timestamp}.html"
    KGGen.visualize(clustered_graph, str(viz_file), open_in_browser=False)
    print(f"✓ Visualization saved to: {viz_file}")

    print("\n" + "=" * 80)
    print("COMPLETE!")
    print("=" * 80)
    print(f"\nAggregated {len(graphs)} graphs into a unified knowledge graph")
    print(f"Open {viz_file} in a browser to explore the graph visually")
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
