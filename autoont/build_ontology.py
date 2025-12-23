#!/usr/bin/env python3
"""
Build ontology from SPO triples using statistical clustering and FCA.

Pipeline:
1. Cluster predicates by embedding similarity
2. Cluster nouns (subjects/objects) by embedding similarity
3. Name clusters by mode (most frequent term)
4. Build FCA matrix (which classes participate in which relations)
5. Extract concept lattice for hierarchy
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import HDBSCAN


class OntologyBuilder:
    """Build ontology from SPO triples using statistical methods."""

    def __init__(self):
        print("Loading sentence transformer...")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.triples = []

    def load_triples(self, triples_path: Path):
        """Load SPO triples from JSONL."""
        with triples_path.open() as f:
            self.triples = [json.loads(line) for line in f]
        print(f"Loaded {len(self.triples)} triples")

    def cluster_by_embedding(self, strings: List[str], min_cluster_size: int = 2) -> Dict[str, List[str]]:
        """
        Cluster similar strings using embeddings.

        Returns:
            Dict mapping cluster_id to list of strings in that cluster
        """
        if not strings:
            return {}

        unique_strings = list(set(strings))

        if len(unique_strings) < min_cluster_size:
            # Too few items, treat each as its own cluster
            return {s: [s] for s in unique_strings}

        # Embed all strings
        print(f"  Embedding {len(unique_strings)} unique strings...")
        embeddings = self.embedder.encode(unique_strings)

        # Cluster with HDBSCAN (auto-determines number of clusters)
        print(f"  Clustering...")
        clusterer = HDBSCAN(min_cluster_size=min_cluster_size, metric='cosine')
        labels = clusterer.fit_predict(embeddings)

        # Group by cluster
        clusters = {}
        for string, label in zip(unique_strings, labels):
            if label == -1:
                # Noise / singleton cluster
                cluster_id = f"singleton_{string}"
                clusters[cluster_id] = [string]
            else:
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(string)

        print(f"  Found {len(clusters)} clusters")
        return clusters

    def canonicalize_clusters(self, clusters: Dict, all_strings: List[str]) -> Dict[str, Dict]:
        """
        Name each cluster by its mode (most frequent term in original data).

        Returns:
            Dict mapping canonical_name to cluster metadata
        """
        ontology = {}

        for cluster_id, cluster_members in clusters.items():
            # Count frequency of each member in original data
            freq = Counter([s for s in all_strings if s in cluster_members])

            if freq:
                # Mode = most frequent term
                canonical = freq.most_common(1)[0][0]
                total_freq = sum(freq.values())
            else:
                # Fallback
                canonical = cluster_members[0]
                total_freq = 0

            ontology[canonical] = {
                'canonical': canonical,
                'variants': sorted(cluster_members),
                'frequency': total_freq,
                'variant_count': len(cluster_members)
            }

        return ontology

    def build_ontology(self) -> Dict:
        """
        Build complete ontology from triples.

        Returns:
            Dict with predicates, classes, and statistics
        """
        # Extract all predicates and nouns
        all_predicates = [t['predicate'] for t in self.triples]
        all_subjects = [t['subject'] for t in self.triples]
        all_objects = [t['object'] for t in self.triples]
        all_nouns = all_subjects + all_objects

        print("\n=== Clustering Predicates ===")
        pred_clusters = self.cluster_by_embedding(all_predicates, min_cluster_size=2)
        predicate_ontology = self.canonicalize_clusters(pred_clusters, all_predicates)

        print("\n=== Clustering Nouns ===")
        noun_clusters = self.cluster_by_embedding(all_nouns, min_cluster_size=2)
        class_ontology = self.canonicalize_clusters(noun_clusters, all_nouns)

        print("\n=== Building FCA Matrix ===")
        fca_matrix = self.build_fca_matrix(class_ontology, predicate_ontology)

        print("\n=== Extracting Hierarchy ===")
        hierarchy = self.extract_simple_hierarchy(fca_matrix)

        return {
            'predicates': predicate_ontology,
            'classes': class_ontology,
            'fca_matrix': fca_matrix,
            'hierarchy': hierarchy,
            'stats': {
                'total_triples': len(self.triples),
                'unique_predicates': len(predicate_ontology),
                'unique_classes': len(class_ontology),
            }
        }

    def build_fca_matrix(self, class_ontology: Dict, predicate_ontology: Dict) -> Dict:
        """
        Build FCA matrix: which classes participate in which relations?

        Matrix[class][predicate] = True if any triple has (class_variant, pred_variant, *)
        """
        matrix = {}

        for class_name, class_data in class_ontology.items():
            matrix[class_name] = {}

            for pred_name, pred_data in predicate_ontology.items():
                # Check if any triple has this (class, predicate) combination
                # Check both as subject and object
                participates = any(
                    (t['subject'] in class_data['variants'] and t['predicate'] in pred_data['variants']) or
                    (t['object'] in class_data['variants'] and t['predicate'] in pred_data['variants'])
                    for t in self.triples
                )
                matrix[class_name][pred_name] = participates

        return matrix

    def extract_simple_hierarchy(self, fca_matrix: Dict) -> List[Tuple[str, str]]:
        """
        Extract simple hierarchy based on predicate subsumption.

        If class A participates in all predicates that class B does (and more),
        then A is more specific than B.

        Returns:
            List of (parent, child) relationships
        """
        classes = list(fca_matrix.keys())
        hierarchy = []

        for i, class_a in enumerate(classes):
            for class_b in classes[i+1:]:
                preds_a = set(p for p, v in fca_matrix[class_a].items() if v)
                preds_b = set(p for p, v in fca_matrix[class_b].items() if v)

                if preds_a > preds_b:  # A is superset (more specific)
                    hierarchy.append((class_b, class_a))  # B is parent of A
                elif preds_b > preds_a:  # B is superset
                    hierarchy.append((class_a, class_b))  # A is parent of B

        return hierarchy


def write_text_ontology(ontology: Dict, output_path: Path):
    """Write ontology to human-readable text file."""

    with output_path.open('w') as f:
        f.write("=" * 80 + "\n")
        f.write("AUTOONT - Automatically Generated Ontology\n")
        f.write("=" * 80 + "\n\n")

        # Stats
        f.write("=== Statistics ===\n")
        for key, value in ontology['stats'].items():
            f.write(f"{key}: {value}\n")
        f.write("\n")

        # Predicates
        f.write("=" * 80 + "\n")
        f.write("=== PREDICATES ===\n")
        f.write("=" * 80 + "\n\n")

        predicates = sorted(ontology['predicates'].items(),
                          key=lambda x: x[1]['frequency'],
                          reverse=True)

        for canonical, data in predicates:
            f.write(f"{canonical}\n")
            f.write(f"  Frequency: {data['frequency']}\n")
            if len(data['variants']) > 1:
                f.write(f"  Variants: {', '.join(data['variants'])}\n")
            f.write("\n")

        # Classes
        f.write("=" * 80 + "\n")
        f.write("=== CLASSES ===\n")
        f.write("=" * 80 + "\n\n")

        classes = sorted(ontology['classes'].items(),
                        key=lambda x: x[1]['frequency'],
                        reverse=True)

        for canonical, data in classes:
            f.write(f"{canonical}\n")
            f.write(f"  Frequency: {data['frequency']}\n")
            if len(data['variants']) > 1:
                f.write(f"  Variants: {', '.join(data['variants'])}\n")

            # Show which predicates this class participates in
            predicates = [p for p, v in ontology['fca_matrix'][canonical].items() if v]
            if predicates:
                f.write(f"  Predicates: {', '.join(predicates[:10])}")
                if len(predicates) > 10:
                    f.write(f" ... (+{len(predicates)-10} more)")
                f.write("\n")
            f.write("\n")

        # Hierarchy
        if ontology['hierarchy']:
            f.write("=" * 80 + "\n")
            f.write("=== HIERARCHY ===\n")
            f.write("=" * 80 + "\n\n")

            for parent, child in ontology['hierarchy']:
                f.write(f"{parent}\n")
                f.write(f"  └─ {child}\n")
                f.write("\n")

    print(f"\nOntology written to: {output_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Build ontology from SPO triples')
    parser.add_argument('--input', type=Path, required=True, help='Input triples JSONL')
    parser.add_argument('--output', type=Path, default=Path('ontology.txt'), help='Output ontology text file')

    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    builder = OntologyBuilder()
    builder.load_triples(args.input)

    ontology = builder.build_ontology()

    write_text_ontology(ontology, args.output)

    print("\n✓ Ontology generation complete!")


if __name__ == '__main__':
    main()
