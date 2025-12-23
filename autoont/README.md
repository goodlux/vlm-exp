# AutoOnt - Automatic Ontology Generation

Generate personal ontologies automatically from image observations using VLMs and statistical clustering.

## Approach

1. **VLM Observation**: Extract simple SPO sentences from images
2. **Statistical Clustering**: Group similar predicates and nouns using embeddings
3. **Mode Naming**: Name clusters by most frequent term
4. **FCA Hierarchy**: Derive concept lattice using Formal Concept Analysis

## Pipeline

```
Images → VLM (simple sentences) → SPO parser → Embedding clusters → FCA lattice → Ontology
```

## Key Features

- **No pre-existing ontology needed** - emerges from data
- **Grounded in observations** - reflects what's actually in your images
- **Statistically derived** - reproducible, data-driven
- **Personal** - ontology reflects YOUR data, not the world

## Usage

```bash
# Install dependencies
uv sync

# Generate observations from images
python observe.py --images data/test_images/ --output observations.jsonl

# Build ontology from observations
python build_ontology.py --input observations.jsonl --output ontology.txt

# Or do both in one step
python autoont.py --images data/test_images/ --output ontology.txt
```

## Output Format

Text-based ontology with:
- Predicate clusters (canonical name + variants + frequency)
- Class clusters (canonical name + variants + frequency)
- Hierarchy (from FCA lattice)
- Sample observations for each concept

## Dependencies

- Ollama (with qwen3-vl:2b or similar VLM)
- spaCy (for SPO parsing)
- sentence-transformers (for embeddings)
- HDBSCAN (for clustering)
- concepts (for FCA)
