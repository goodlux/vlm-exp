# Semantic Fingerprint Evaluation & Personal Ontology Architecture

## Overview

This document covers two related ideas:

1. **Evaluation framework**: Comparing LLMs/VLMs by having them extract semantic observations from standardized documents, revealing their "cognitive fingerprint"
2. **Personal ontology architecture**: A practical system for auto-generating ontologies from personal data, with interoperability between agents

Both build on the same insight: different observers produce different ontologies from the same data. This divergence can be measured (eval) and bridged (architecture).

## Core Insight

Given identical input, different models produce different ontologies. This difference is signal, not noise. It reveals:

- What entities the model notices
- What relations it infers
- What abstraction level it operates at
- What it ignores or hallucinates

## Theoretical Foundation

### Relation to Formal Concept Analysis (FCA)

FCA finds structure in data by identifying "concepts"—maximal clusters of objects sharing properties. A concept is: *all objects that share a given set of properties, where that set is everything they have in common.*

In this eval:
- Each model's triples form a formal context
- Subjects = objects (rows)
- Predicates = properties (columns)
- FCA extracts the concept lattice

Different models → different lattices → different worldviews.

### Approximating Platonic Forms

Single-model output is observer-dependent. But when multiple independent observers (models with different architectures, training) converge on the same triple, that convergence is evidence of underlying structure.

| Agreement Level | Epistemological Status |
|-----------------|------------------------|
| Universal (all models) | Closest approximation to ground truth |
| High consensus | Probable structure |
| Low consensus | Perspective-dependent |
| Singleton (one model) | Model artifact / noise |

The "universal" triples are our best empirical approximation of what's actually in the data—the shadow on the cave wall that all observers agree on.

---

## Evaluation Design

### Controls (Held Constant)

- Same documents
- Same document order
- Same output format (atomic SPO triples)
- Same downstream analysis (lattice comparison, consensus calculation)

### Single Variable

- The model

### Output Format

Atomic tuples force commitment. No hedging, no prose filler.

```
(Alice, worksFor, Acme)
(Acme, locatedIn, NewYork)
(Alice, hasRole, Engineer)
```

Optional enrichment:
```
Triple: (Alice, worksFor, Acme)
Source: "Alice has been contracting with Acme since 2019"
Confidence: high
```

---

## Document Corpus

### Recommended Size

- **Minimum viable**: 30 documents (fingerprints may stabilize)
- **Robust comparison**: 50-100 documents
- **Full stylometry**: 100+ documents across varied domains

### Document Selection

| Category | Purpose |
|----------|---------|
| Mixed domains | Technical, narrative, legal, casual |
| Mixed lengths | 1 page to 20 pages |
| Mixed complexity | Simple factual to dense relational |

### Training Data Contamination

Options to ensure novelty:
- **Synthetic documents**: Guarantees unseen, but may lack naturalistic complexity
- **Recent documents**: Published after training cutoffs
- **Obscure domains**: Niche technical manuals, specialized academic fields
- **Mixed approach**: Some known (baseline), some novel (true test)

For fingerprinting, contamination matters less—you're testing perception style, not recall. For platonic approximation, novel documents are cleaner.

### Recommended Corpus: arXiv

arXiv receives ~24,000 new submissions per month (as of late 2024). This provides:

- **Guaranteed novelty**: Last month's papers are post-training-cutoff for all models
- **Structured variety**: Physics, CS, math, biology, economics—multiple domains
- **Multiple formats available**: HTML for text-only LLMs, PDF for multimodal models
- **Free and accessible**: Open access, bulk download available

**Sampling strategy**:
- Pull 100-500 papers from the most recent month
- Balance across subject areas (cs, physics, math, q-bio, etc.)
- Mix paper lengths (short letters to full papers)

### Vision Corpus: Original Photography

For VLM evaluation, use original camera photos with zero training contamination:

**Standard scene kit** (example):
- 50 indoor scenes (rooms, objects, arrangements)
- 50 outdoor scenes (streets, nature, architecture)
- 50 isolated objects (varied lighting, angles)
- 50 documents/diagrams (handwritten, printed, mixed)

Same photos for every VLM. Pure perception test—no model has seen these images before.

---

## Metrics

### Per-Model Metrics

| Metric | Definition | Reveals |
|--------|------------|---------|
| Triple count | Total triples produced | Verbosity / extraction density |
| Predicate vocabulary size | Unique predicates used | Granularity of world model |
| Hallucination rate | Triples not grounded in source | Confabulation tendency |
| Coverage | Consensus triples captured | Blind spots |
| Consistency | Same triples across multiple runs | Reliability |
| Abstraction level | Concrete vs. abstract entities | Operating level preference |

### Cross-Model Metrics

| Metric | Definition | Reveals |
|--------|------------|---------|
| Triple overlap (Jaccard) | Intersection / Union of triple sets | Raw similarity |
| Consensus rate | % of triples in majority agreement | Convergence |
| Distance from centroid | Divergence from average model | Outlier status |
| Model-to-model distance | Pairwise divergence | Clustering of model types |

### Efficiency Score

```
Efficiency = (Consensus-aligned triples) / (Total triples produced)
```

High efficiency = model says what matters without noise.

### F1 Against Consensus

```
Precision = (Your triples ∩ Consensus) / (Your triples)
Recall = (Your triples ∩ Consensus) / (Consensus triples)
F1 = harmonic mean(Precision, Recall)
```

---

## Stylometry Application

### Fingerprint Identification

With sufficient documents (50+), each model develops a recognizable signature:
- Preferred predicate vocabulary
- Granularity preferences
- Relation type biases (taxonomic vs. causal vs. compositional)
- Abstraction tendencies

### Identification Test

1. Train on N-1 documents per model
2. Given anonymous ontology from held-out document
3. Predict which model produced it

Accuracy significantly above random (1/num_models) confirms fingerprint validity.

### Fingerprint Features

- Predicate frequency distribution
- Entity granularity distribution
- Taxonomic vs. relational ratio
- Abstraction level histogram
- Hallucination patterns

---

## Consensus Ontology

### Construction

A triple enters the consensus at threshold T:

```
Consensus(T) = {triple | (models producing triple) / (total models) >= T}
```

| Threshold | Interpretation |
|-----------|----------------|
| T = 1.0 | Universal agreement only |
| T = 0.8 | Strong consensus |
| T = 0.5 | Majority rules |

### Uses

- **Ground truth proxy**: When no gold labels exist
- **Confidence weighting**: Weight triples by agreement level for downstream use
- **Model evaluation**: Compare individual models against consensus

---

## Extending to Vision and Multimodal

### Three Evaluation Tracks

| Track | Input | Models Eligible |
|-------|-------|-----------------|
| Text-only | Documents | Any LLM |
| Image-only | Images | Any VLM |
| Multimodal | Docs + embedded images | GPT-4o, Claude, Gemini, etc. |

### Why Vision is Interesting

Text has pre-segmented units (words, named entities). Images are raw perception:

- Model decides what counts as an entity
- Model decides where boundaries are
- Model decides abstraction level

Same image, wildly different triples:

| Model | Example Triples |
|-------|-----------------|
| A | (dog, beside, tree), (sky, is, blue) |
| B | (golden_retriever, sits_near, oak), (scene, suggests, afternoon) |
| C | (mammal, occupies, foreground), (vegetation, occupies, midground) |

Richer fingerprint signal than text.

### Multimodal Integration Test

Beyond isolated perception—tests synthesis:

- Does caption inform image parsing?
- Does image disambiguate text?
- Are cross-modal triples produced? `(diagram_element, illustrates, paragraph_concept)`

---

## Implementation

### Phase 1: Pilot

- 10 documents
- 3-5 models
- 1 run per model per document
- Goal: Confirm fingerprints diverge, validate pipeline

### Phase 2: Full Text Eval

- 50-100 documents
- 10+ models
- 3 runs per model per document (consistency measurement)
- Full metrics suite

### Phase 3: Vision Extension

- 50-100 images
- VLMs only
- Same metrics adapted for visual triples

### Phase 4: Multimodal

- Mixed corpus (docs with embedded images)
- Multimodal models only
- Integration-specific metrics

---

## What This Eval Offers That Others Don't

| Property | Standard Evals | This Eval |
|----------|----------------|-----------|
| Ground truth required | Yes | No (comparison-based) |
| Tests what | Performance on tasks | Perception and structuring |
| Gameable | Yes (teach to benchmark) | No (fingerprint is intrinsic) |
| Output interpretability | Scores | Inspectable divergence |
| Tests | Point knowledge | Integration across observations |
| Novel capability | - | Model identification from output |

---

## Open Questions

1. **Optimal corpus size**: At what N do fingerprints stabilize? Pilot with 30, scale to 100+.
2. **Predicate normalization**: Embed and cluster synonyms post-hoc, or leave raw for fingerprinting?
3. **Weighting by document**: Should complex documents count more than simple ones?
4. **Cross-domain transfer**: Does a model's fingerprint on CS papers predict its fingerprint on biology?
5. **Temporal stability**: Does a model's fingerprint change across versions (GPT-4 → GPT-4o)?

---

## Quick Start

1. Pull 50 recent arxiv papers (HTML format)
2. Select 3-5 models to compare
3. Prompt each: "Extract all semantic triples (subject, predicate, object) from this document"
4. Collect outputs, compute overlap metrics
5. Run FCA to generate concept lattices
6. Compare lattices, identify universal vs. singleton triples

If fingerprints diverge clearly at N=50, the method works. Scale from there.

---

# Part 2: Personal Ontology Architecture

## The Problem

Personal LLM agents will need to:
1. Build ontologies from user data automatically
2. Communicate with other agents using different ontologies
3. Bridge to standard vocabularies (FOAF, schema.org) when needed

The domain isn't "legal" or "medical"—it's **the individual's data as observed by an LLM**.

## The Pipeline

```
Raw Data (images, docs, etc.)
    ↓
VLM/LLM produces atomic sentences
    ↓
AMR parser normalizes to semantic graphs
    ↓
Embedding + clustering for concept/predicate grouping
    ↓
FCA derives hierarchy
    ↓
Ontology (versioned, stored in Git)
```

### Step 1: Atomic Sentence Extraction

Ask the VLM/LLM to produce one-line, subject-predicate-object sentences:

```
The dog is sitting on the grass.
The grass is green.
A tree is behind the dog.
The dog appears to be a golden retriever.
```

Each line: one fact, simple structure, natural phrasing.

**Why this approach:**
- VLM does what it's good at (perception + decomposition)
- Not asking it to invent ontology vocabulary
- Simple sentences parse better in downstream steps

### Step 2: AMR Normalization

Abstract Meaning Representation normalizes linguistic variation:

- "sitting on" / "resting on" / "seated upon" → same semantic frame
- Fixed vocabulary (~100 relations)
- Grounded in PropBank, FrameNet

```
(s / sit-01
   :ARG0 (d / dog)
   :location (g / grass))
```

**Why AMR:**
- Consistency comes from parser, not LLM's memory
- Principled rather than arbitrary normalization
- Interoperability: two agents using AMR can communicate cleanly

### Step 3: Statistical Ontology Derivation

**Cluster predicates:**
Embed all predicate strings. Similar ones cluster together.
"sitting_on", "resting_upon", "seated_on" → cluster → canonical name: `spatial_support`

Naming: **mode** (most frequent term in cluster) becomes the canonical predicate.

**Cluster subjects/objects:**
"Dog", "golden retriever", "canine" → cluster → class: `Dog`

**FCA for hierarchy:**
Build matrix: which classes participate in which relations?
Lattice gives you hierarchy. "Animal" above "Dog" because everything Dog does, Animal does.

**No pre-existing ontology required.** Structure emerges from:
- Embedding similarity (what's "the same")
- Co-occurrence patterns (what goes with what)
- FCA closure (what implies what)

## Multi-Agent Architecture

### Storage: Named Graphs in Oxigraph

```
Oxigraph
├── :graph-llm-a        (raw observations)
├── :graph-llm-b        (raw observations)
└── :graph-observations (pooled, for union derivation)
```

Derived ontologies are computed, not stored as authoritative:
- Ontology A (from LLM-A's observations)
- Ontology B (from LLM-B's observations)
- Ontology Union (from pooled observations)

### The Divergence as Bridge

If Ontology A has `sitting_on` and Ontology B has `resting_upon`, Union clusters them into `spatial_support`:

```
A:sitting_on    →  Union:spatial_support
B:resting_upon  →  Union:spatial_support
```

Translation A→B:
```
A:sitting_on → Union:spatial_support → B:resting_upon
```

**The union is the hub. Each ontology's divergence from the hub is a spoke.**

### Cross-Agent Search

LLM-A wants to search LLM-B's data:

```
A: "Find things where dog sitting_on grass"
     ↓ (A's divergence map)
Union: "Find things where dog spatial_support grass"
     ↓ (B's divergence map, reversed)
B: "Find things where dog resting_upon grass"
     → results from B's graph
```

Neither sees the other's ontology. Both speak to the union. Diff maps handle translation invisibly.

## Versioning: Git as Ontology Backend

### Why Git

| Git feature | Ontology benefit |
|-------------|------------------|
| Commits | Snapshots at each data ingestion |
| Diff | Exactly what changed, line by line |
| Blame | When did this concept first appear? |
| Branches | Experimental ontology variations |
| Merge | Combine ontologies from different sources |
| Tags | Mark stable versions |
| History | Full evolution, rollback anytime |

### Serialization

Turtle or N-Triples are line-based, text-friendly:

```diff
+ :golden_retriever rdfs:subClassOf :dog .
- :sitting_on owl:equivalentProperty :resting_upon .
+ :spatial_support owl:equivalentProperty :sitting_on, :resting_upon .
```

### Temporal Bridging

Same LLM, growing data:

```
Time 1: 100 docs → Ontology v1
Time 2: 200 docs → Ontology v2 (re-derived from all 200)
```

The diff between v1 and v2 is your migration path:

```
v1:sitting_on → v2:spatial_support (generalized)
v1:dog → v2:dog (unchanged)
(new) → v2:sunset (concept didn't exist before)
```

**This is semantic version control.** Old queries still work via diff chain.

### Repository Structure

```
ontology-repo/
├── observations/
│   ├── llm-a.ttl
│   └── llm-b.ttl
├── derived/
│   ├── ontology-a.ttl
│   ├── ontology-b.ttl
│   └── ontology-union.ttl
├── diffs/
│   ├── a-to-union.ttl
│   └── b-to-union.ttl
└── history/
    ├── v1/ 
    └── v2/
```

CI/CD regenerates derived files on each push.

## Bridging to Standard Ontologies

### LLM-Assisted Mapping

The LLM has seen FOAF, schema.org, Dublin Core in training. It knows them.

Workflow:
1. Auto-generate your ontology (statistical, grounded)
2. Hand to LLM with reference ontology
3. "Map my concepts to FOAF where possible. Flag what doesn't map."

```
my:person      → foaf:Person       (direct)
my:knows       → foaf:knows        (direct)
my:sitting_on  → ???               (no FOAF equivalent, keep local)
my:works_for   → schema:worksFor   (bridge to schema.org)
```

### Sovereignty

Your ontology is yours. The bridge to FOAF is just a view. You're not forcing data into someone else's schema—you're providing a translation layer.

This is Solid's vision with a practical path.

## Saturation and Confidence

### Detecting "Enough" Data

Track ontology stability across ingestion batches:

| Batch | New concepts | Changed concepts | Stable concepts |
|-------|--------------|------------------|-----------------|
| 100 docs | 50 | - | - |
| 200 docs | 20 | 10 | 40 |
| 300 docs | 8 | 5 | 57 |
| 400 docs | 3 | 2 | 68 |
| 500 docs | 1 | 1 | 71 |

When new + changed approaches zero, you're **saturated**.

Plot the change rate. When it flattens, you have enough data for stable ontology.

### Per-Concept Confidence

```turtle
:dog a :Concept ;
    :observationCount 847 ;
    :firstSeen "batch-001" ;
    :lastModified "batch-003" ;
    :stability 0.95 .
```

Stability = batches unchanged / total batches since first seen

| Pattern | Interpretation |
|---------|----------------|
| High count, early, stable | Core concept, high confidence |
| Low count, late, stable | Rare but real |
| Medium count, unstable | Still resolving |
| Low count, flickering | Noise, don't trust yet |

### Confidence-Gated Bridging

When mapping to FOAF, only map concepts above confidence threshold. Unstable concepts stay local until proven.

## Key Principles

1. **Ontology is derived, not authoritative** — regenerate anytime from observations
2. **Keep raw observations** — they're the ground truth, ontology is just a view
3. **Divergence is the bridge** — don't eliminate differences, use them for translation
4. **Domain is the individual** — not "legal" or "medical" but "my data as observed by my LLM"
5. **Version everything** — ontology evolution is semantic version control
6. **Bridge, don't force** — connect to standards via mapping, don't abandon local concepts

## Relation to FCA

Formal Concept Analysis gives us:
- **A concept** = all objects sharing a given set of properties, where that set is everything they have in common
- **The lattice** = how concepts nest by inclusion

FCA is an observer's approximation of platonic forms. Multiple observers (LLMs) triangulate toward ground truth. Universal agreement = closest to real structure.

The ontology isn't truth. It's a measurement of truth, observer-dependent but empirically grounded.