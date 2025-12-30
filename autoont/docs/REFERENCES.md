# References

## Knowledge Graph Generation

### KGGen: An Open-Source Tool for Generative Knowledge Graph Construction
- **Paper**: https://arxiv.org/abs/2502.09956
- **HTML**: https://arxiv.org/html/2502.09956v2
- **GitHub**: https://github.com/stair-lab/kg-gen
- **Summary**: LLM-based knowledge graph extraction from plain text with intelligent entity/relation deduplication
- **Key Innovation**: Two-stage clustering (embeddings + LLM) to consolidate near-synonyms and reduce graph sparsity
- **Relevance to AutoOnt**: AutoOnt uses KGGen for SPO extraction from VLM descriptions. KGGen's approach validates our original design: LLM-based extraction + statistical clustering for ontology building.

## Related Work

### Formal Concept Analysis (FCA)
- Used in AutoOnt for extracting hierarchical relationships from the FCA matrix
- Classes related by predicate subsumption → parent/child relationships

### Vision-Language Models
- **Qwen-VL**: Multi-modal model family used for image description generation
- Used in AutoOnt step 1 (observe.py) to generate detailed natural language descriptions

## Open Questions

1. Should we adopt KGGen's two-stage clustering approach (embeddings + LLM verification)?
2. Can we use KGGen as a library for step 2 instead of building our own?
3. How does relation extraction quality compare: KGGen vs simple LLM prompting?
