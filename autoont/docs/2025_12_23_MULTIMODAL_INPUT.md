# Multi-Modal Input Enrichment for Knowledge Graph Extraction

**Status:** Speculative / Design exploration
**Date:** 2025-12-23
**Context:** Can we enrich VLM descriptions with EXIF, segmentation, and positional data to improve KG quality?

## Core Question

**Current pipeline:**
```
Image → VLM (text description) → KGGen → Knowledge Graph
```

**Proposed enrichment:**
```
Image + EXIF + Segmentation → Enhanced description → KGGen → Richer KG
```

Can KGGen extract meaningful relationships from heterogeneous data sources (text + metadata + spatial)?

## Data Source 1: EXIF / Lightroom Metadata

### What's Available

Lightroom catalogs contain rich metadata:

```python
exif_data = {
    # People (face recognition)
    "people": ["John Smith", "Jane Doe"],
    "face_regions": [
        {"name": "John Smith", "bbox": [0.2, 0.3, 0.15, 0.25]},
        {"name": "Jane Doe", "bbox": [0.6, 0.35, 0.12, 0.22]}
    ],

    # Location
    "gps": {"lat": 37.9715, "lon": 23.7267},
    "location": "Acropolis, Athens, Greece",
    "country": "Greece",
    "city": "Athens",

    # Technical
    "camera": "Sony A7R V",
    "lens": "24-70mm f/2.8",
    "focal_length": "35mm",
    "aperture": "f/5.6",
    "iso": 200,
    "shutter_speed": "1/250",

    # Content metadata
    "keywords": ["architecture", "ancient", "tourism", "historical"],
    "rating": 4,
    "label": "Green",
    "caption": "Tourist visiting the Parthenon on a sunny afternoon",

    # Temporal
    "date_taken": "2024-06-15 14:32:18",
    "date_modified": "2024-06-20 10:15:43"
}
```

### Integration Approach A: Append to VLM Description

```python
def create_enriched_description(image_path):
    """Combine VLM output with EXIF metadata."""

    # Get VLM natural language description
    vlm_text = get_vlm_description(image_path)

    # Extract EXIF/Lightroom data
    exif = read_exif(image_path)
    lr_metadata = read_lightroom_metadata(image_path)

    # Format as structured text
    metadata_text = f"""

METADATA:
Location: {lr_metadata.get('location', 'Unknown')}
Date: {exif.get('date_taken', 'Unknown')}
People in photo: {', '.join(lr_metadata.get('people', []))}
Keywords: {', '.join(lr_metadata.get('keywords', []))}
Photographer caption: {lr_metadata.get('caption', '')}
    """

    # Combine
    full_description = vlm_text + metadata_text

    return full_description
```

**Example combined input:**
```
This image captures an ancient stone building with weathered light gray columns
standing against a blue sky with scattered white clouds. A person in a blue shirt
walks near the building's base, while a woman in a brown dress moves through the
midground...

METADATA:
Location: Acropolis, Athens, Greece
Date: 2024-06-15 14:32:18
People in photo: Sarah Chen, Michael Rodriguez
Keywords: architecture, ancient, tourism, UNESCO, Parthenon
Photographer caption: Tourists exploring the Parthenon on a beautiful summer afternoon
```

**Can KGGen handle this?**

Let's think about what KGGen would extract:

```python
Entities:
- "ancient stone building"
- "weathered light gray columns"
- "blue sky"
- "person in blue shirt"
- "woman in brown dress"
- "Acropolis"  # from metadata
- "Athens"     # from metadata
- "Greece"     # from metadata
- "Sarah Chen" # from metadata
- "Michael Rodriguez" # from metadata
- "Parthenon"  # from metadata

Relations:
- (ancient stone building, located_in, Acropolis)
- (Acropolis, located_in, Athens)
- (Athens, located_in, Greece)
- (Sarah Chen, appears_in, photo)
- (Michael Rodriguez, appears_in, photo)
- (person in blue shirt, is, Sarah Chen)  # POTENTIAL LINK!
- (woman in brown dress, is, Michael Rodriguez)  # POTENTIAL LINK!
- (photo, taken_on, 2024-06-15)
- (Parthenon, is, ancient stone building)
```

**The key insight:** KGGen's LLM might be able to **resolve co-references** between VLM descriptions and EXIF metadata.

### Integration Approach B: Structured Metadata Section

```python
def create_structured_description(image_path):
    """Present metadata in a more structured format."""

    vlm_text = get_vlm_description(image_path)
    exif = read_exif(image_path)
    lr_metadata = read_lightroom_metadata(image_path)

    structured = f"""
VISUAL DESCRIPTION:
{vlm_text}

IDENTIFIED PEOPLE:
{format_people_list(lr_metadata.get('people', []))}

LOCATION INFORMATION:
- Place: {lr_metadata.get('location', 'Unknown')}
- GPS: {format_gps(exif.get('gps', {}))}

TEMPORAL INFORMATION:
- Captured: {exif.get('date_taken', 'Unknown')}

SEMANTIC TAGS:
{', '.join(lr_metadata.get('keywords', []))}

PHOTOGRAPHER NOTES:
{lr_metadata.get('caption', 'None')}
    """

    return structured
```

**Hypothesis:** Structured sections might help the LLM understand the semantic relationships better.

### Integration Approach C: JSON-LD Format

```python
def create_jsonld_description(image_path):
    """Use JSON-LD for truly structured metadata."""

    vlm_text = get_vlm_description(image_path)
    exif = read_exif(image_path)
    lr_metadata = read_lightroom_metadata(image_path)

    # Create schema.org structured data
    structured_data = {
        "@context": "https://schema.org",
        "@type": "Photograph",
        "description": vlm_text,
        "contentLocation": {
            "@type": "Place",
            "name": lr_metadata.get('location'),
            "geo": {
                "@type": "GeoCoordinates",
                "latitude": exif.get('gps', {}).get('lat'),
                "longitude": exif.get('gps', {}).get('lon')
            }
        },
        "about": [
            {"@type": "Person", "name": person}
            for person in lr_metadata.get('people', [])
        ],
        "dateCreated": exif.get('date_taken'),
        "keywords": lr_metadata.get('keywords', [])
    }

    # Convert to readable text for KGGen
    return format_jsonld_as_text(structured_data)
```

**Question:** Can KGGen parse JSON-LD? Or would we need to pre-process into triples?

## Data Source 2: Segmentation Data (SAM)

### What's Available

Segment Anything Model outputs:

```python
sam_output = {
    "masks": [
        {
            "id": 0,
            "area": 12540,
            "bbox": [120, 80, 180, 420],  # [x, y, width, height]
            "predicted_iou": 0.94,
            "point_coords": [[200, 250]],
            "stability_score": 0.96,
            "crop_box": [100, 60, 200, 440]
        },
        {
            "id": 1,
            "area": 8930,
            "bbox": [450, 120, 140, 380],
            "predicted_iou": 0.91,
            ...
        }
    ]
}
```

### Integration Approach A: Spatial Descriptions in Text

```python
def add_spatial_context(vlm_description, sam_masks):
    """Augment VLM text with spatial information."""

    # Parse VLM description to identify entities
    entities = extract_entities_from_text(vlm_description)

    # Match entities to SAM masks (heuristic or VLM-guided)
    entity_mask_pairs = match_entities_to_masks(entities, sam_masks)

    # Add spatial descriptions
    spatial_text = "\n\nSPATIAL INFORMATION:\n"
    for entity, mask in entity_mask_pairs:
        position = describe_position(mask['bbox'])
        size = describe_size(mask['area'])
        spatial_text += f"- {entity}: {position}, {size}\n"

    return vlm_description + spatial_text


def describe_position(bbox):
    """Convert bbox to natural language position."""
    x, y, w, h = bbox
    center_x = x + w/2
    center_y = y + h/2

    # Divide image into 3x3 grid
    if center_x < image_width/3:
        horizontal = "left"
    elif center_x < 2*image_width/3:
        horizontal = "center"
    else:
        horizontal = "right"

    if center_y < image_height/3:
        vertical = "upper"
    elif center_y < 2*image_height/3:
        vertical = "middle"
    else:
        vertical = "lower"

    return f"{vertical} {horizontal}"


def describe_size(area):
    """Convert area to relative size descriptor."""
    relative_area = area / total_image_area

    if relative_area > 0.3:
        return "large"
    elif relative_area > 0.1:
        return "medium"
    else:
        return "small"
```

**Example output:**
```
This image captures weathered columns against blue sky...

SPATIAL INFORMATION:
- column: middle left, large (15% of image)
- person in blue shirt: lower center, small (3% of image)
- sky: upper region, large (40% of image)
- stone fragments: lower right, medium (8% of image)
```

### Integration Approach B: Bounding Box Coordinates

**More precise but less natural:**
```
OBJECT LOCATIONS:
- "weathered column" at bbox [120, 80, 180, 420]
- "person in blue shirt" at bbox [450, 320, 80, 160]
- "blue sky" at bbox [0, 0, 800, 300]
```

**Question:** Can KGGen's LLM understand numerical coordinates? Or would it treat them as noise?

### Integration Approach C: Relational Descriptions

**Focus on spatial relationships:**
```python
def generate_spatial_relations(sam_masks, entity_labels):
    """Generate spatial relation text from masks."""

    relations = []
    for i, (entity1, mask1) in enumerate(entity_labels):
        for entity2, mask2 in entity_labels[i+1:]:
            rel = compute_spatial_relation(mask1, mask2)
            if rel:
                relations.append(f"{entity1} is {rel} {entity2}")

    return "\n".join(relations)


def compute_spatial_relation(mask1, mask2):
    """Compute spatial predicate between two masks."""

    bbox1 = mask1['bbox']
    bbox2 = mask2['bbox']

    # Check vertical relationship
    if bbox1['y'] + bbox1['h'] < bbox2['y']:
        return "above"
    elif bbox2['y'] + bbox2['h'] < bbox1['y']:
        return "below"

    # Check horizontal relationship
    if bbox1['x'] + bbox1['w'] < bbox2['x']:
        return "left of"
    elif bbox2['x'] + bbox2['w'] < bbox1['x']:
        return "right of"

    # Check containment
    if is_contained(bbox1, bbox2):
        return "inside"
    elif is_contained(bbox2, bbox1):
        return "contains"

    # Check overlap
    if iou(mask1, mask2) > 0.1:
        return "overlapping with"

    return "near"
```

**Example output:**
```
SPATIAL RELATIONS:
- column is above stone fragments
- column is left of person in blue shirt
- person in blue shirt is below sky
- sky is above column
- stone fragments are near column
```

**This is interesting!** We're essentially pre-computing spatial predicates that KGGen would otherwise have to infer from text.

## Data Source 3: VLM with Positional Output

### Can Qwen3-VL Output Positions?

Some VLMs support **grounded generation** - outputting bounding boxes alongside text:

```python
prompt = """
Describe this image in detail. For each object you mention, provide its
approximate location as [x1, y1, x2, y2] coordinates where the image is
1000x1000 pixels.

Format: <object description> at [x1, y1, x2, y2]
"""

# Hypothetical output:
output = """
A weathered stone column [120, 80, 300, 500] rises against a blue sky [0, 0, 1000, 300].
A person in a blue shirt [450, 320, 530, 480] stands near the column base.
Stone fragments [100, 600, 400, 800] scatter across the ground.
"""
```

**Question:** Does Qwen3-VL support this? Let's check the model capabilities.

### Alternative: Two-Pass VLM

```python
def get_grounded_description(image_path):
    """Get both description and object locations."""

    # Pass 1: Natural description
    desc = vlm.generate(image_path, prompt="Describe in detail")

    # Pass 2: Object detection
    objects = vlm.generate(
        image_path,
        prompt="List all objects with bounding boxes in format: object [x1,y1,x2,y2]"
    )

    # Combine
    return f"{desc}\n\nOBJECT LOCATIONS:\n{objects}"
```

### Alternative: Use a Vision-Language Grounding Model

Models like **GLIP**, **GDINO**, or **GroundingDINO** are specifically designed for this:

```python
from groundingdino import GroundingDINO

def get_grounded_entities(image_path, text_description):
    """Ground text entities to image regions."""

    model = GroundingDINO()

    # Extract entities from text
    entities = extract_entities(text_description)

    # Ground each entity to image
    grounded = {}
    for entity in entities:
        boxes = model.ground(image_path, text_query=entity)
        grounded[entity] = boxes

    return grounded


# Example:
entities = ["column", "person", "sky", "stone fragments"]
grounded = get_grounded_entities("acropolis.jpg", vlm_description)
# → {
#     "column": [[120, 80, 300, 500], [350, 90, 520, 510]],
#     "person": [[450, 320, 530, 480]],
#     "sky": [[0, 0, 1000, 300]],
#     ...
# }
```

## Combined Pipeline: The Full Enchilada

```python
def extract_multimodal_kg(image_path):
    """Extract KG from image using all available data sources."""

    # 1. VLM description (natural language)
    vlm_text = get_vlm_description(image_path)

    # 2. EXIF/Lightroom metadata
    exif = read_exif(image_path)
    lr_metadata = read_lightroom_metadata(image_path)

    # 3. Segmentation (SAM)
    sam_masks = segment_anything(image_path)

    # 4. Grounding (optional - GroundingDINO)
    grounded_entities = get_grounded_entities(image_path, vlm_text)

    # 5. Combine into enriched description
    enriched_description = f"""
VISUAL DESCRIPTION:
{vlm_text}

IDENTIFIED PEOPLE:
{format_people(lr_metadata.get('people', []))}

LOCATION:
{lr_metadata.get('location', 'Unknown')}
({exif.get('gps', {}).get('lat', 'N/A')}, {exif.get('gps', {}).get('lon', 'N/A')})

DATE:
{exif.get('date_taken', 'Unknown')}

KEYWORDS:
{', '.join(lr_metadata.get('keywords', []))}

SPATIAL LAYOUT:
{generate_spatial_descriptions(sam_masks, grounded_entities)}

SPATIAL RELATIONS:
{generate_spatial_relations(sam_masks, grounded_entities)}
    """

    # 6. Extract KG with KGGen
    kg = kg_gen.generate(
        enriched_description,
        cluster=True,
        context="Multi-modal image description with metadata and spatial information"
    )

    # 7. Add spatial metadata to entities
    for entity in kg.entities:
        if entity.name in grounded_entities:
            entity.spatial_data = {
                'bboxes': grounded_entities[entity.name],
                'masks': find_matching_masks(sam_masks, grounded_entities[entity.name])
            }

    return kg
```

## Expected Benefits

### 1. Entity Resolution
**Before:**
```
Entities: ["person in blue shirt", "woman in brown dress", "Sarah Chen", "Michael Rodriguez"]
Relations: None connecting them
```

**After (with EXIF people data):**
```
Entities: ["Sarah Chen", "Michael Rodriguez"]
Relations:
- (Sarah Chen, wears, blue shirt)
- (Michael Rodriguez, wears, brown dress)
- (Sarah Chen, photographed_at, Acropolis)
```

### 2. Spatial Grounding
**Before:**
```
Relations: [(child, gazes at, Parthenon)]
```

**After (with SAM/grounding):**
```
Relations:
- (child, gazes at, Parthenon)
- (child, located_below, Parthenon)  # from spatial analysis
- (child, smaller_than, Parthenon)   # from size comparison
- (child, distance_from, Parthenon, "20 meters")  # from depth estimation
```

### 3. Semantic Enrichment
**Before:**
```
Entities: ["ancient building"]
Relations: None
```

**After (with EXIF keywords/location):**
```
Entities: ["Parthenon", "ancient building", "Acropolis", "Athens", "Greece", "UNESCO site"]
Relations:
- (Parthenon, is_a, ancient building)
- (Parthenon, located_at, Acropolis)
- (Acropolis, located_in, Athens)
- (Athens, located_in, Greece)
- (Parthenon, has_designation, UNESCO site)  # from keywords
```

### 4. Temporal Context
**Before:**
```
No temporal information
```

**After (with EXIF date):**
```
Relations:
- (photo, captured_on, 2024-06-15)
- (photo, captured_at_time, 14:32:18)
- (scene, season, summer)  # inferred from date
- (scene, time_of_day, afternoon)  # inferred from time + solar position
```

## Potential Issues

### 1. Information Overload
**Risk:** Too much structured data might confuse the LLM

**Mitigation:**
- Test different formatting approaches (append vs structured sections)
- Limit metadata to most relevant fields
- Use semantic sections with clear headers

### 2. Entity Matching Errors
**Risk:** VLM says "person in blue shirt" but EXIF says "Sarah Chen" - wrong match

**Mitigation:**
- Use face region bboxes from Lightroom to match VLM descriptions
- Add confidence scores to matches
- Allow LLM to make the connection (don't force it)

### 3. Coordinate Interpretation
**Risk:** LLM might not understand numerical coordinates

**Mitigation:**
- Convert to natural language ("upper left", "near the center")
- Use relative descriptions ("larger than", "to the left of")
- Focus on relationships, not absolute positions

### 4. KGGen Input Format Limitations
**Risk:** KGGen might be optimized for natural text, not structured metadata

**Testing needed:**
- Does KGGen handle multi-section text?
- Does it extract relations across sections?
- Does it benefit from structure or is it noise?

## Experiments to Run

### Experiment 1: EXIF Metadata Integration
```python
# Test with/without EXIF data
kg_vanilla = kg_gen.generate(vlm_description)
kg_enriched = kg_gen.generate(vlm_description + exif_metadata)

# Compare:
# - Number of entities
# - Number of relations
# - Entity resolution quality (are people identified?)
# - Location hierarchy (image → place → city → country)
```

### Experiment 2: Spatial Description Formats
```python
# Test different spatial formats
formats = [
    "append_natural",      # "column: upper left, large"
    "append_coordinates",  # "column: bbox [120, 80, 300, 500]"
    "append_relations",    # "column is above fragments"
    "structured_section",  # Separate SPATIAL section
]

for format in formats:
    spatial_desc = format_spatial_data(sam_masks, format)
    kg = kg_gen.generate(vlm_description + spatial_desc)
    evaluate(kg)
```

### Experiment 3: Grounding Models
```python
# Compare different grounding approaches
approaches = [
    "sam_only",           # Segment Anything only
    "sam_matching",       # SAM + heuristic entity matching
    "grounding_dino",     # GroundingDINO for precise grounding
    "vlm_two_pass",       # VLM second pass for locations
]

for approach in approaches:
    grounded = ground_entities(image, vlm_description, approach)
    kg = kg_gen.generate(create_enriched_desc(vlm_description, grounded))
    evaluate(kg)
```

### Experiment 4: Full Enchilada
```python
# All sources combined
full_description = combine_all_sources(
    vlm_text=get_vlm_description(image),
    exif=read_exif(image),
    lr_metadata=read_lightroom_metadata(image),
    sam_masks=segment_anything(image),
    grounding=ground_entities(image, vlm_text)
)

kg_full = kg_gen.generate(full_description)

# Compare to baseline (VLM only)
kg_baseline = kg_gen.generate(get_vlm_description(image))

# Metrics:
# - Entity count (more entities = more complete?)
# - Relation count (more relations = richer KG?)
# - Relation types (spatial, temporal, semantic)
# - Entity resolution (co-reference accuracy)
# - Hierarchy depth (location, object part-of, etc.)
```

## Implementation Priority

### Phase 1: EXIF Integration (Easy)
- ✅ EXIF data is already available
- ✅ Easy to append to VLM description
- ✅ Clear value (people, location, temporal context)
- Test: Does KGGen extract useful relations from metadata?

### Phase 2: Spatial Descriptions (Medium)
- Run SAM on images (already have SAM models)
- Match entities to masks (heuristic matching)
- Convert to natural language spatial descriptions
- Test: Does spatial context improve clustering?

### Phase 3: Grounding Models (Hard)
- Integrate GroundingDINO or similar
- Precisely ground VLM entities to image regions
- Add bounding box metadata to KG
- Test: Does precise grounding enable new applications?

### Phase 4: Full Integration (Research)
- Combine all sources optimally
- Find best format for KGGen input
- Measure impact on KG quality
- Document best practices

## Open Questions

1. **Format sensitivity:** How much does input format affect KGGen output quality?
2. **Information saturation:** Is there a point where more metadata hurts rather than helps?
3. **Cross-modal reasoning:** Can KGGen connect entities across modalities (VLM text + EXIF people + SAM masks)?
4. **Spatial understanding:** Does the LLM benefit from coordinates, or should we stick to natural language?
5. **Lightroom integration:** Can we efficiently extract LR metadata from catalogs? (DNG/XMP sidecars?)

## Next Steps

1. Extract EXIF from one test image (acropolis.dng)
2. Append EXIF to VLM description
3. Run KGGen on enriched description
4. Compare output to vanilla VLM → KGGen
5. Document findings

## References

- Segment Anything: https://github.com/facebookresearch/segment-anything
- GroundingDINO: https://github.com/IDEA-Research/GroundingDINO
- GLIP: https://github.com/microsoft/GLIP
- ExifTool: https://exiftool.org/
- Adobe XMP: https://www.adobe.com/devnet/xmp.html

---

**Status:** Speculative but testable. All components exist, need integration and evaluation.
