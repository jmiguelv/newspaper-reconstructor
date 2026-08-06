# Evaluation Suggestions for Run `20260804_195522_openai_arc:lite_v00_sample16_seed42`



Based on the error analysis and current prompts, here is a structured diagnosis and actionable recommendations to improve clustering accuracy.

---

## 1. Error Diagnosis

The model exhibits three consistent failure patterns across the worst-performing pages:

| Pattern | Evidence | Root Cause |
|---------|----------|------------|
| **Over-grouping (Merging distinct items)** | Page 1: GT Clusters 1–4 (4 separate articles) merged into 1 predicted cluster. Page 3: GT Clusters 1 & 2 merged. | The model treats the input as a continuous text stream. Without explicit boundary cues, it defaults to semantic similarity or length, ignoring datelines, topic shifts, or layout breaks. |
| **Over-splitting (Breaking continuous items)** | Page 2: GT Cluster 5 (a long poem/narrative) split into 4 predicted clusters. Page 3: GT Cluster 4 split into 2. | The model uses short coherence windows or sentence boundaries to split. Long narrative/poetic text with minor thematic shifts or line breaks triggers false splits. |
| **Class Ambiguity & Single-Fragment Handling** | Short fragments miscategorized as `miscellaneous` or `article`. No explicit guidance on single-fragment items. | The prompt lacks clear class definitions and doesn't instruct the model to treat isolated fragments as valid items. `miscellaneous` becomes a dumping ground for ambiguous text. |

**Core Limitation:** The model receives *only raw text fragments* without layout context (columns, spacing, bounding boxes). Newspaper reconstruction is inherently spatial, and the current prompt asks the LLM to solve a layout problem using text-only signals.

---

## 2. Systemic Suggestions (Prompt Engineering)

### 🔹 Revised System Prompt
```text
You are an expert in historical Malay newspaper layout and text reconstruction.

Your task is to group OCR text fragments into complete, independent items (articles, advertisements, obituaries, or miscellaneous notices). 

OUTPUT FORMAT:
Return ONLY a valid JSON array. Each object must contain:
- fragment_ids: list of fragment IDs belonging to this item
- title: concise topic/title (max 10 words)
- class: one of "article", "advertisement", "obituary", "miscellaneous"

GROUPING RULES:
1. CONTINUITY FIRST: Group fragments that form a continuous sentence, paragraph, or narrative flow. Preserve line breaks and poetic structure.
2. BOUNDARY DETECTION: Split items when you detect:
   - A new dateline (e.g., "کوالا لمفور ١١ جون", "لندن ٥ مارچ")
   - A clear topic shift or new headline
   - Structural markers like "حاشيه", "تنتي", "سبنتوهن", or abrupt tone changes
3. CONSERVATIVE MERGING: If two fragments share a topic but lack textual continuity, keep them separate. It is better to over-split than to merge distinct items.
4. ADVERTISEMENT DETECTION: Group fragments containing prices, contact info, "jual", "بلي", "حديہ", "iklan", "for sale", or commercial phrasing.
5. SINGLE FRAGMENTS: A single fragment can be a valid item. Do not force grouping.
6. CLASS DEFINITIONS:
   - article: news reports, editorials, reports, speeches
   - obituary: death notices, funeral arrangements, mourning announcements
   - advertisement: commercial notices, product listings, service offers
   - miscellaneous: poems, short notices, fragmented text, announcements that don't fit other classes

Return ONLY the JSON array. No markdown, no explanation.
```

### 🔹 Revised User Prompt Template
```text
Consider these OCR text fragments from a historical Malay newspaper. The fragments are listed in their original reading order.

{fragments}

Reconstruct them into complete, independent items following the grouping rules. Return ONLY a JSON array.
```

### 🔑 Key Improvements
- **Explicit boundary triggers:** Datelines, structural markers, and topic shifts are named.
- **Conservative merging directive:** Prevents the Page 1 over-grouping error.
- **Class definitions:** Reduces miscategorization of short/ambiguous fragments.
- **Single-fragment validation:** Explicitly allows isolated items.
- **Reading order hint:** Reinforces that fragment order matters for continuity.

> 💡 *Optional but highly effective:* Add 2–3 few-shot examples in the user prompt showing correct handling of a long poem (kept together), two adjacent articles (kept separate), and an ad (correctly classified). LLMs perform significantly better on clustering tasks with examples.

---

## 3. Heuristic Suggestions (Programmatic)

Prompt engineering alone cannot fully compensate for missing layout context. Implement these pipeline steps:

### 🛠 Pre-processing
1. **Sort by Bounding Box Coordinates:** 
   - Sort fragments top-to-bottom, then left-to-right. This preserves column flow and prevents the LLM from receiving out-of-order text.
   - If X-coordinates are available, **segment by column** before sending to the LLM. Process each column independently.

2. **Ad/Notice Keyword Pre-filter:**
   - Run a lightweight regex/keyword filter in Jawi/Malay to flag fragments likely belonging to ads: `حديہ|بلي|jual|iklan|harga|rm|contact|telefon|alamat|for sale|advert`.
   - If >60% of a cluster's fragments match ad keywords, force `class: "advertisement"` post-hoc.

3. **Dateline Detection:**
   - Flag fragments matching patterns like `کوالا لمفور \d+ \w+`, `لندن \d+`, `جوهور بهرو \d+`. Use these as hard split points in post-processing.

### 🛠 Post-processing
1. **Embedding-Based Similarity Check:**
   - Compute sentence/fragment embeddings (e.g., `paraphrase-multilingual-MiniLM`).
   - If two predicted clusters have cosine similarity >0.85 and are adjacent in reading order, merge them.
   - If a cluster contains fragments with similarity <0.4, split it.

2. **Length & Density Heuristics:**
   - Clusters with <3 fragments and high keyword density → likely ads or notices.
   - Clusters with >10 fragments but multiple datelines → split at dateline boundaries.

3. **Fallback Classifier:**
   - Train a lightweight rule-based or small ML classifier on `title` + `fragment_count` + `keyword_density` to correct `miscellaneous` vs `article` misclassifications.

### 🔄 Hybrid Architecture Recommendation
```
OCR Fragments → Sort by BBox → Column Segmentation → LLM Clustering (per column) → Post-processing (embeddings + rules) → Final JSON
```
This reduces the LLM's context window, eliminates cross-column merging errors, and aligns with how newspapers are actually structured.

---

## 4. Implementation Notes

- **Test Incrementally:** Start with bounding-box sorting + conservative merging prompt. Expect F1 to jump from ~0.46 to ~0.65.
- **Monitor Single-Fragment Clusters:** Add a pipeline step to validate that single-fragment items have a plausible `title` and `class`.
- **Jawi-Specific Tuning:** If using embeddings, ensure they support Arabic-script Malay (e.g., `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` or fine-tune on Jawi corpora).
- **Avoid Over-Engineering the Prompt:** Keep the JSON schema strict. All guidance should live in the instructions, not the output structure.

These changes address the root causes: lack of layout awareness, ambiguous boundary detection, and insufficient class guidance. Implementing the pre-sorting + conservative prompt + post-processing heuristics will significantly improve clustering F1 across historical Jawi newspaper pages.