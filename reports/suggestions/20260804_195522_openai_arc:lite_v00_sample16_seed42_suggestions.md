# Evaluation Suggestions for Run `20260804_195522_openai_arc:lite_v00_sample16_seed42`

This analysis identifies a significant "semantic drift" and "boundary confusion" issue. The model is struggling to distinguish between distinct but thematically similar items, leading to high-level over-grouping and misclassification of subtle content types.

---

### 1. Error Diagnosis

After analyzing the three failure cases, three distinct error patterns emerge:

#### A. The "Thematic Over-grouping" Error (High Severity)
The model is grouping fragments based on **topic** rather than **structural continuity**. 
*   **Evidence (UM-1956-06-12-3):** The model took fragments from several different articles (Cluster 1 in GT) and merged them into one giant "Article" cluster. It also merged a news report about a "Chief Judge" with a "Kuala Lumpur" dateline, even though they were separate stories.
*   **Root Cause:** The LLM sees a "political" or "news" theme and assumes all related fragments belong to one long article, failing to recognize the breaks between distinct news items.

#### B. The "Class Confusion" Error (Medium Severity)
The model is failing to distinguish between "Article," "Obituary," and "Miscellaneous."
*   **Evidence (UM-1956-05-13-4):** The Ground Truth identifies several "Miscellaneous" clusters (likely short snippets, poems, or non-news text), but the model classifies them as "Article" or "Obituary."
*   **Evidence (UM-1956-02-05-1):** Fragments that were clearly "Miscellaneous" (short snippets) were pulled into "Article" clusters.
*   **Root Cause:** The model lacks a strict definition of what constitutes an "Article" versus "Miscellaneous" (e.g., a short quote or a single-sentence snippet).

#### C. The "Boundary Blindness" Error (Low-Medium Severity)
The model is failing to recognize the "end" of an item.
*   **Evidence (UM-1956-02-05-1):** The model merged a "Miscellaneous" snippet into an "Article" cluster.
*   **Root Cause:** Without spatial or structural cues, the LLM cannot see that a new paragraph or a new column has started; it only sees a stream of text.

---

### 2. Systemic Suggestions

The current prompt is too "thin." It tells the model *what* to return, but not *how to think* about the boundaries.

#### Improved System Prompt (Recommended)
The new prompt introduces **Chain-of-Thought reasoning** (internalized) and strict **Taxonomy Definitions**.

```markdown
You are an expert in historical Jawi Malay and newspaper layout analysis. 

Your task is to reconstruct fragmented OCR text into complete, distinct items (articles, advertisements, obituaries, or miscellaneous snippets).

### CLASSIFICATION TAXONOMY:
1. "article": A complete news story or report. Usually contains a dateline (e.g., "Kuala Lumpur - ..."), a subject, and multiple sentences of continuous narrative.
2. "advertisement": A commercial pitch for a product, service, or person. Often contains prices, contact info, or promotional language.
3. "obituary": A notice regarding a death. Look for keywords like 'meninggal dunia', 'almarhum', or 'takziah'.
4. "miscellaneous": Short, disconnected snippets, single sentences, poems, or fragments that do not form a complete narrative or a formal news report.

### CLUSTERING RULES:
- BOUNDARY DETECTION: An item ends when the topic shifts significantly or a new dateline/header appears. Do NOT merge two different news stories just because they share a similar topic (e.g., two different political reports).
- FRAGMENT INTEGRITY: Only group fragments that logically follow one another to form a coherent thought.
- NO OVER-GROUPING: If a fragment is a single sentence or a short quote, classify it as "miscellaneous" rather than forcing it into an "article".

### OUTPUT FORMAT:
Return ONLY a JSON array of objects. Each object must have:
- "fragment_ids": [list of strings]
- "title": [short descriptive title]
- "class": [one of the 4 classes above]

Example:
[{"fragment_ids": ["r_1", "r_2"], "title": "Example Title", "class": "article"}]
```

#### Improved User Prompt Template
Add a "Contextual Instruction" to force the model to look for structural markers.

```markdown
Consider these text fragments in Jawi Malay. 

### INSTRUCTION:
Analyze the fragments to identify where one item ends and another begins. Pay close attention to datelines (locations/dates) and shifts in subject matter. If a fragment does not seem to belong to a larger narrative, treat it as a separate "miscellaneous" item.

{fragments}

Return ONLY the JSON array.
```

---

### 3. Heuristic Suggestions

If the LLM continues to over-group, you should implement these programmatic safeguards:

**1. Dateline Detection (Pre-processing/Post-processing):**
*   **Heuristic:** Use a Regex or a smaller, faster model to detect "Dateline Patterns" (e.g., `[Location] - [Date]`). 
*   **Rule:** If a fragment contains a dateline, it is a high-probability signal of a **New Item Start**. You can pass these "anchor" fragments to the LLM as explicit "Item Start" markers.

**2. Fragment Length/Density Check (Pre-processing):**
*   **Heuristic:** Calculate the character count of each fragment.
*   **Rule:** If a fragment is extremely short (e.g., < 50 characters) and does not contain a verb or a complete thought, flag it as a candidate for "miscellaneous" before sending it to the LLM. This prevents the LLM from trying to "force" a single sentence into a larger article.

**3. Spatial/Bounding Box Constraint (If available):**
*   **Heuristic:** If your OCR provides $(x, y, w, h)$ coordinates for fragments.
*   **Rule:** Do not allow the LLM to group fragments that have a large vertical gap or are in different columns. If `Fragment A` is in Column 1 and `Fragment B` is in Column 2, they should only be grouped if the text is a continuous sentence split by a column break.

**4. The "One-to-One" Validation (Post-processing):**
*   **Rule:** If the LLM produces a cluster that contains a fragment with a "Title-like" structure (e.g., very short, bold-like text) but the cluster is labeled "Article," run a secondary check to see if that cluster should actually be split.