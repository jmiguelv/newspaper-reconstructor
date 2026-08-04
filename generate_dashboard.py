"""Generate a self-contained evaluation dashboard HTML from eval JSON files.

Usage:
    uv run python generate_dashboard.py
    uv run python generate_dashboard.py --eval-dir data/2_evaluations --output dashboard.html
"""

import argparse
import glob
import json
import os

HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <meta name="color-scheme" content="light dark" />
    <title>Evaluation Dashboard — Jawi Newspaper Reconstruction</title>
    <link rel="icon" href="https://fav.farm/📰" />
    <link rel="preconnect" href="https://fonts.googleapis.com" />
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
    <link href="https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,300..700;1,6..72,300..700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet" />
    <style>
        /* ── Tokens ───────────────────────────────────────────── */

        :root {
            --font-serif: "Newsreader", "Georgia", "Times New Roman", serif;
            --font-mono: "JetBrains Mono", ui-monospace, "Cascadia Code", monospace;

            --step--2: clamp(0.6944rem, 0.6597rem + 0.1736cqi, 0.8rem);
            --step--1: clamp(0.8333rem, 0.7754rem + 0.2899cqi, 1rem);
            --step-0:  clamp(1rem, 0.9rem + 0.5cqi, 1.25rem);
            --step-1:  clamp(1.2rem, 1.0469rem + 0.7653cqi, 1.5625rem);
            --step-2:  clamp(1.44rem, 1.2037rem + 1.1813cqi, 1.9531rem);
            --step-3:  clamp(1.728rem, 1.3755rem + 1.7627cqi, 2.4414rem);

            --space-3xs: clamp(0.25rem, 0.2rem + 0.25cqi, 0.375rem);
            --space-2xs: clamp(0.5rem, 0.45rem + 0.25cqi, 0.625rem);
            --space-xs:  clamp(0.75rem, 0.65rem + 0.5cqi, 1rem);
            --space-s:   clamp(1rem, 0.9rem + 0.5cqi, 1.25rem);
            --space-m:   clamp(1.5rem, 1.35rem + 0.75cqi, 1.875rem);
            --space-l:   clamp(2rem, 1.8rem + 1cqi, 2.5rem);
            --space-xl:  clamp(3rem, 2.7rem + 1.5cqi, 3.75rem);

            --measure: 72ch;

            --ink:        #1a1a1a;
            --ink-light:  #555;
            --ink-faint:  #999;
            --surface:    #fff;
            --surface-2:  #f7f7f5;
            --surface-3:  #eeeee9;
            --rule:       #ddd;

            --green:      #2e7d32;
            --green-bg:   #e8f5e9;
            --amber:      #e65100;
            --amber-bg:   #fff3e0;
            --red:        #c62828;
            --red-bg:     #fce4ec;
            --blue:       #1565c0;
            --blue-bg:    #e3f2fd;
            --purple:     #6a1b9a;
            --purple-bg:  #f3e5f5;
            --neutral:    #616161;
            --neutral-bg: #f5f5f5;
        }

        @media (prefers-color-scheme: dark) {
            :root {
                --ink:        #e8e6e3;
                --ink-light:  #aaa;
                --ink-faint:  #777;
                --surface:    #1a1a1a;
                --surface-2:  #242422;
                --surface-3:  #2e2e2b;
                --rule:       #3a3a37;

                --green:      #66bb6a;
                --green-bg:   #1b2e1b;
                --amber:      #ffab40;
                --amber-bg:   #2e2210;
                --red:        #ef5350;
                --red-bg:     #2e1515;
                --blue:       #64b5f6;
                --blue-bg:    #152535;
                --purple:     #ce93d8;
                --purple-bg:  #251530;
                --neutral:    #bdbdbd;
                --neutral-bg: #2a2a2a;
            }
        }

        /* ── Reset ────────────────────────────────────────────── */

        *, *::before, *::after { box-sizing: border-box; margin: 0; }

        /* ── Base ─────────────────────────────────────────────── */

        html {
            font-family: var(--font-serif);
            font-size: 100%;
            line-height: 1.6;
            color: var(--ink);
            background: var(--surface);
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
            text-rendering: optimizeLegibility;
            font-feature-settings: "liga" 1, "calt" 1, "cv01" 1, "cv02" 1;
        }

        body {
            max-width: 100rem;
            margin: 0 auto;
            padding: var(--space-m) var(--space-l);
        }

        /* ── Typography ───────────────────────────────────────── */

        h1 { font-size: var(--step-3); font-weight: 700; letter-spacing: -0.025em; line-height: 1.15; }
        h2 { font-size: var(--step-2); font-weight: 600; letter-spacing: -0.02em; line-height: 1.2; }
        h3 { font-size: var(--step-1); font-weight: 600; letter-spacing: -0.015em; line-height: 1.3; }
        h4 { font-size: var(--step-0); font-weight: 600; line-height: 1.4; }

        p, li, dd, td, th {
            font-size: var(--step--1);
        }

        small, figcaption {
            font-size: var(--step--2);
            color: var(--ink-light);
        }

        code, kbd, samp {
            font-family: var(--font-mono);
            font-size: 0.875em;
            background: var(--surface-2);
            padding: 0.1em 0.35em;
            border-radius: 0.25rem;
        }

        a { color: var(--blue); text-decoration-thickness: 1px; text-underline-offset: 0.15em; }
        a:hover { text-decoration-thickness: 2px; }

        /* ── Layout ───────────────────────────────────────────── */

        header[role="banner"] {
            border-bottom: 1px solid var(--rule);
            padding-bottom: var(--space-m);
            margin-bottom: var(--space-l);
        }

        header[role="banner"] p {
            color: var(--ink-light);
            margin-top: var(--space-3xs);
        }

        section {
            margin-bottom: var(--space-xl);
        }

        section > h2,
        section > h3 {
            margin-bottom: var(--space-s);
        }

        section > h3 {
            margin-top: var(--space-l);
        }

        /* ── Table ────────────────────────────────────────────── */

        .table-wrap {
            overflow-x: auto;
            -webkit-overflow-scrolling: touch;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            font-variant-numeric: tabular-nums;
        }

        th, td {
            padding: var(--space-2xs) var(--space-xs);
            text-align: left;
            border-bottom: 1px solid var(--rule);
            white-space: nowrap;
        }

        th {
            font-size: var(--step--2);
            font-weight: 500;
            color: var(--ink-light);
            text-transform: uppercase;
            letter-spacing: 0.06em;
            cursor: pointer;
            user-select: none;
        }

        th:hover { color: var(--ink); }

        tbody tr {
            cursor: pointer;
            transition: background-color 0.1s ease;
        }

        tbody tr:hover { background-color: var(--surface-2); }
        tbody tr[data-active="true"] {
            background-color: var(--surface-3);
            box-shadow: inset 3px 0 0 var(--ink);
        }

        /* ── Inline page detail row ───────────────────────────── */

        .page-detail-row td {
            padding: 0;
            border-bottom: 2px solid var(--rule);
            cursor: default;
        }

        .page-detail-row:hover { background: none; }

        .page-detail {
            padding: var(--space-m) var(--space-l);
            background: var(--surface-2);
            border-top: 1px solid var(--rule);
        }

        .page-detail .page-metrics-summary {
            display: flex;
            flex-wrap: wrap;
            gap: var(--space-xs) var(--space-m);
            margin-bottom: var(--space-m);
            padding-bottom: var(--space-s);
            border-bottom: 1px solid var(--rule);
        }

        .page-detail .page-metrics-summary span {
            font-size: var(--step--2);
        }

        .page-detail .page-metrics-summary strong {
            font-weight: 600;
        }

        /* ── Metrics ──────────────────────────────────────────── */

        .m-good { color: var(--green); font-weight: 600; }
        .m-ok   { color: var(--amber); font-weight: 600; }
        .m-bad  { color: var(--red);   font-weight: 600; }
        .m-na   { color: var(--ink-faint); }

        /* ── Stat cards (summary bar) ─────────────────────────── */

        .stat-bar {
            display: flex;
            flex-wrap: wrap;
            gap: var(--space-xs);
            padding: var(--space-s) 0;
            border-top: 1px solid var(--rule);
            border-bottom: 1px solid var(--rule);
            margin-bottom: var(--space-m);
        }

        .stat-bar figure {
            flex: 1 1 7rem;
            text-align: center;
        }

        .stat-bar .stat-value {
            font-size: var(--step-1);
            font-weight: 600;
            font-variant-numeric: tabular-nums;
            display: block;
            line-height: 1.2;
        }

        .stat-bar figcaption {
            text-transform: uppercase;
            letter-spacing: 0.06em;
            font-size: var(--step--2);
        }

        /* ── Badges ───────────────────────────────────────────── */

        .badge {
            display: inline-block;
            font-size: var(--step--2);
            font-weight: 600;
            padding: 0.15em 0.6em;
            border-radius: 0.25rem;
            text-transform: capitalize;
        }

        .badge-article       { background: var(--blue-bg);    color: var(--blue); }
        .badge-advertisement  { background: var(--amber-bg);   color: var(--amber); }
        .badge-obituary       { background: var(--purple-bg);  color: var(--purple); }
        .badge-miscellaneous  { background: var(--neutral-bg); color: var(--neutral); }

        /* ── Fragment chips ───────────────────────────────────── */

        .chip {
            display: inline-block;
            font-family: var(--font-mono);
            font-size: var(--step--2);
            padding: 0.1em 0.4em;
            margin: 0.15rem;
            border-radius: 0.2rem;
        }

        .chip-0  { background: var(--blue-bg);    color: var(--blue); }
        .chip-1  { background: var(--green-bg);   color: var(--green); }
        .chip-2  { background: var(--amber-bg);   color: var(--amber); }
        .chip-3  { background: var(--red-bg);     color: var(--red); }
        .chip-4  { background: var(--purple-bg);  color: var(--purple); }
        .chip-5  { background: #e0f7fa; color: #00838f; }
        .chip-6  { background: #fff8e1; color: #f57f17; }
        .chip-7  { background: #e8eaf6; color: #283593; }
        .chip-8  { background: #efebe9; color: #4e342e; }
        .chip-9  { background: #e0f2f1; color: #00695c; }
        .chip-10 { background: #f1f8e9; color: #558b2f; }
        .chip-11 { background: #fce4ec; color: #880e4f; }
        .chip-unmatched { background: var(--neutral-bg); color: var(--ink-faint); }

        @media (prefers-color-scheme: dark) {
            .chip-5  { background: #0d3337; color: #4dd0e1; }
            .chip-6  { background: #332b0e; color: #ffd54f; }
            .chip-7  { background: #1a1d3a; color: #9fa8da; }
            .chip-8  { background: #2c2420; color: #bcaaa4; }
            .chip-9  { background: #0d2e28; color: #80cbc4; }
            .chip-10 { background: #1e2e12; color: #aed581; }
            .chip-11 { background: #2e151e; color: #f48fb1; }
        }

        .chips {
            display: flex;
            flex-wrap: wrap;
            gap: 0.25rem;
            line-height: 2;
        }

        /* ── Comparison grid ──────────────────────────────────── */

        .comparison-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: var(--space-l);
        }

        @media (max-width: 60rem) {
            .comparison-grid { grid-template-columns: 1fr; }
        }

        .comparison-grid > div > h3 { margin-bottom: var(--space-s); }

        /* ── Item cards ───────────────────────────────────────── */

        .item-card {
            background: var(--surface-2);
            border-radius: 0.5rem;
            padding: var(--space-s) var(--space-m);
            margin-bottom: var(--space-xs);
            border-left: 3px solid transparent;
            transition: border-color 0.15s ease, box-shadow 0.15s ease;
        }

        .item-card.match    { border-left-color: var(--green); }
        .item-card.mismatch { border-left-color: var(--red); }

        .item-card-header {
            display: flex;
            align-items: center;
            gap: var(--space-2xs);
            margin-bottom: var(--space-3xs);
        }

        .match-icon   { color: var(--green); font-weight: 700; font-size: var(--step--1); }
        .mismatch-icon { color: var(--red);  font-weight: 700; font-size: var(--step--1); }

        .item-card h4 {
            font-size: var(--step--1);
            font-weight: 500;
            margin-bottom: var(--space-3xs);
        }

        .item-card code {
            font-size: var(--step--2);
        }

        /* ── Topic tags ───────────────────────────────────────── */

        .topic-tag {
            display: inline-block;
            font-size: var(--step--2);
            background: var(--surface-3);
            color: var(--ink-light);
            padding: 0.1em 0.55em;
            border-radius: 1rem;
            margin: 0.15rem;
        }

        /* ── Details / summary (config, errors) ───────────────── */

        details {
            border: 1px solid var(--rule);
            border-radius: 0.5rem;
            margin-bottom: var(--space-m);
        }

        summary {
            font-size: var(--step--1);
            font-weight: 500;
            padding: var(--space-2xs) var(--space-s);
            cursor: pointer;
            list-style: none;
            user-select: none;
        }

        summary::before {
            content: "▸ ";
            display: inline-block;
            transition: transform 0.15s ease;
        }

        details[open] > summary::before {
            content: "▾ ";
        }

        summary::-webkit-details-marker { display: none; }

        details > :not(summary) {
            padding: 0 var(--space-s) var(--space-s);
        }

        /* ── Definition list (config panel) ───────────────────── */

        .config-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(14rem, 1fr));
            gap: var(--space-xs) var(--space-m);
        }

        dt {
            font-size: var(--step--2);
            font-weight: 500;
            color: var(--ink-faint);
            text-transform: uppercase;
            letter-spacing: 0.04em;
            margin-bottom: 0.15em;
        }

        dd {
            margin: 0;
            word-break: break-word;
            font-size: var(--step--1);
        }

        /* ── Error list ───────────────────────────────────────── */

        .error-columns {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: var(--space-m);
        }

        @media (max-width: 50rem) {
            .error-columns { grid-template-columns: 1fr; }
        }

        .error-list {
            max-height: 30vh;
            overflow-y: auto;
        }

        .error-list h4 {
            font-size: var(--step--1);
            margin-bottom: var(--space-3xs);
        }

        .error-list p {
            color: var(--ink-faint);
            font-size: var(--step--2);
            margin-bottom: var(--space-2xs);
        }

        .error-list ul {
            list-style: none;
            padding: 0;
        }

        .error-list li {
            font-size: var(--step--2);
            padding: var(--space-3xs) 0;
            border-bottom: 1px solid var(--rule);
        }

        .error-list li:last-child { border-bottom: none; }

        /* ── Metrics legend ────────────────────────────────────── */

        .metrics-legend {
            margin-bottom: var(--space-l);
        }

        .metrics-legend dl {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(20rem, 1fr));
            gap: var(--space-3xs) var(--space-l);
        }

        .metrics-legend dt {
            font-size: var(--step--1);
            font-weight: 600;
            color: var(--ink);
            text-transform: none;
            letter-spacing: normal;
            margin-top: var(--space-xs);
        }

        .metrics-legend dt code {
            font-size: var(--step--2);
            margin-left: 0.25em;
        }

        .metrics-legend dd {
            font-size: var(--step--2);
            color: var(--ink-light);
            line-height: 1.5;
        }

        .metrics-legend .legend-scale {
            display: flex;
            gap: var(--space-s);
            margin-top: var(--space-xs);
            padding-top: var(--space-xs);
            border-top: 1px solid var(--rule);
            font-size: var(--step--2);
        }

        .metrics-legend .legend-scale span::before {
            content: "";
            display: inline-block;
            width: 0.65em;
            height: 0.65em;
            border-radius: 50%;
            margin-right: 0.35em;
            vertical-align: middle;
        }

        .legend-good::before { background: var(--green) !important; }
        .legend-ok::before   { background: var(--amber) !important; }
        .legend-bad::before  { background: var(--red) !important; }

        /* ── Misc helpers ─────────────────────────────────────── */

        .muted { color: var(--ink-faint); }

        .run-subtitle {
            color: var(--ink-light);
            margin-top: var(--space-3xs);
            margin-bottom: var(--space-m);
        }

        [x-cloak] { display: none !important; }

        /* Smooth section reveals */
        section {
            animation: fadeUp 0.25s ease both;
        }

        @keyframes fadeUp {
            from { opacity: 0; transform: translateY(0.5rem); }
            to   { opacity: 1; transform: translateY(0); }
        }

        /* ── Print ────────────────────────────────────────────── */

        @media print {
            body { padding: 0.5cm; }
            .table-wrap { overflow: visible; }
            details { border: none; }
            details[open] > summary { display: none; }
            .item-card { break-inside: avoid; }
        }
    </style>
</head>
<body>
    <header role="banner">
        <h1>Evaluation Dashboard</h1>
        <p x-data x-text="EVAL_DATA.length + ' evaluation runs — Jawi Newspaper Article Reconstruction'">…</p>
        <p style="margin-top: var(--space-s); max-width: 72ch; font-size: var(--step--1);">
            Reconstructs newspaper articles from OCR text fragments (ALTO XML) by prompting an LLM to group fragments into complete items (articles, advertisements, etc.). Includes evaluation against ground truth article XML using pairwise clustering F1, class accuracy, and coverage metrics.
            <br><br>
            Developed for Jawi (Arabic script) Malay newspapers from the Utusan Melayu 1956 collection.
        </p>
    </header>

    <main x-data="dashboard()" x-cloak>

        <!-- ── Reference Section ────────────────────────────────────── -->
        <section style="margin-bottom: var(--space-xl);">
            <h2>Reference</h2>

            <details class="models-info" style="margin-bottom: var(--space-s);">
                <summary>Models reference</summary>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(20rem, 1fr)); gap: var(--space-m); margin-top: var(--space-s);">

                <article class="item-card" style="margin-bottom: 0;">
                    <div class="item-card-header">
                        <span style="font-size: var(--step--2); padding: 0.1em 0.55em; border-radius: 1rem; background: var(--blue-bg); color: var(--blue); font-weight: 500;">Standard</span>
                        <h4 style="margin: 0; font-family: var(--font-mono);">arc:nexus</h4>
                    </div>
                    <p style="font-size: var(--step--2); color: var(--ink-light); margin-bottom: var(--space-s);">Designed for every day tasks and general use.</p>
                    <dl class="config-grid" style="gap: var(--space-2xs) var(--space-s);">
                        <div><dt>Context</dt><dd>262,144</dd></div>
                        <div><dt>Model</dt><dd>Qwen/Qwen3.6-35B-A3B</dd></div>
                        <div><dt>Size</dt><dd>35B-A3B</dd></div>
                        <div><dt>Quantization</dt><dd>NVFP4</dd></div>
                        <div><dt>Type</dt><dd>MOE</dd></div>
                    </dl>
                </article>

                <article class="item-card" style="margin-bottom: 0;">
                    <div class="item-card-header">
                        <span style="font-size: var(--step--2); padding: 0.1em 0.55em; border-radius: 1rem; background: var(--green-bg); color: var(--green); font-weight: 500;">Fast</span>
                        <h4 style="margin: 0; font-family: var(--font-mono);">arc:lite</h4>
                    </div>
                    <p style="font-size: var(--step--2); color: var(--ink-light); margin-bottom: var(--space-s);">A versatile model, suitable for a wide range of tasks.</p>
                    <dl class="config-grid" style="gap: var(--space-2xs) var(--space-s);">
                        <div><dt>Context</dt><dd>131,072</dd></div>
                        <div><dt>Model</dt><dd>Google/Gemma4:26B-A4B</dd></div>
                        <div><dt>Size</dt><dd>26B-A4B</dd></div>
                        <div><dt>Quantization</dt><dd>NVFP4</dd></div>
                        <div><dt>Type</dt><dd>MOE</dd></div>
                    </dl>
                </article>
            </div>
            </details>

            <!-- ── Prompts Info ────────────────────────────────────── -->
            <details class="prompts-info" style="margin-bottom: var(--space-s);">
                <summary>Prompts reference</summary>
                <div style="padding: var(--space-s); background: var(--surface-2); border-radius: 0.5rem; border: 1px solid var(--rule); margin-top: var(--space-s); display: flex; flex-direction: column; gap: var(--space-s);">
                    
                    <details>
                        <summary><strong style="color: var(--ink);">v00</strong>: Baseline prompt</summary>
                        <pre style="margin-top: var(--space-2xs); padding: var(--space-s); background: var(--bg); border: 1px solid var(--rule); border-radius: 0.25rem; font-size: var(--step--2); overflow-x: auto; white-space: pre-wrap;"># System Prompt

You are an expert in historical Malay.

For each item, return a JSON object with:
- fragment_ids: list of constitutive fragment IDs
- title: short title or topic description
- class: one of "article", "advertisement", "obituary", "miscellaneous"

Return ONLY a JSON array. No other text, no explanation. Example:
[
  {"fragment_ids": ["r_1", "r_2"], "title": "Language congress report", "class": "article"},
  {"fragment_ids": ["r_3"], "title": "Eye drops advertisement", "class": "advertisement"}
]

# User Prompt Template

Consider these text fragments in Malay.

{fragments}

Reconstruct them into full items. Return ONLY a JSON array.</pre>
                    </details>

                    <details>
                        <summary><strong style="color: var(--ink);">v01</strong>: Improved formatting</summary>
                        <pre style="margin-top: var(--space-2xs); padding: var(--space-s); background: var(--bg); border: 1px solid var(--rule); border-radius: 0.25rem; font-size: var(--step--2); overflow-x: auto; white-space: pre-wrap;"># System Prompt

You are given text fragments extracted by OCR from a Malay newspaper page written in Jawi (Arabic script).

Each fragment is a JSON object with the following fields:

- id: fragment identifier
- text: OCR text content
- type: block type (e.g., "text", "header")
- hpos: horizontal position (pixels from left)
- vpos: vertical position (pixels from top)
- width: block width in pixels
- height: block height in pixels

Your task: reconstruct these fragments into complete items (articles, advertisements, etc.).
Some items may consist of a single fragment. Group fragments that belong to the same item together.

For each reconstructed item, provide:

1. fragment_ids: list of constitutive fragment IDs (in reading order)
2. title: a short title or topic description
3. class: one of "article", "advertisement", "obituary", "miscellaneous"

Return ONLY a JSON array. No other text, no explanation. Example:
[
{"fragment_ids": ["r_1", "r_2"], "title": "Union meeting report", "class": "article"},
{"fragment_ids": ["r_3"], "title": "Eye drops advertisement", "class": "advertisement"}
]

# User Prompt Template

Fragments from a Malay newspaper page (Jawi / Arabic script):

{fragments}

Reconstruct these fragments into complete items. Return ONLY a JSON array.</pre>
                    </details>

                    <details>
                        <summary><strong style="color: var(--ink);">v02</strong>: Few-shot examples and additional heuristics</summary>
                        <pre style="margin-top: var(--space-2xs); padding: var(--space-s); background: var(--bg); border: 1px solid var(--rule); border-radius: 0.25rem; font-size: var(--step--2); overflow-x: auto; white-space: pre-wrap;"># System Prompt

You are given text fragments extracted by OCR from a Malay newspaper page written in Jawi (Arabic script).

Each fragment is a JSON object with the following fields:

- id: fragment identifier
- text: OCR text content
- type: block type (e.g., "text", "header")
- hpos: horizontal position (pixels from left)
- vpos: vertical position (pixels from top)
- width: block width in pixels
- height: block height in pixels

Your task: reconstruct these fragments into complete items (articles, advertisements, etc.).
Some items may consist of a single fragment. Group fragments that belong to the same item together.

Use spatial reasoning to help group fragments:

- Fragments that are physically close and vertically aligned likely belong to the same article.
- Articles typically flow downward first, then to the next column to the left.
- Fragments with similar horizontal positions (hpos) and overlapping vertical ranges (vpos + height) are likely in the same column.
- Large blocks may span multiple columns; check if adjacent fragments share a vertical boundary.
- Headers, titles, and standalone blocks may be single-fragment items.

For each reconstructed item, provide:

1. fragment_ids: list of constitutive fragment IDs (in reading order)
2. title: a short title or topic description
3. class: one of "article", "advertisement", "obituary", "miscellaneous"

Return ONLY a JSON array. No other text, no explanation. Example:
[
{"fragment_ids": ["r_1", "r_2"], "title": "Union meeting report", "class": "article"},
{"fragment_ids": ["r_3"], "title": "Eye drops advertisement", "class": "advertisement"}
]

# User Prompt Template

Fragments from a Malay newspaper page (Jawi / Arabic script):

{fragments}

Use the position and size fields (hpos, vpos, width, height) to determine which fragments belong together. Reconstruct these fragments into complete items. Return ONLY a JSON array.</pre>
                    </details>
                </div>
            </details>

            <!-- ── Metrics legend ─────────────────────────────────── -->

            <details class="metrics-legend">
                <summary>Metrics reference</summary>
            <dl>
                <div>
                    <dt>Clustering F1 <code>F1</code></dt>
                    <dd>Harmonic mean of pairwise clustering precision and recall. Measures how well predicted fragment groupings match ground truth by comparing all fragment pairs.</dd>
                </div>
                <div>
                    <dt>B-Cubed F1 <code>B³ F1</code></dt>
                    <dd>Element-level clustering metric. For each fragment, computes precision and recall based on how its cluster overlaps with the ground truth cluster, then averages across all fragments.</dd>
                </div>
                <div>
                    <dt>Precision</dt>
                    <dd>Fraction of predicted same-cluster fragment pairs that are correct—i.e., the two fragments truly belong together in ground truth.</dd>
                </div>
                <div>
                    <dt>Recall</dt>
                    <dd>Fraction of ground truth same-cluster fragment pairs that were correctly predicted as belonging together.</dd>
                </div>
                <div>
                    <dt>Coverage</dt>
                    <dd>Fraction of ground truth fragments that appear in at least one predicted item. A value below 1.0 means some fragments were missed entirely.</dd>
                </div>
                <div>
                    <dt>Class Accuracy <code>Class Acc</code></dt>
                    <dd>Fraction of predicted items whose class label (article, advertisement, obituary, miscellaneous) matches the ground truth item they best align with.</dd>
                </div>
                <div>
                    <dt>True Positives <code>TP</code></dt>
                    <dd>Number of fragment pairs correctly predicted as belonging to the same cluster.</dd>
                </div>
                <div>
                    <dt>False Positives <code>FP</code></dt>
                    <dd>Fragment pairs predicted as same-cluster but actually belonging to different ground truth clusters.</dd>
                </div>
                <div>
                    <dt>False Negatives <code>FN</code></dt>
                    <dd>Fragment pairs in the same ground truth cluster that were not predicted together.</dd>
                </div>
            </dl>
            <div class="legend-scale">
                <span class="legend-good">≥ 0.80 good</span>
                <span class="legend-ok">≥ 0.60 fair</span>
                <span class="legend-bad">&lt; 0.60 poor</span>
            </div>
        </details>
        </section>

        <!-- ── Runs table ─────────────────────────────────────── -->

        <section>
            <h2>Runs</h2>

            <div x-show="bestF1Run" style="margin-bottom: var(--space-m); font-size: var(--step--1); color: var(--ink-light); max-width: 80ch;">
                The highest Clustering F1 score (<strong x-text="fmt(bestF1Run.aggregate.mean_clustering_f1)"></strong>) was achieved by <strong x-text="bestF1Run.config.model" style="color: var(--ink);"></strong> using prompt <code x-text="bestF1Run.config.prompt_name"></code>.
                For Class Accuracy, the top performer is <strong x-text="bestAccRun.config.model" style="color: var(--ink);"></strong> (<strong x-text="fmt(bestAccRun.aggregate.mean_class_accuracy)"></strong>),
                while the best Coverage was seen in <strong x-text="bestCovRun.config.model" style="color: var(--ink);"></strong> (<strong x-text="fmt(bestCovRun.aggregate.mean_coverage)"></strong>).
            </div>
            <div class="table-wrap">
            <table>
                <thead>
                    <tr>
                        <th @click="sort('config.model')">Model <span x-show="sortKey==='config.model'" x-text="sortDir==='asc'?'↑':'↓'"></span></th>
                        <th @click="sort('config.prompt_name')">Prompt <span x-show="sortKey==='config.prompt_name'" x-text="sortDir==='asc'?'↑':'↓'"></span></th>
                        <th @click="sort('config.sample_size')">Sample <span x-show="sortKey==='config.sample_size'" x-text="sortDir==='asc'?'↑':'↓'"></span></th>
                        <th @click="sort('aggregate.total_pages')">Pages <span x-show="sortKey==='aggregate.total_pages'" x-text="sortDir==='asc'?'↑':'↓'"></span></th>
                        <th @click="sort('aggregate.mean_clustering_f1')" title="Pairwise clustering F1: harmonic mean of precision and recall over all fragment pairs">F1 <span x-show="sortKey==='aggregate.mean_clustering_f1'" x-text="sortDir==='asc'?'↑':'↓'"></span></th>
                        <th @click="sort('aggregate.mean_bcubed_f1')" title="B-Cubed F1: element-level clustering metric averaged across all fragments">B³ F1 <span x-show="sortKey==='aggregate.mean_bcubed_f1'" x-text="sortDir==='asc'?'↑':'↓'"></span></th>
                        <th @click="sort('aggregate.mean_clustering_precision')" title="Fraction of predicted same-cluster pairs that are correct">Precision <span x-show="sortKey==='aggregate.mean_clustering_precision'" x-text="sortDir==='asc'?'↑':'↓'"></span></th>
                        <th @click="sort('aggregate.mean_clustering_recall')" title="Fraction of ground truth same-cluster pairs correctly predicted">Recall <span x-show="sortKey==='aggregate.mean_clustering_recall'" x-text="sortDir==='asc'?'↑':'↓'"></span></th>
                        <th @click="sort('aggregate.mean_coverage')" title="Fraction of ground truth fragments appearing in at least one predicted item">Coverage <span x-show="sortKey==='aggregate.mean_coverage'" x-text="sortDir==='asc'?'↑':'↓'"></span></th>
                        <th @click="sort('aggregate.mean_class_accuracy')" title="Fraction of predicted items whose class label matches ground truth">Class Acc <span x-show="sortKey==='aggregate.mean_class_accuracy'" x-text="sortDir==='asc'?'↑':'↓'"></span></th>
                        <th @click="sort('config.execution_time_seconds')" title="Total execution time in seconds">Time <span x-show="sortKey==='config.execution_time_seconds'" x-text="sortDir==='asc'?'↑':'↓'"></span></th>
                    </tr>
                </thead>
                <tbody>
                    <template x-for="run in sortedRuns" :key="run.run_id">
                        <tr @click="toggleRun(run.run_id)" :data-active="expandedRun === run.run_id">
                            <td x-text="run.config.model"></td>
                            <td x-text="run.config.prompt_name"></td>
                            <td x-text="run.config.sample_size || 'all'"></td>
                            <td x-text="run.aggregate.total_pages"></td>
                            <td :class="metricClass(run.aggregate.mean_clustering_f1)" x-text="fmt(run.aggregate.mean_clustering_f1)"></td>
                            <td :class="metricClass(run.aggregate.mean_bcubed_f1)" x-text="fmt(run.aggregate.mean_bcubed_f1)"></td>
                            <td :class="metricClass(run.aggregate.mean_clustering_precision)" x-text="fmt(run.aggregate.mean_clustering_precision)"></td>
                            <td :class="metricClass(run.aggregate.mean_clustering_recall)" x-text="fmt(run.aggregate.mean_clustering_recall)"></td>
                            <td :class="metricClass(run.aggregate.mean_coverage)" x-text="fmt(run.aggregate.mean_coverage)"></td>
                            <td :class="metricClass(run.aggregate.mean_class_accuracy)" x-text="fmt(run.aggregate.mean_class_accuracy)"></td>
                            <td x-text="run.config.execution_time_seconds ? run.config.execution_time_seconds.toFixed(3) + 's' : '—'"></td>
                        </tr>
                    </template>
                </tbody>
            </table>
            </div>
        </section>

        <!-- ── Selected run detail ────────────────────────────── -->

        <template x-if="activeRun">
            <section>
                <h2 x-text="activeRun.config.model + ' / ' + activeRun.config.prompt_name"></h2>
                <p class="run-subtitle">
                    <time x-text="activeRun.timestamp"></time>
                    · sample <span x-text="activeRun.config.sample_size || 'all'"></span>
                </p>

                <!-- Stat bar -->
                <nav class="stat-bar" aria-label="Aggregate metrics">
                    <figure>
                        <span class="stat-value" :class="metricClass(activeRun.aggregate.mean_clustering_f1)" x-text="fmt(activeRun.aggregate.mean_clustering_f1)"></span>
                        <figcaption>Clustering F1</figcaption>
                    </figure>
                    <figure>
                        <span class="stat-value" :class="metricClass(activeRun.aggregate.mean_bcubed_f1)" x-text="fmt(activeRun.aggregate.mean_bcubed_f1)"></span>
                        <figcaption>B³ F1</figcaption>
                    </figure>
                    <figure>
                        <span class="stat-value" :class="metricClass(activeRun.aggregate.mean_clustering_precision)" x-text="fmt(activeRun.aggregate.mean_clustering_precision)"></span>
                        <figcaption>Precision</figcaption>
                    </figure>
                    <figure>
                        <span class="stat-value" :class="metricClass(activeRun.aggregate.mean_clustering_recall)" x-text="fmt(activeRun.aggregate.mean_clustering_recall)"></span>
                        <figcaption>Recall</figcaption>
                    </figure>
                    <figure>
                        <span class="stat-value" :class="metricClass(activeRun.aggregate.mean_coverage)" x-text="fmt(activeRun.aggregate.mean_coverage)"></span>
                        <figcaption>Coverage</figcaption>
                    </figure>
                    <figure>
                        <span class="stat-value" :class="metricClass(activeRun.aggregate.mean_class_accuracy)" x-text="fmt(activeRun.aggregate.mean_class_accuracy)"></span>
                        <figcaption>Class Acc</figcaption>
                    </figure>
                    <figure>
                        <span class="stat-value" x-text="activeRun.aggregate.total_pages"></span>
                        <figcaption>Pages</figcaption>
                    </figure>
                </nav>

                <!-- Config -->
                <details>
                    <summary>Configuration</summary>
                    <dl class="config-grid">
                        <div><dt>Provider</dt><dd x-text="activeRun.config.provider"></dd></div>
                        <div><dt>Model</dt><dd x-text="activeRun.config.model"></dd></div>
                        <div><dt>Prompt</dt><dd x-text="activeRun.config.prompt_name"></dd></div>
                        <div><dt>Sample size</dt><dd x-text="activeRun.config.sample_size || 'all'"></dd></div>
                        <div><dt>Seed</dt><dd x-text="activeRun.config.seed || '—'"></dd></div>
                        <div><dt>Base URL</dt><dd x-text="activeRun.config.base_url || '—'"></dd></div>
                        <div><dt>System prompt</dt><dd x-text="truncate(activeRun.config.system_prompt, 500)"></dd></div>
                        <div><dt>User prompt template</dt><dd x-text="truncate(activeRun.config.user_prompt_template, 500)"></dd></div>
                    </dl>
                </details>

                <!-- Per-page table -->
                <h3>Per-page metrics</h3>
                <div class="table-wrap">
                <table>
                    <thead>
                        <tr>
                            <th @click="sortPage('page_id')">Page <span x-show="pageSortKey === 'page_id'" x-text="pageSortDir === 'asc' ? '▲' : '▼'"></span></th>
                            <th title="Pairwise clustering F1" @click="sortPage('metrics.clustering_f1')">F1 <span x-show="pageSortKey === 'metrics.clustering_f1'" x-text="pageSortDir === 'asc' ? '▲' : '▼'"></span></th>
                            <th title="B-Cubed F1" @click="sortPage('metrics.bcubed_f1')">B³ F1 <span x-show="pageSortKey === 'metrics.bcubed_f1'" x-text="pageSortDir === 'asc' ? '▲' : '▼'"></span></th>
                            <th title="Fraction of ground truth fragments covered" @click="sortPage('metrics.coverage')">Coverage <span x-show="pageSortKey === 'metrics.coverage'" x-text="pageSortDir === 'asc' ? '▲' : '▼'"></span></th>
                            <th title="Fraction of predicted items with correct class label" @click="sortPage('metrics.class_accuracy')">Class Acc <span x-show="pageSortKey === 'metrics.class_accuracy'" x-text="pageSortDir === 'asc' ? '▲' : '▼'"></span></th>
                            <th title="Number of predicted items" @click="sortPage('metrics.num_predicted_items')">Pred <span x-show="pageSortKey === 'metrics.num_predicted_items'" x-text="pageSortDir === 'asc' ? '▲' : '▼'"></span></th>
                            <th title="Number of ground truth items" @click="sortPage('metrics.num_ground_truth_items')">Truth <span x-show="pageSortKey === 'metrics.num_ground_truth_items'" x-text="pageSortDir === 'asc' ? '▲' : '▼'"></span></th>
                            <th title="True positive fragment pairs" @click="sortPage('metrics.tp')">TP <span x-show="pageSortKey === 'metrics.tp'" x-text="pageSortDir === 'asc' ? '▲' : '▼'"></span></th>
                            <th title="False positive fragment pairs" @click="sortPage('metrics.fp')">FP <span x-show="pageSortKey === 'metrics.fp'" x-text="pageSortDir === 'asc' ? '▲' : '▼'"></span></th>
                            <th title="False negative fragment pairs" @click="sortPage('metrics.fn')">FN <span x-show="pageSortKey === 'metrics.fn'" x-text="pageSortDir === 'asc' ? '▲' : '▼'"></span></th>
                        </tr>
                    </thead>
                    <template x-for="page in sortedPages" :key="page.page_id">
                        <tbody>
                            <tr @click="togglePage(page.page_id)" :data-active="expandedPage === page.page_id">
                                <td x-text="page.page_id"></td>
                                <td :class="metricClass(page.metrics.clustering_f1)" x-text="fmt(page.metrics.clustering_f1)"></td>
                                <td :class="metricClass(page.metrics.bcubed_f1)" x-text="fmt(page.metrics.bcubed_f1)"></td>
                                <td :class="metricClass(page.metrics.coverage)" x-text="fmt(page.metrics.coverage)"></td>
                                <td :class="metricClass(page.metrics.class_accuracy)" x-text="fmt(page.metrics.class_accuracy)"></td>
                                <td x-text="page.metrics.num_predicted_items"></td>
                                <td x-text="page.metrics.num_ground_truth_items"></td>
                                <td x-text="page.metrics.tp"></td>
                                <td x-text="page.metrics.fp"></td>
                                <td x-text="page.metrics.fn"></td>
                            </tr>
                            <tr x-show="expandedPage === page.page_id" class="page-detail-row">
                                <td colspan="10">
                                    <div class="page-detail">
                                        <div class="page-metrics-summary">
                                            <span>F1 <strong :class="metricClass(page.metrics.clustering_f1)" x-text="fmt(page.metrics.clustering_f1)"></strong></span>
                                            <span>B³ F1 <strong :class="metricClass(page.metrics.bcubed_f1)" x-text="fmt(page.metrics.bcubed_f1)"></strong></span>
                                            <span>Precision <strong :class="metricClass(page.metrics.clustering_precision)" x-text="fmt(page.metrics.clustering_precision)"></strong></span>
                                            <span>Recall <strong :class="metricClass(page.metrics.clustering_recall)" x-text="fmt(page.metrics.clustering_recall)"></strong></span>
                                            <span>Coverage <strong :class="metricClass(page.metrics.coverage)" x-text="fmt(page.metrics.coverage)"></strong></span>
                                            <span>Class Acc <strong :class="metricClass(page.metrics.class_accuracy)" x-text="fmt(page.metrics.class_accuracy)"></strong></span>
                                            <span>Fragments <strong x-text="page.metrics.num_fragments"></strong></span>
                                        </div>

                                        <div class="comparison-grid">
                                            <div style="grid-column: 1 / -1; display: grid; grid-template-columns: 1fr 1fr; gap: var(--space-l);">
                                                <h3 style="margin: 0;">Predicted <small x-text="'(' + page.predicted_items.length + ')'"></small></h3>
                                                <h3 style="margin: 0;">Ground truth <small x-text="'(' + (page.ground_truth_items || []).length + ')'"></small></h3>
                                            </div>
                                            <template x-for="(pair, idx) in getAlignedItems(page)" :key="idx">
                                                <div style="grid-column: 1 / -1; display: grid; grid-template-columns: 1fr 1fr; gap: var(--space-l); margin-bottom: var(--space-xs);">
                                                    <!-- Predicted side -->
                                                    <template x-if="pair.pred">
                                                        <article class="item-card" :class="isPredMatch(page, pair.pred) ? 'match' : 'mismatch'" style="margin-bottom: 0;">
                                                            <div class="item-card-header">
                                                                <span class="badge" :class="'badge-' + pair.pred.class" x-text="pair.pred.class"></span>
                                                                <span :class="isPredMatch(page, pair.pred) ? 'match-icon' : 'mismatch-icon'" x-text="isPredMatch(page, pair.pred) ? '✓' : '✗'"></span>
                                                            </div>
                                                            <h4 x-text="pair.pred.title"></h4>
                                                            <div class="chips">
                                                                <template x-for="fid in pair.pred.fragment_ids" :key="fid">
                                                                    <span class="chip" :class="getChipClass(page, fid)" x-text="fid"></span>
                                                                </template>
                                                            </div>
                                                        </article>
                                                    </template>
                                                    <template x-if="!pair.pred">
                                                        <div class="item-card" style="border: 1px dashed var(--rule); background: transparent; opacity: 0.5;"></div>
                                                    </template>

                                                    <!-- Ground Truth side -->
                                                    <template x-if="pair.truth">
                                                        <article class="item-card" :class="isTruthMatch(page, pair.truth) ? 'match' : 'mismatch'" style="margin-bottom: 0;">
                                                            <div class="item-card-header">
                                                                <span class="badge" :class="'badge-' + pair.truth.class" x-text="pair.truth.class"></span>
                                                                <small :class="isTruthMatch(page, pair.truth) ? 'match-icon' : 'mismatch-icon'" x-text="isTruthMatch(page, pair.truth) ? 'recovered' : 'missed'"></small>
                                                            </div>
                                                            <code x-text="pair.truth.uuid"></code>
                                                            <div class="chips">
                                                                <template x-for="fid in pair.truth.fragment_ids" :key="fid">
                                                                    <span class="chip" :class="getChipClass(page, fid)" x-text="fid"></span>
                                                                </template>
                                                            </div>
                                                            <template x-if="pair.truth.topics && pair.truth.topics.length">
                                                                <div>
                                                                    <template x-for="topic in pair.truth.topics" :key="topic">
                                                                        <span class="topic-tag" x-text="topic"></span>
                                                                    </template>
                                                                </div>
                                                            </template>
                                                        </article>
                                                    </template>
                                                    <template x-if="!pair.truth">
                                                        <div class="item-card" style="border: 1px dashed var(--rule); background: transparent; opacity: 0.5;"></div>
                                                    </template>
                                                </div>
                                            </template>
                                        </div>

                                        <template x-if="page.metrics.false_positives.length || page.metrics.false_negatives.length">
                                            <details>
                                                <summary>Errors <small x-text="'(' + (page.metrics.false_positives.length + page.metrics.false_negatives.length) + ')'"></small></summary>
                                                <div class="error-columns">
                                                    <template x-if="page.metrics.false_positives.length">
                                                        <div class="error-list">
                                                            <h4>False positives <small x-text="'(' + page.metrics.false_positives.length + ')'"></small></h4>
                                                            <p>Fragment pairs incorrectly grouped together</p>
                                                            <ul>
                                                                <template x-for="(pair, i) in page.metrics.false_positives" :key="'fp'+i">
                                                                    <li><code x-text="pair[0]"></code> ↔ <code x-text="pair[1]"></code></li>
                                                                </template>
                                                            </ul>
                                                        </div>
                                                    </template>
                                                    <template x-if="page.metrics.false_negatives.length">
                                                        <div class="error-list">
                                                            <h4>False negatives <small x-text="'(' + page.metrics.false_negatives.length + ')'"></small></h4>
                                                            <p>Fragment pairs that should have been grouped together</p>
                                                            <ul>
                                                                <template x-for="(pair, i) in page.metrics.false_negatives" :key="'fn'+i">
                                                                    <li><code x-text="pair[0]"></code> ↔ <code x-text="pair[1]"></code></li>
                                                                </template>
                                                            </ul>
                                                        </div>
                                                    </template>
                                                </div>
                                            </details>
                                        </template>
                                    </div>
                                </td>
                            </tr>
                        </tbody>
                    </template>
                </table>
                </div>
            </section>
        </template>

        <p x-show="runs.length === 0" class="muted">No evaluation files found. Run evaluations first.</p>
    </main>

    <script defer src="https://cdn.jsdelivr.net/npm/alpinejs@3.x.x/dist/cdn.min.js"></script>
    <script>
        const EVAL_DATA = __DATA_PLACEHOLDER__;

        function dashboard() {
            return {
                runs: EVAL_DATA,
                sortKey: 'aggregate.mean_clustering_f1',
                sortDir: 'desc',
                pageSortKey: 'page_id',
                pageSortDir: 'asc',
                expandedRun: null,
                expandedPage: null,

                get sortedRuns() {
                    return [...this.runs].sort((a, b) => {
                        const aVal = this.getNested(a, this.sortKey);
                        const bVal = this.getNested(b, this.sortKey);
                        if (typeof aVal === 'string') {
                            return this.sortDir === 'asc' ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
                        }
                        const aNum = aVal ?? 0;
                        const bNum = bVal ?? 0;
                        return this.sortDir === 'asc' ? aNum - bNum : bNum - aNum;
                    });
                },

                get bestF1Run() {
                    if (!this.runs || this.runs.length === 0) return null;
                    return [...this.runs].sort((a, b) => {
                        const aVal = a.aggregate?.mean_clustering_f1 || 0;
                        const bVal = b.aggregate?.mean_clustering_f1 || 0;
                        return bVal - aVal;
                    })[0];
                },
                get bestAccRun() {
                    if (!this.runs || this.runs.length === 0) return null;
                    return [...this.runs].sort((a, b) => {
                        const aVal = a.aggregate?.mean_class_accuracy || 0;
                        const bVal = b.aggregate?.mean_class_accuracy || 0;
                        return bVal - aVal;
                    })[0];
                },
                get bestCovRun() {
                    if (!this.runs || this.runs.length === 0) return null;
                    return [...this.runs].sort((a, b) => {
                        const aVal = a.aggregate?.mean_coverage || 0;
                        const bVal = b.aggregate?.mean_coverage || 0;
                        return bVal - aVal;
                    })[0];
                },

                get sortedPages() {
                    if (!this.activeRun) return [];
                    return [...this.activeRun.pages].sort((a, b) => {
                        const aVal = this.getNested(a, this.pageSortKey);
                        const bVal = this.getNested(b, this.pageSortKey);
                        if (typeof aVal === 'string') {
                            return this.pageSortDir === 'asc' ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
                        }
                        const aNum = aVal ?? 0;
                        const bNum = bVal ?? 0;
                        return this.pageSortDir === 'asc' ? aNum - bNum : bNum - aNum;
                    });
                },

                get activeRun() {
                    return this.runs.find(r => r.run_id === this.expandedRun) || null;
                },

                get activePage() {
                    if (!this.activeRun) return null;
                    return this.activeRun.pages.find(p => p.page_id === this.expandedPage) || null;
                },

                getNested(obj, path) {
                    return path.split('.').reduce((o, k) => (o == null ? undefined : o[k]), obj);
                },

                sort(key) {
                    if (this.sortKey === key) {
                        this.sortDir = this.sortDir === 'asc' ? 'desc' : 'asc';
                    } else {
                        this.sortKey = key;
                        this.sortDir = 'desc';
                    }
                },

                sortPage(key) {
                    if (this.pageSortKey === key) {
                        this.pageSortDir = this.pageSortDir === 'asc' ? 'desc' : 'asc';
                    } else {
                        this.pageSortKey = key;
                        this.pageSortDir = 'desc';
                    }
                },

                toggleRun(runId) {
                    this.expandedRun = this.expandedRun === runId ? null : runId;
                    this.expandedPage = null;
                },

                togglePage(pageId) {
                    this.expandedPage = this.expandedPage === pageId ? null : pageId;
                },

                metricClass(val) {
                    if (val == null) return 'm-na';
                    if (val >= 0.8) return 'm-good';
                    if (val >= 0.6) return 'm-ok';
                    return 'm-bad';
                },

                fmt(val, decimals) {
                    if (val == null) return '—';
                    let d = decimals !== undefined ? decimals : 3;
                    return val.toLocaleString(undefined, { minimumFractionDigits: d, maximumFractionDigits: d });
                },

                getChipClass(page, fragmentId) {
                    const idx = page.predicted_items.findIndex(item =>
                        item.fragment_ids.includes(fragmentId)
                    );
                    return idx !== -1 ? 'chip chip-' + (idx % 12) : 'chip chip-unmatched';
                },

                isPredMatch(page, item) {
                    const truth = page.ground_truth_items || [];
                    return truth.some(t =>
                        item.fragment_ids.every(fid => t.fragment_ids.includes(fid))
                    );
                },

                isTruthMatch(page, item) {
                    const pred = page.predicted_items || [];
                    return pred.some(p =>
                        item.fragment_ids.every(fid => p.fragment_ids.includes(fid))
                    );
                },


                getAlignedItems(page) {
                    if (!page) return [];
                    const preds = page.predicted_items || [];
                    const truths = page.ground_truth_items || [];

                    let aligned = [];
                    let usedTruths = new Set();

                    for (const pred of preds) {
                        let bestTruth = null;
                        let bestScore = -1;
                        let bestTruthIdx = -1;

                        for (let i = 0; i < truths.length; i++) {
                            const truth = truths[i];
                            const intersection = pred.fragment_ids.filter(id => truth.fragment_ids.includes(id)).length;
                            if (intersection > 0) {
                                const union = new Set([...pred.fragment_ids, ...truth.fragment_ids]).size;
                                const score = intersection / union;
                                if (score > bestScore) {
                                    bestScore = score;
                                    bestTruth = truth;
                                    bestTruthIdx = i;
                                }
                            }
                        }

                        if (bestTruth) {
                            aligned.push({ pred: pred, truth: bestTruth });
                            usedTruths.add(bestTruthIdx);
                        } else {
                            aligned.push({ pred: pred, truth: null });
                        }
                    }

                    for (let i = 0; i < truths.length; i++) {
                        if (!usedTruths.has(i)) {
                            aligned.push({ pred: null, truth: truths[i] });
                        }
                    }

                    return aligned;
                },
                truncate(str, len) {
                    if (!str) return '';
                    return str.length > len ? str.slice(0, len) + '\u2026' : str;
                },
            };
        }
    </script>
</body>
</html>"""


def load_evaluations(eval_dir: str = "data/2_evaluations") -> list[dict]:
    """Load all evaluation JSON files from the given directory."""
    runs = []
    for path in sorted(glob.glob(os.path.join(eval_dir, "*.json"))):
        with open(path, encoding="utf-8") as f:
            runs.append(json.load(f))
    return runs


def generate_html(runs: list[dict]) -> str:
    """Generate a self-contained HTML dashboard with embedded eval data."""
    embedded = json.dumps(runs, ensure_ascii=False)
    return HTML_TEMPLATE.replace("__DATA_PLACEHOLDER__", embedded)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate an evaluation dashboard HTML from eval JSON files."
    )
    parser.add_argument(
        "--eval-dir",
        default="data/2_evaluations",
        help="Directory containing evaluation JSON files (default: data/2_evaluations)",
    )
    parser.add_argument(
        "--output",
        default="data/2_evaluations/dashboard.html",
        help="Output HTML file path (default: data/2_evaluations/dashboard.html)",
    )
    args = parser.parse_args()

    runs = load_evaluations(args.eval_dir)
    if not runs:
        print(f"No evaluation files found in {args.eval_dir}")

    html = generate_html(runs)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Dashboard written to {args.output} ({len(runs)} runs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
