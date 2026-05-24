# Video Moment Retrieval
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A pipeline that collects and analyzes YouTube video comments, combines them with video segment captions, and automatically generates search queries for **moment retrieval**.
![TCVP Pipeline](figures/TCVP%20pipeline.png)

🔗 Project Page: [ANONYMIZED_PROJECT_PAGE]

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Data Flow](#data-flow)
- [License](#license)

---

## Features

- **Popular video discovery**: Build video lists by channel/category using `yt-dlp`
- **Comment crawling**: Per-video comment collection with parallel workers
- **Comment preprocessing**: Timestamp extraction, deduplication, language filtering
- **Video caption generation**: Segment-level visual/audio descriptions (FFmpeg + AI)
- **Modality gating**: Structured LLM-based filtering into visual/audio/mixed/unrelated
- **Moment query generation**: Structured query fields from grounded gating outputs

---

## Installation

### Requirements

- **Python** 3.8+
- **yt-dlp**: YouTube metadata and download
- **FFmpeg**: Video segment extraction
- **API keys** (optional): OpenAI or Letsur AI Gateway

```bash
git clone [ANONYMIZED_REPOSITORY]
cd TCVP
pip install -r requirements.txt
# FFmpeg: brew install ffmpeg (macOS) / sudo apt install ffmpeg (Linux)
```

### Environment variables

```bash
# OpenAI
export OPENAI_API_KEY='your-openai-api-key'

# Or Letsur AI Gateway
export LETSUR_API_KEY='your-letsur-api-key'
export LETSUR_MODEL='gpt-4.1'  # optional
```

---

## Usage

Run all commands **from the project root**. Pipeline order: **1 → 2 → 3 → 4(optional) → 5 → 6 → 7 → 8**

### Step 1: Build popular video list

```bash
# Single channel
python pipeline/find_popular_videos.py [USER] --top 10 --save --channel-name "Channel Name" --category "category"

# Batch (uses data/csv/channel_categories.csv)
python pipeline/find_popular_videos.py --batch --top 10
```

### Step 2: Crawl comments

```bash
python pipeline/crawl_comments.py --workers 5
```

### Step 3: Merge & language filter

```bash
python pipeline/yt_merge_with_dedup_lang.py
```

Output: `data/csv/merged_filtered_comments_with_dedup_lang.csv`

### Step 4: Download videos (optional)

```bash
python pipeline/download_videos.py --resolution 360
```

### Step 5: Generate captions

```bash
python pipeline/generate_captions_range.py 1 100
```

Output: `data/captions/captions_<video_id>.json`

### Step 6: Local caption window

`generate_captions_range.py` stores three consecutive 6-second segment captions
around each timestamp. The paper-version pipeline consumes these raw local
captions directly, so the old segment-integration step is no longer needed.

### Step 7: Modality gating

Paper-version structured LLM gating:

```bash
python pipeline/modality_gating.py data/captions --output data/gated_outputs
```

The full prompt templates are provided under `prompts/`:

- `modality_specific_captioning.txt`
- `comment_filtering_modality_gating.txt`
- `query_generation_from_gating_output.txt`

Output: `*_gated.json`, with fields such as `comment_focus_summary`,
`grounded_visual_evidence`, `grounded_audio_evidence`,
`primary_modality_hint`, `visual_focus`, and `audio_focus`.

### Step 8: Moment query generation

```bash
python pipeline/query_generator.py data/gated_outputs --output data/query_outputs
```

Output: `*_moment_queries.json`, with `visual_query`, `audio_query`, and
`final_query`. If these query fields already exist, the script preserves them
unless `--regenerate` is passed.

---

## Project Structure

```
TCVP/
├── pipeline/                         # Pipeline scripts
│   ├── find_popular_videos.py        # 1. Popular video discovery
│   ├── crawl_comments.py             # 2. Comment crawling
│   ├── yt_merge_with_dedup_lang.py   # 3. Comment merge & language filter
│   ├── download_videos.py            # 4. Video download (optional)
│   ├── generate_captions_range.py    # 5. Caption generation
│   ├── OptimizedMomentQueryGenerator.py  # Core module for caption generation
│   ├── modality_gating.py            # 7. Modality gating
│   └── query_generator.py            # 8. Query generation
├── data/
│   ├── csv/                          # Channel/video mapping and merged comments
│   ├── comments/                     # Per-video crawled comments
│   └── captions/                     # Local caption, gated, and query JSONs
├── figures/
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Data Flow

```
data/csv/channel_categories.csv
        ↓
pipeline/find_popular_videos.py     →  data/csv/video_id_mapping.csv
        ↓
pipeline/crawl_comments.py          →  data/comments/<video_id>_comments.csv
        ↓
pipeline/yt_merge_with_dedup_lang.py  →  data/csv/merged_filtered_comments_with_dedup_lang.csv
        ↓
pipeline/download_videos.py (optional)
        ↓
pipeline/generate_captions_range.py →  data/captions/captions_<video_id>.json
        ↓
pipeline/modality_gating.py         →  *_gated.json
        ↓
pipeline/query_generator.py         →  *_moment_queries.json
```

---

## License

This project is distributed under the [MIT License](LICENSE).
