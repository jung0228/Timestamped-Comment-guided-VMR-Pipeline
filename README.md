# Timestamped-Comment-Guided Video Moment Retrieval

This repository contains anonymized supplementary materials for a double-blind review submission on timestamped-comment-guided video moment retrieval.

## Contents

- `pipeline/`: anonymized pipeline code for video discovery, timestamped-comment processing, modality-specific captioning, comment filtering, modality gating, and query generation.
- `prompts/`: prompt templates used for modality-specific captioning, modality gating, and query generation.
- `data/`: anonymized processed data used to inspect the pipeline outputs, including video IDs, timestamps, categories, modality labels, grounded evidence fields, and generated retrieval queries.
- `requirements.txt`: Python dependencies.

## Data Summary

- 1,014 videos
- 19,445 structured gating results
- 17,147 retained visual/audio/mixed moment-query pairs
- 20 video categories
- 1,115 caption files

## Setup

```bash
pip install -r requirements.txt
```

FFmpeg is required for local video segment extraction.

```bash
# macOS
brew install ffmpeg

# Linux
sudo apt install ffmpeg
```

Set an OpenAI-compatible API key when running the structured gating and query-generation steps.

```bash
export OPENAI_API_KEY='your-api-key'
```

## Pipeline

Run commands from the repository root.

```bash
# Build video list from channel/category metadata
python pipeline/find_popular_videos.py --batch --top 10

# Crawl and merge timestamped comments
python pipeline/crawl_comments.py --workers 5
python pipeline/yt_merge_with_dedup_lang.py

# Optional: download videos for local captioning
python pipeline/download_videos.py --resolution 360

# Generate local visual/audio captions around each timestamp
python pipeline/generate_captions_range.py 1 100

# Paper-version structured modality gating
python pipeline/modality_gating.py data/captions --output data/gated_outputs

# Query generation from grounded gating outputs
python pipeline/query_generator.py data/gated_outputs --output data/query_outputs
```

The prompt templates used by the paper-version pipeline are in `prompts/`.

## Data Files

- `data/ouputs/manifest.json`: summary metadata for processed outputs.
- `data/ouputs/videos/`: per-video JSON files with comments, modality labels, grounded evidence, and generated queries.
- `data/captions/`: local captioning outputs around timestamped comments.
- `data/csv/`: channel/category mappings and timestamped-comment metadata.

## Notes

Raw videos, audio files, API keys, local paths, and author-identifying metadata are not included. Public comment user handles and email-like strings have been anonymized where detected.
