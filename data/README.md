# Anonymous Supplementary Data

This archive contains anonymized processed data accompanying the submission.

## Contents

- `ouputs/manifest.json`: summary metadata for the processed output shards.
- `ouputs/videos/`: per-video JSON files containing timestamped comments, modality labels, grounded visual/audio evidence, and generated retrieval queries.
- `captions/`: local captioning outputs around timestamped comments.
- `csv/`: channel/category mappings and timestamped-comment metadata.

## Summary

- 1,014 videos
- 19,445 timestamped comment instances
- 17,147 retained visual/audio/mixed moment-query pairs
- 20 video categories

## Anonymization

- Local filesystem paths were removed.
- Author-identifying project/GitHub URLs were removed.
- Email-like strings and user handles in public comments were replaced with placeholders.
- Raw video/audio files and credentials are not included.

The data preserves video IDs, timestamps, categories, generated captions/queries, modality labels, and other non-author-identifying fields needed to inspect the pipeline outputs.
