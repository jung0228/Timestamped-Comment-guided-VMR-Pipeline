# Anonymous Supplementary Materials

This repository contains anonymized supplementary materials for a double-blind review submission on timestamped-comment-guided video moment retrieval.

## Main Contents

- `pipeline/`: anonymized pipeline code for video discovery, timestamped-comment processing, modality-specific captioning, comment filtering, modality gating, and query generation.
- `prompts/`: prompt templates used for modality-specific captioning, modality gating, and query generation.
- `data/`: anonymized processed data used to inspect the pipeline outputs, including video IDs, timestamps, categories, modality labels, grounded evidence fields, and generated retrieval queries.
- `SOFTWARE_README.md`: detailed software usage and pipeline commands.

## Data Summary

- 1,014 videos
- 19,445 timestamped comment instances
- 17,147 retained visual/audio/mixed moment-query pairs
- 20 video categories

## Notes

Raw videos, audio files, API keys, local paths, and author-identifying metadata are not included. Public comment user handles and email-like strings have been anonymized where detected.

For formal submission, the code/prompt files and `data/` folder can also be uploaded as separate `.zip` archives through the review system's supplementary material fields.
