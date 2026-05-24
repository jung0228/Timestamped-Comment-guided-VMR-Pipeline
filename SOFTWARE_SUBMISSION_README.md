# Anonymous Supplementary Software

This archive contains anonymized code for the TCVP data construction pipeline.

Start with `SOFTWARE_README.md` for the end-to-end pipeline commands. The main paper-version scripts are `pipeline/modality_gating.py` and `pipeline/query_generator.py`.

## Contents

- `pipeline/`: scripts for video discovery, comment crawling, caption generation, modality gating, and query generation.
- `pipeline/modality_gating.py`: paper-version structured LLM pipeline for comment filtering and visual/audio/mixed/unrelated modality gating.
- `pipeline/query_generator.py`: paper-version query generation from grounded gating outputs.
- `prompts/`: exact prompt templates corresponding to the generation prompts described in the paper appendix.
- `requirements.txt`: Python package dependencies.
- `SOFTWARE_README.md`: usage instructions with author-identifying links removed.

## Notes

API keys, local machine paths, GitHub/project URLs, and author-identifying metadata have been removed or replaced with placeholders. Large raw videos and private credentials are not included.
