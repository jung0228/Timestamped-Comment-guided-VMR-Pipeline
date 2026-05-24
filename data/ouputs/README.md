# One-call Query Results by Video

This directory contains the per-video shards derived from the merged GPT-5-mini focus-v2 one-call query generation output.

## Files

- `manifest.json`: index file for all video shards.
- `videos/*.json`: one JSON file per YouTube video.

The source merged file is:

```text
data/onecall_query/onecall_full_captions_by_video_gpt-5-mini_focus_v2_merged.json
```

Current split statistics:

- Total records: `19,445`
- Total videos: `1,014`
- Shard type: by `video_id`

## Directory Layout

```text
by_video_gpt-5-mini_focus_v2/
  README.md
  manifest.json
  videos/
    --Cv18I6gxw.json
    -1NFirxhXWE.json
    ...
```

## `manifest.json`

`manifest.json` is the entry point for this sharded dataset.

Top-level fields:

| Field | Type | Description |
| --- | --- | --- |
| `source_file` | string | Path to the merged JSON file used to create these shards. |
| `created_at` | string | UTC timestamp when the split files were generated. |
| `shard_type` | string | Sharding strategy. Currently `by_video`. |
| `num_results` | integer | Total number of result records across all shards. |
| `num_videos` | integer | Number of unique `video_id` shards. |
| `source_metadata` | object | Metadata copied from the merged source file. |
| `source_cost_estimate` | object | Token/cost estimate copied from the merged source file. |
| `videos` | array | List of video shard descriptors. |

Each item in `videos`:

| Field | Type | Description |
| --- | --- | --- |
| `video_id` | string | YouTube video ID. |
| `num_results` | integer | Number of result records in this video shard. |
| `file` | string | Relative path to the video shard JSON file. |

Example:

```json
{
  "video_id": "--Cv18I6gxw",
  "num_results": 20,
  "file": "videos/--Cv18I6gxw.json"
}
```

## Video Shard Files

Each file in `videos/*.json` contains all generated records for one YouTube video.

Top-level fields:

| Field | Type | Description |
| --- | --- | --- |
| `video_id` | string | YouTube video ID for this shard. |
| `num_results` | integer | Number of result records in this shard. |
| `results` | array | Generated query/modality records for timestamped comments from this video. |

Example structure:

```json
{
  "video_id": "--Cv18I6gxw",
  "num_results": 20,
  "results": [
    {
      "video_id": "--Cv18I6gxw",
      "comment_timestamp": 255.0,
      "comment": "...",
      "primary_modality_hint": "visual",
      "final_query": "..."
    }
  ]
}
```

## Result Record Fields

Each item in a shard's `results` array corresponds to one timestamped-comment example and its generated retrieval query.

| Field | Type | Description |
| --- | --- | --- |
| `source_file` | string | Path to the caption/context file used as model input. |
| `video_id` | string | YouTube video ID. |
| `comment_index` | integer | Original comment index within the comment source file. |
| `category` | string | Video/category label used in the dataset. |
| `comment_timestamp` | number | Timestamp in seconds extracted from the comment. |
| `comment` | string | Original timestamped YouTube comment. |
| `comment_focus_summary` | string | Model summary of what the comment is referring to. |
| `comment_focus_keywords` | array[string] | Key phrases extracted from the comment. |
| `integrated_visual` | string | Visual evidence around the timestamp, integrated from available captions/metadata. |
| `integrated_audio` | string | Audio/speech evidence around the timestamp, integrated from captions/ASR-like signals. |
| `grounded_visual_evidence` | string | Visual evidence that explicitly supports the generated query or modality decision. |
| `grounded_audio_evidence` | string | Audio evidence that explicitly supports the generated query or modality decision. |
| `visual_query` | string | Query candidate based primarily on visual evidence. Empty if unsupported or irrelevant. |
| `audio_query` | string | Query candidate based primarily on audio evidence. Empty if unsupported or irrelevant. |
| `primary_modality_hint` | string | Model-predicted primary evidence type, e.g. `visual`, `audio`, `mixed`, or `unrelated`. |
| `audio_subtype` | string | Finer-grained audio type, e.g. speech/music/sound/none, when available. |
| `visual_focus` | string | Finer-grained visual focus, e.g. scene/action/text/object, when available. |
| `audio_focus` | string | Finer-grained audio focus, e.g. spoken content/sound/music/none, when available. |
| `focus_target` | string | Concise target that the retrieval query should localize. |
| `modality_gate_reason` | string | Model rationale for choosing or rejecting visual/audio evidence. |
| `retrieval_keywords` | array[string] | Keywords expected to help retrieve the target moment. |
| `final_query` | string | Final text query generated for retrieving the commented moment. |
| `notes` | string | Additional model notes, caveats, or grounding limitations. |
| `input_tokens_est` | integer | Estimated input tokens used for this generation. |
| `output_tokens_est` | integer | Estimated output tokens produced for this generation. |
| `generated_at` | string | Local timestamp when this record was generated. |

## Modality Labels

The most important field for modality analysis is `primary_modality_hint`.

Common values:

| Value | Meaning |
| --- | --- |
| `visual` | The commented moment is primarily retrievable from visual evidence. |
| `audio` | The commented moment is primarily retrievable from audio/speech evidence. |
| `mixed` | Both visual and audio evidence are important. |
| `unrelated` | The comment does not clearly refer to a retrievable moment in the video. |

## Query Fields

For retrieval/query analysis, the main fields are:

- `visual_query`
- `audio_query`
- `final_query`
- `retrieval_keywords`
- `focus_target`

`final_query` is the primary output used as the generated text query. `visual_query` and `audio_query` are intermediate modality-specific candidates.

## Regenerating the Split Files

Run:

```bash
python3 [LOCAL_PATH] \
  --input [LOCAL_PATH] \
  --output-dir [LOCAL_PATH]
```

## Integrity Check

The split is valid if:

- `manifest.json.num_results == 19445`
- `manifest.json.num_videos == 1014`
- The sum of `num_results` over all video shards equals `19445`
- The number of files in `videos/` equals `1014`

