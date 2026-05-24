#!/usr/bin/env python3
"""
Paper-version comment filtering and modality gating.

This script applies the structured prompt from Appendix F to each
timestamped comment and its local segment captions. It assigns one of
visual, audio, mixed, or unrelated and writes the grounded evidence fields
used by the query generation stage.

Expected input formats:
- Caption files with top-level {"comments": [...]} and per-comment "segments".
- Existing result files with top-level {"results": [...]}.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


DEFAULT_MODEL = "gpt-5-mini"
PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompts"
GATING_KEYS = [
    "comment_focus_summary",
    "comment_focus_keywords",
    "integrated_visual",
    "integrated_audio",
    "grounded_visual_evidence",
    "grounded_audio_evidence",
    "primary_modality_hint",
    "audio_subtype",
    "visual_focus",
    "audio_focus",
    "focus_target",
    "modality_gate_reason",
    "notes",
]


def load_prompt() -> str:
    return (PROMPT_DIR / "comment_filtering_modality_gating.txt").read_text(encoding="utf-8")


def strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_json_response(text: str) -> Dict[str, Any]:
    text = strip_code_fence(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise


def output_path_for(input_file: Path, output_dir: Path) -> Path:
    stem = input_file.stem
    for suffix in ["_integrated", "_classified", "_gated"]:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return output_dir / f"{stem}_gated.json"


def get_items(data: Any) -> List[Dict[str, Any]]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if isinstance(data.get("comments"), list):
            return data["comments"]
        if isinstance(data.get("results"), list):
            return data["results"]
    return []


def set_items(container: Any, items: List[Dict[str, Any]]) -> Dict[str, Any]:
    if isinstance(container, dict):
        result = dict(container)
        if "results" in result:
            result["results"] = items
        else:
            result["comments"] = items
        return result
    return {"comments": items}


def extract_segment_captions(item: Dict[str, Any]) -> List[str]:
    segments = item.get("segments") or []
    captions = []
    for segment in segments[:3]:
        caption = segment.get("query") or segment.get("caption") or ""
        if caption:
            captions.append(caption.strip())
    if captions:
        return captions

    visual = item.get("integrated_visual", "")
    audio = item.get("integrated_audio", "")
    if visual or audio:
        return [f"VISUAL: {visual}\nAUDIO: {audio}".strip()]
    return []


def build_user_prompt(item: Dict[str, Any]) -> str:
    captions = extract_segment_captions(item)
    caption_block = "\n\n".join(
        f"Segment {idx + 1}:\n{caption}" for idx, caption in enumerate(captions)
    )
    return f"""Original timestamped comment:
{item.get("comment", "")}

Local segment captions:
{caption_block}
"""


def has_gating_fields(item: Dict[str, Any]) -> bool:
    return bool(item.get("primary_modality_hint") and item.get("comment_focus_summary"))


def normalize_label(label: str) -> str:
    label = (label or "unrelated").strip().lower()
    aliases = {
        "visual-related": "visual",
        "audio-related": "audio",
        "unrel": "unrelated",
        "unknown": "unrelated",
    }
    return aliases.get(label, label if label in {"visual", "audio", "mixed", "unrelated"} else "unrelated")


class ModalityGater:
    def __init__(self, model: str = DEFAULT_MODEL, api_key: Optional[str] = None):
        if OpenAI is None:
            raise SystemExit("Install dependencies first: pip install -r requirements.txt")
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.prompt = load_prompt()

    def complete_json(self, user_prompt: str) -> Dict[str, Any]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )
        return parse_json_response(response.choices[0].message.content or "{}")

    def gate_item(self, item: Dict[str, Any], skip_existing: bool = True) -> Dict[str, Any]:
        if skip_existing and has_gating_fields(item):
            result = dict(item)
            result["primary_modality_hint"] = normalize_label(result.get("primary_modality_hint"))
            return result

        gating = self.complete_json(build_user_prompt(item))
        result = dict(item)
        for key in GATING_KEYS:
            if key in gating:
                result[key] = gating[key]
        result["primary_modality_hint"] = normalize_label(result.get("primary_modality_hint"))
        result["gating_model"] = self.model
        result["gated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        return result


def summarize(items: List[Dict[str, Any]]) -> Dict[str, int]:
    stats = {"total": len(items), "visual": 0, "audio": 0, "mixed": 0, "unrelated": 0}
    for item in items:
        label = normalize_label(item.get("primary_modality_hint"))
        stats[label] = stats.get(label, 0) + 1
    stats["retained"] = stats["visual"] + stats["audio"] + stats["mixed"]
    return stats


def process_file(
    input_file: Path,
    output_file: Path,
    gater: Optional[ModalityGater],
    skip_existing: bool = True,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    data = json.loads(input_file.read_text(encoding="utf-8"))
    items = get_items(data)
    processed = []
    for idx, item in enumerate(items):
        if limit is not None and idx >= limit:
            break
        if gater is None:
            if not has_gating_fields(item):
                raise ValueError(f"{input_file} has items without gating fields; provide an API key/model.")
            processed.append(item)
        else:
            processed.append(gater.gate_item(item, skip_existing=skip_existing))

    output = set_items(data, processed)
    output["modality_gating_stats"] = summarize(processed)
    output["modality_gating_metadata"] = {
        "input_file": input_file.name,
        "model": gater.model if gater else "existing_fields",
        "prompt_file": "prompts/comment_filtering_modality_gating.txt",
        "processed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def iter_files(path: Path, pattern: str) -> Iterable[Path]:
    if path.is_file():
        yield path
    else:
        yield from sorted(Path(p) for p in glob.glob(str(path / pattern)))


def main() -> None:
    parser = argparse.ArgumentParser(description="TCVP paper-version modality gating")
    parser.add_argument("input", nargs="?", default="data/captions", help="Input JSON file or folder")
    parser.add_argument("--output", "-o", default=None, help="Output file or folder")
    parser.add_argument("--pattern", default="captions_*.json", help="Input glob for folder mode")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--normalize-only",
        action="store_true",
        help="Do not call the LLM; only validate/normalize files that already contain gating fields.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else (
        input_path if input_path.is_dir() else input_path.with_name(f"{input_path.stem}_gated.json")
    )
    gater = None if args.normalize_only else ModalityGater(model=args.model, api_key=args.api_key)

    for input_file in iter_files(input_path, args.pattern):
        out_file = output_path_for(input_file, output_path) if output_path.suffix == "" else output_path
        if out_file.exists() and not args.overwrite:
            print(f"skip existing: {out_file}")
            continue
        print(f"gating: {input_file} -> {out_file}")
        output = process_file(input_file, out_file, gater, skip_existing=True, limit=args.limit)
        print(output["modality_gating_stats"])


if __name__ == "__main__":
    main()
