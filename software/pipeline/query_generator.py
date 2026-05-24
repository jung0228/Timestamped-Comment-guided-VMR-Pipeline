#!/usr/bin/env python3
"""
Paper-version query generation from modality-gating outputs.

Input items should contain the fields produced by modality_gating.py:
comment_focus_summary, grounded_visual_evidence, grounded_audio_evidence,
primary_modality_hint, retrieval_keywords, and related evidence fields.

If visual_query/audio_query/final_query are already present, the script can
normalize and summarize them without calling the API. Use --regenerate to
force query generation with the Appendix F prompt.
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
QUERY_KEYS = [
    "visual_query",
    "audio_query",
    "final_query",
    "used_visual_support",
    "used_audio_support",
    "query_keywords",
    "query_reason",
]


def load_prompt() -> str:
    return (PROMPT_DIR / "query_generation_from_gating_output.txt").read_text(encoding="utf-8")


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


def normalize_label(label: str) -> str:
    label = (label or "unrelated").strip().lower()
    aliases = {"visual-related": "visual", "audio-related": "audio", "unrel": "unrelated"}
    return aliases.get(label, label if label in {"visual", "audio", "mixed", "unrelated"} else "unrelated")


def get_items(data: Any) -> List[Dict[str, Any]]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if isinstance(data.get("comments"), list):
            return data["comments"]
        if isinstance(data.get("results"), list):
            return data["results"]
        if isinstance(data.get("generated_queries"), list):
            return data["generated_queries"]
    return []


def set_items(container: Any, items: List[Dict[str, Any]]) -> Dict[str, Any]:
    if isinstance(container, dict):
        result = dict(container)
        if "results" in result:
            result["results"] = items
        elif "generated_queries" in result:
            result["generated_queries"] = items
        else:
            result["comments"] = items
        return result
    return {"comments": items}


def has_query_fields(item: Dict[str, Any]) -> bool:
    return bool(item.get("final_query") or item.get("visual_query") or item.get("audio_query"))


def build_user_prompt(item: Dict[str, Any]) -> str:
    keywords = item.get("retrieval_keywords") or item.get("comment_focus_keywords") or []
    return f"""Original timestamped comment:
{item.get("comment", "")}

Modality label:
{normalize_label(item.get("primary_modality_hint") or item.get("modality_type"))}

Comment focus:
{item.get("comment_focus_summary", "")}

Grounded visual caption:
{item.get("grounded_visual_evidence") or item.get("integrated_visual", "")}

Grounded audio caption:
{item.get("grounded_audio_evidence") or item.get("integrated_audio", "")}

Retrieval keywords:
{json.dumps(keywords, ensure_ascii=False)}
"""


class QueryGenerator:
    def __init__(self, model: str = DEFAULT_MODEL, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key) if (OpenAI is not None and self.api_key) else None
        self.model = model
        self.prompt = load_prompt()

    def complete_json(self, user_prompt: str) -> Dict[str, Any]:
        if self.client is None:
            raise ValueError(
                "Install openai and provide OPENAI_API_KEY or --api-key when generating new query fields."
            )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )
        return parse_json_response(response.choices[0].message.content or "{}")

    def generate_item(self, item: Dict[str, Any], regenerate: bool = False) -> Dict[str, Any]:
        label = normalize_label(item.get("primary_modality_hint") or item.get("modality_type"))
        result = dict(item)
        result["primary_modality_hint"] = label

        if label == "unrelated":
            query = {
                "visual_query": "",
                "audio_query": "",
                "final_query": "",
                "used_visual_support": False,
                "used_audio_support": False,
                "query_keywords": [],
                "query_reason": "Filtered as unrelated by modality gating.",
            }
        elif has_query_fields(result) and not regenerate:
            query = {
                "visual_query": result.get("visual_query", ""),
                "audio_query": result.get("audio_query", ""),
                "final_query": result.get("final_query", ""),
                "used_visual_support": bool(result.get("visual_query")),
                "used_audio_support": bool(result.get("audio_query")),
                "query_keywords": result.get("retrieval_keywords") or result.get("comment_focus_keywords") or [],
                "query_reason": result.get("query_reason", "Existing query fields preserved."),
            }
        else:
            query = self.complete_json(build_user_prompt(result))

        for key in QUERY_KEYS:
            result[key] = query.get(key, result.get(key, "" if key.endswith("query") else []))
        result["query_model"] = self.model
        result["query_generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        return result


def summarize(items: List[Dict[str, Any]]) -> Dict[str, int]:
    stats = {"total": len(items), "visual": 0, "audio": 0, "mixed": 0, "unrelated": 0}
    with_query = 0
    for item in items:
        label = normalize_label(item.get("primary_modality_hint") or item.get("modality_type"))
        stats[label] = stats.get(label, 0) + 1
        if item.get("final_query"):
            with_query += 1
    stats["with_final_query"] = with_query
    return stats


def output_path_for(input_file: Path, output_dir: Path) -> Path:
    stem = input_file.stem
    for suffix in ["_gated", "_classified", "_moment_queries"]:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return output_dir / f"{stem}_moment_queries.json"


def process_file(
    input_file: Path,
    output_file: Path,
    generator: QueryGenerator,
    regenerate: bool = False,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    data = json.loads(input_file.read_text(encoding="utf-8"))
    items = get_items(data)
    processed = []
    for idx, item in enumerate(items):
        if limit is not None and idx >= limit:
            break
        processed.append(generator.generate_item(item, regenerate=regenerate))

    output = set_items(data, processed)
    output["query_generation_stats"] = summarize(processed)
    output["query_generation_metadata"] = {
        "input_file": input_file.name,
        "model": generator.model,
        "prompt_file": "prompts/query_generation_from_gating_output.txt",
        "regenerate": regenerate,
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
    parser = argparse.ArgumentParser(description="TCVP paper-version query generation")
    parser.add_argument("input", nargs="?", default="data/captions_gated", help="Input JSON file or folder")
    parser.add_argument("--output", "-o", default=None, help="Output file or folder")
    parser.add_argument("--pattern", default="*_gated.json", help="Input glob for folder mode")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--regenerate", action="store_true", help="Regenerate even when query fields exist")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else (
        input_path if input_path.is_dir() else input_path.with_name(f"{input_path.stem}_moment_queries.json")
    )
    generator = QueryGenerator(model=args.model, api_key=args.api_key)

    for input_file in iter_files(input_path, args.pattern):
        out_file = output_path_for(input_file, output_path) if output_path.suffix == "" else output_path
        if out_file.exists() and not args.overwrite:
            print(f"skip existing: {out_file}")
            continue
        print(f"query generation: {input_file} -> {out_file}")
        output = process_file(input_file, out_file, generator, regenerate=args.regenerate, limit=args.limit)
        print(output["query_generation_stats"])


if __name__ == "__main__":
    main()
