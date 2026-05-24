#!/usr/bin/env python3
"""
지정된 범위의 댓글 캡션 생성 스크립트 (video_id별 저장)

이 스크립트는 다음 단계로 구성됩니다.
1. CSV에서 댓글 데이터를 읽고 범위를 필터링합니다.
2. `OptimizedMomentQueryGenerator`를 사용하여 비디오 세그먼트 생성 + 분석을 수행합니다.
3. 댓글 처리 중 오류(잘못된 타임스탬프, 비디오 파일 없음, FFmpeg 실패 등)를 감지하면
   해당 댓글만 건너뛰고 다음 댓글로 진행합니다.
4. 결과는 비디오별로 `captions_by_video/captions_<video_id>.json`에 저장하고,
   전체 요약은 `captions_summary_<range>.json`에 저장합니다.

핵심 개선 사항:
* FFmpeg 오류(Invalid NAL unit size 등) 발생 시 예외를 잡고 해당 댓글만 실패 처리
* 스레드 풀 사용 시 안정적으로 future를 회수하도록 개선
* 반복되는 타입 변환/안전한 파일명 변환 로직을 별도 함수로 분리
* 풍부한 로그 메시지와 통계 출력
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, Future, wait, FIRST_COMPLETED
from queue import Queue
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import pandas as pd

from OptimizedMomentQueryGenerator import OptimizedMomentQueryGenerator

# Project root is parent of scripts/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)


# ---------------------------------------------------------------------------
# 로깅 설정
# ---------------------------------------------------------------------------
LOG_FORMAT = "[%(asctime)s] %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("generate_captions_range")


# ---------------------------------------------------------------------------
# 데이터 구조
# ---------------------------------------------------------------------------
[USER](slots=True)
class CommentJob:
    """한 개의 댓글을 처리하기 위한 JOB 메타 정보."""

    comment_index: int  # CSV 상의 1-based 인덱스
    video_id: str
    comment: str
    language: str
    channel_name: str
    category: str
    likes: Union[int, float]
    avg_time: float
    timestamp_raw: str
    video_path: str


[USER](slots=True)
class CommentFailure:
    """실패한 댓글 정보를 담기 위한 구조체."""

    comment_index: int
    video_id: str
    comment: str
    language: str
    channel_name: str
    category: str
    likes: Union[int, float]
    error: str

    def to_dict(self) -> Dict[str, Union[str, int, float]]:
        return {
            "comment_index": self.comment_index,
            "video_id": self.video_id,
            "comment": self.comment,
            "language": self.language,
            "channel_name": self.channel_name,
            "category": self.category,
            "likes": self.likes,
            "error": self.error,
            "analysis_type": "failed",
        }


class GeneratorPool:
    """여러 GPU에 모델 인스턴스를 생성하여 라운드 로빈 방식으로 제공."""

    def __init__(self, factories: Iterable[Callable[[], OptimizedMomentQueryGenerator]]):
        self._generators: List[OptimizedMomentQueryGenerator] = []
        self._available: Queue[int] = Queue()

        for idx, factory in enumerate(factories):
            generator = factory()
            self._generators.append(generator)
            self._available.put(idx)

        if not self._generators:
            raise ValueError("GeneratorPool을 초기화할 수 없습니다 (비어 있는 factories).")

    def acquire(self) -> int:
        """사용 가능한 제너레이터 인덱스를 가져옵니다 (블로킹)."""
        return self._available.get()

    def release(self, idx: int) -> None:
        """사용이 끝난 제너레이터 인덱스를 반환합니다."""
        self._available.put(idx)

    def get(self, idx: int) -> OptimizedMomentQueryGenerator:
        return self._generators[idx]

    [USER]
    def reference(self) -> OptimizedMomentQueryGenerator:
        return self._generators[0]

    def __len__(self) -> int:
        return len(self._generators)


# ---------------------------------------------------------------------------
# 유틸리티 함수
# ---------------------------------------------------------------------------
def parse_max_memory(arg: Optional[str]) -> Optional[Dict[Union[int, str], str]]:
    """'cuda:0=38GiB,cuda:1=38GiB' 형태 문자열을 dict로 파싱."""
    if not arg:
        return None
    
    result: Dict[Union[int, str], str] = {}
    for item in arg.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            logger.warning("max_memory 항목 무시: %s (형식: cuda:0=38GiB)", item)
            continue

        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            logger.warning("max_memory 항목 무시: %s", item)
            continue
        
        normalized_key: Union[int, str] = key
        if key.lower().startswith(("cuda:", "gpu:")):
            _, _, idx = key.partition(":")
            normalized_key = int(idx) if idx.isdigit() else key
        elif key.isdigit():
            normalized_key = int(key)
        
        result[normalized_key] = value

    return result or None


def safe_video_filename(video_id: str) -> str:
    """파일 시스템에 안전한 video_id로 변환."""
    replacements = {
        "/": "_slash_",
        "&": "_amp_",
        "?": "_qmark_",
        ":": "_colon_",
        "\\": "_backslash_",
        "*": "_star_",
        "|": "_pipe_",
        "<": "_lt_",
        ">": "_gt_",
        '"': "_quote_",
    }
    safe = video_id
    for key, value in replacements.items():
        safe = safe.replace(key, value)
    return safe


def restore_video_id(safe_name: str) -> str:
    """safe_video_filename 에서 변환된 video_id를 원래대로 복구."""
    replacements = {
        "_slash_": "/",
        "_amp_": "&",
        "_qmark_": "?",
        "_colon_": ":",
        "_backslash_": "\\",
        "_star_": "*",
        "_pipe_": "|",
        "_lt_": "<",
        "_gt_": ">",
        "_quote_": '"',
    }
    original = safe_name
    for key, value in replacements.items():
        original = original.replace(key, value)
    return original


def convert_json_types(obj):
    """numpy/pandas 타입을 기본 Python 타입으로 변환."""
    if hasattr(obj, "item"):
        return obj.item()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: convert_json_types(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_json_types(item) for item in obj]
    return obj


def ensure_dir(path: Union[str, Path]) -> Path:
    """디렉터리를 생성하고 Path 객체 반환."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_comments(start_idx: int, end_idx: Optional[int]) -> Tuple[pd.DataFrame, int]:
    """CSV에서 댓글 범위를 읽어 DataFrame과 전체 수를 반환."""
    csv_path = Path(PROJECT_ROOT) / "csv" / "merged_filtered_comments_with_dedup_lang.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {csv_path}")

    df = pd.read_csv(csv_path)
    total_comments = len(df)

    if total_comments == 0:
        raise ValueError("CSV에 댓글이 존재하지 않습니다.")

    end_idx = end_idx or total_comments

    start_zero = start_idx - 1
    end_zero = min(end_idx, total_comments)

    if start_zero < 0 or start_zero >= total_comments:
        raise ValueError(f"시작 인덱스가 잘못되었습니다: {start_idx} (1~{total_comments})")
    if end_zero <= start_zero:
        raise ValueError(f"끝 인덱스가 잘못되었습니다: {end_idx} (시작보다 커야 함)")

    subset = df.iloc[start_zero:end_zero].copy()
    return subset, total_comments


# ---------------------------------------------------------------------------
# CommentJob 준비 및 실행 함수
# ---------------------------------------------------------------------------
def prepare_comment_job(generator: OptimizedMomentQueryGenerator, row: pd.Series, index_1based: int) -> Tuple[Optional[CommentJob], Optional[CommentFailure]]:
    """하나의 댓글 정보를 CommentJob으로 준비."""
    video_id = row["video_id"]
    comment = row["comment"]
    language = row["language"]
    channel_name = row.get("channel_name", "Unknown")
    category = row.get("category", "Unknown")
    likes = row.get("likes", 0)
    timestamp_raw = row["timestamp"]

    try:
        parsed = generator.parse_timestamp(timestamp_raw)
        avg_time = generator.calculate_average_timestamp(parsed)
    except Exception as exc:  # timestamp 파싱 예외
        logger.warning("타임스탬프 파싱 실패 (index=%s, video_id=%s): %s", index_1based, video_id, exc)
        failure = CommentFailure(
            comment_index=index_1based,
            video_id=video_id,
            comment=comment,
            language=language,
            channel_name=channel_name,
            category=category,
            likes=likes,
            error=f"Invalid timestamp: {timestamp_raw}",
        )
        return None, failure
    
    if avg_time <= 0:
        failure = CommentFailure(
            comment_index=index_1based,
            video_id=video_id,
            comment=comment,
            language=language,
            channel_name=channel_name,
            category=category,
            likes=likes,
            error="Invalid timestamp (<= 0)",
        )
        return None, failure

    video_path = generator.find_video_file(video_id, channel_name, category)
    if not video_path:
        failure = CommentFailure(
            comment_index=index_1based,
            video_id=video_id,
            comment=comment,
            language=language,
            channel_name=channel_name,
            category=category,
            likes=likes,
            error=f"Video file not found: {video_id}",
        )
        return None, failure

    video_path = Path(video_path)
    if not video_path.exists():
        failure = CommentFailure(
            comment_index=index_1based,
            video_id=video_id,
            comment=comment,
            language=language,
            channel_name=channel_name,
            category=category,
            likes=likes,
            error=f"Video file does not exist: {video_path}",
        )
        return None, failure

    if not os.access(video_path, os.R_OK):
        failure = CommentFailure(
            comment_index=index_1based,
            video_id=video_id,
            comment=comment,
            language=language,
            channel_name=channel_name,
            category=category,
            likes=likes,
            error=f"Video file not readable: {video_path}",
        )
        return None, failure
    
    job = CommentJob(
        comment_index=index_1based,
        video_id=video_id,
        comment=comment,
        language=language,
        channel_name=channel_name,
        category=category,
        likes=likes,
        avg_time=avg_time,
        timestamp_raw=timestamp_raw,
        video_path=str(video_path),
    )
    return job, None


def create_segments_task(
    generator: OptimizedMomentQueryGenerator,
    job: CommentJob,
    use_sequential: bool = False,
) -> Tuple[List[str], Optional[str]]:
    """
    FFmpeg 세그먼트 생성 태스크.

    Returns:
        (segment_paths, error_message)
    """
    try:
        if use_sequential and hasattr(generator, "create_segments_sequential"):
            segments = generator.create_segments_sequential(job.video_path, job.avg_time)
        else:
            segments = generator.create_segments_parallel(job.video_path, job.avg_time)
        if not segments:
            return [], "create_segments_parallel returned no segments"
        return segments, None
    except Exception as exc:
        return [], str(exc)


def run_inference(generator: OptimizedMomentQueryGenerator, job: CommentJob, segments: List[str]) -> Tuple[Dict, bool]:
    """생성된 세그먼트로 AI 분석을 실행."""
    valid_segments = [Path(seg) for seg in segments if seg and Path(seg).exists()]
    if not valid_segments:
        failure = CommentFailure(
            comment_index=job.comment_index,
            video_id=job.video_id,
            comment=job.comment,
            language=job.language,
            channel_name=job.channel_name,
            category=job.category,
            likes=job.likes,
            error="All segments failed to create",
        )
        return failure.to_dict(), False

    if len(valid_segments) < generator.num_segments:
        logger.warning(
            "일부 세그먼트만 생성되었습니다 (%s/%s, comment_index=%s, video_id=%s)",
            len(valid_segments),
            generator.num_segments,
            job.comment_index,
            job.video_id,
        )

    logger.info("AI 분석 시작 (comment_index=%s, video_id=%s)", job.comment_index, job.video_id)
    start = time.time()
    try:
        queries = generator.process_segments_parallel([str(p) for p in valid_segments], job.comment)
    except Exception as exc:
        failure = CommentFailure(
            comment_index=job.comment_index,
            video_id=job.video_id,
            comment=job.comment,
            language=job.language,
            channel_name=job.channel_name,
            category=job.category,
            likes=job.likes,
            error=f"Segment processing failed: {exc}",
        )
        logger.exception("세그먼트 분석 실패 (comment_index=%s, video_id=%s)", job.comment_index, job.video_id)
        return failure.to_dict(), False

    elapsed = time.time() - start
    logger.info("AI 분석 완료 (%.2fs, comment_index=%s, video_id=%s)", elapsed, job.comment_index, job.video_id)

    segments_result = []
    for idx, query in enumerate(queries):
        time_start = max(0.0, job.avg_time - generator.half_duration + idx * generator.segment_duration)
        time_end = max(0.0, job.avg_time - generator.half_duration + (idx + 1) * generator.segment_duration)
        segments_result.append(
            {
                "segment_index": idx + 1,
                "time_range": f"{time_start:.1f}s - {time_end:.1f}s",
                "query": query,
            }
        )

    result = {
            "comment_index": job.comment_index,
            "video_id": job.video_id,
            "comment": job.comment,
            "language": job.language,
            "channel_name": job.channel_name,
            "category": job.category,
            "likes": job.likes,
            "comment_timestamp": job.avg_time,
            "total_duration": f"{generator.total_duration}s ({generator.num_segments} segments of {generator.segment_duration}s each)",
        "segments": segments_result,
            "analysis_type": "optimized_video_analysis",
        "processing_time": elapsed,
    }
    return result, True


# ---------------------------------------------------------------------------
# 비디오 단위 처리
# ---------------------------------------------------------------------------
def process_video_comments(
    generator_pool: GeneratorPool,
    video_id: str,
    video_comments: pd.DataFrame,
    sequential_ffmpeg: bool,
) -> Tuple[List[Dict], Dict[str, Union[str, int, float]]]:
    """
    1개의 비디오에 속한 댓글들을 처리하고 결과와 통계를 반환.

    FFmpeg 세그먼트 생성은 순차적으로 수행하고, 생성된 세그먼트는
    GPU 인스턴스 풀에서 순환하며 병렬 추론을 수행합니다.
    """
    logger.info("비디오 처리 시작: %s (%s개의 댓글)", video_id, len(video_comments))

    successful = 0
    failed = 0
    results: List[Dict] = []

    inference_executor = ThreadPoolExecutor(max_workers=max(len(generator_pool), 1))
    inference_futures: List[Tuple[Future, CommentJob]] = []

    def submit_inference(job: CommentJob, segments: List[str], pool_idx: Optional[int] = None) -> None:
        acquired_idx = pool_idx if pool_idx is not None else generator_pool.acquire()

        def task(
            pool_idx: int = acquired_idx,
            job: CommentJob = job,
            segments: List[str] = segments,
        ):
            try:
                generator = generator_pool.get(pool_idx)
                return run_inference(generator, job, segments)
            finally:
                generator_pool.release(pool_idx)

        future = inference_executor.submit(task)
        inference_futures.append((future, job))

    def collect_completed(block: bool = False) -> None:
        nonlocal inference_futures, successful, failed
        if not inference_futures:
            return

        futures_only = [future for future, _ in inference_futures]
        if block:
            done, _ = wait(futures_only)
        else:
            done, _ = wait(futures_only, timeout=0, return_when=FIRST_COMPLETED)
            if not done:
                return

        remaining: List[Tuple[Future, CommentJob]] = []
        for future, job in inference_futures:
            if future in done:
                try:
                    inference_result, ok = future.result()
                except Exception as exc:
                    logger.exception(
                        "인퍼런스 작업 실패 (comment_index=%s, video_id=%s)",
                        job.comment_index,
                        job.video_id,
                    )
                    failure = CommentFailure(
                        comment_index=job.comment_index,
                        video_id=job.video_id,
                        comment=job.comment,
                        language=job.language,
                        channel_name=job.channel_name,
                        category=job.category,
                        likes=job.likes,
                        error=f"Inference task crashed: {exc}",
                    )
                    inference_result = failure.to_dict()
                    ok = False

                results.append(inference_result)
                if ok:
                    successful += 1
                else:
                    failed += 1
            else:
                remaining.append((future, job))

        inference_futures = remaining

    for _, row in video_comments.iterrows():
        collect_completed(block=False)

        pool_idx = generator_pool.acquire()
        generator = generator_pool.get(pool_idx)

        csv_index_1based = int(row.name) + 1
        job, failure = prepare_comment_job(generator, row, csv_index_1based)
        if failure:
            logger.warning(
                "댓글 준비 실패 (index=%s, video_id=%s): %s",
                csv_index_1based,
                row["video_id"],
                failure.error,
            )
            results.append(failure.to_dict())
            failed += 1
            generator_pool.release(pool_idx)
            continue

        logger.info("댓글 처리 시작 (index=%s, video_id=%s)", job.comment_index, job.video_id)
        segments, segment_error = create_segments_task(
            generator,
            job,
            use_sequential=sequential_ffmpeg,
        )

        if segment_error:
            logger.warning(
                "세그먼트 생성 실패 (comment_index=%s, video_id=%s): %s",
                job.comment_index,
                job.video_id,
                segment_error,
            )
            failure = CommentFailure(
                comment_index=job.comment_index,
                video_id=job.video_id,
                comment=job.comment,
                language=job.language,
                channel_name=job.channel_name,
                category=job.category,
                likes=job.likes,
                error=f"Segment creation failed: {segment_error}",
            )
            results.append(failure.to_dict())
            failed += 1
            generator_pool.release(pool_idx)
            continue

        submit_inference(job, segments, pool_idx)

    while inference_futures:
        collect_completed(block=True)

    inference_executor.shutdown(wait=True)

    summary = {
        "video_id": video_id,
        "total_comments": len(video_comments),
        "successful_comments": successful,
        "failed_comments": failed,
        "success_rate": f"{(successful / len(video_comments) * 100):.1f}%" if video_comments.size else "0%",
    }

    logger.info("비디오 처리 완료: %s (성공=%s, 실패=%s)", video_id, successful, failed)
    return results, summary


# ---------------------------------------------------------------------------
# 결과 저장
# ---------------------------------------------------------------------------
def save_video_summary(video_id: str, results: List[Dict], summary: Dict[str, Union[str, int, float]], output_dir: Union[str, Path]) -> Path:
    output_dir = ensure_dir(output_dir)
    safe_name = safe_video_filename(video_id)
    output_path = output_dir / f"captions_{safe_name}.json"

    payload = {
        "video_id": video_id,
        "total_comments": summary["total_comments"],
        "successful_comments": summary["successful_comments"],
        "failed_comments": summary["failed_comments"],
        "success_rate": summary["success_rate"],
        "comments": results,
    }

    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(convert_json_types(payload), fp, ensure_ascii=False, indent=2)

    logger.info("비디오 결과 저장 완료: %s", output_path)
    return output_path


def save_overall_summary(
    start_idx: int,
    end_idx: int,
    total_comments: int,
    processed_comments: int,
    success_count: int,
    failure_count: int,
    total_videos: int,
    elapsed: float,
) -> Path:
    output_path = Path(PROJECT_ROOT) / f"captions_summary_{start_idx}_to_{end_idx}.json"
    payload = {
        "total_comments_in_csv": total_comments,
        "processed_comments": processed_comments,
        "successful_comments": success_count,
        "failed_comments": failure_count,
        "success_rate": f"{(success_count / processed_comments * 100):.1f}%" if processed_comments else "0%",
        "total_videos": total_videos,
        "processing_time_seconds": elapsed,
    }
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(convert_json_types(payload), fp, ensure_ascii=False, indent=2)
    logger.info("전체 요약 저장: %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# 메인 엔트리
# ---------------------------------------------------------------------------
def generate_captions_range(
    start_idx: int,
    end_idx: Optional[int] = None,
    device_map: str = "auto",
    device_cycle: Optional[Iterable[str]] = None,
    max_memory: Optional[Dict[Union[int, str], str]] = None,
    max_workers_segments: int = 3,
    max_workers_analysis: int = 2,  # 현재 OptimizedMomentQueryGenerator 내부에서 사용
    ffmpeg_threads: Optional[int] = None,
    skip_videos: Optional[Iterable[str]] = None,
    sequential_ffmpeg: bool = True,
) -> None:
    comments_df, total_comments = read_comments(start_idx, end_idx)
    if comments_df.empty:
        logger.warning("처리할 댓글이 없습니다.")
        return

    end_idx = int(comments_df.index[-1]) + 1

    logger.info("전체 댓글 수: %s", total_comments)
    logger.info("처리 범위: %s ~ %s (%s개)", start_idx, end_idx, len(comments_df))

    # video_id별 그룹화
    grouped = comments_df.groupby("video_id")
    video_ids = list(grouped.groups.keys())
    logger.info("범위 내 비디오 수: %s", len(video_ids))

    if skip_videos:
        skip_set = {vid.strip() for vid in skip_videos if vid.strip()}
        if skip_set:
            before = len(video_ids)
            video_ids = [vid for vid in video_ids if vid not in skip_set]
            if len(video_ids) != before:
                logger.info("사용자 지정 스킵 비디오 %s개 제외", before - len(video_ids))
                comments_df = comments_df[comments_df["video_id"].isin(video_ids)]
                grouped = comments_df.groupby("video_id")

    # 이미 처리된 비디오 제거
    captions_dir = Path(PROJECT_ROOT) / "captions_by_video"
    if captions_dir.exists():
        existing_ids = {
            restore_video_id(Path(file).stem.replace("captions_", ""))
            for file in os.listdir(captions_dir)
            if file.startswith("captions_") and file.endswith(".json")
        }
        before = len(video_ids)
        video_ids = [vid for vid in video_ids if vid not in existing_ids]
        if len(video_ids) != before:
            logger.info("이미 처리된 비디오 %s개 제외, 남은 비디오 %s개", before - len(video_ids), len(video_ids))
            comments_df = comments_df[comments_df["video_id"].isin(video_ids)]
            grouped = comments_df.groupby("video_id")

    if not video_ids:
        logger.info("새로 처리할 비디오가 없습니다. 종료합니다.")
        return

    logger.info("모델 초기화 중...")
    model_start = time.time()

    devices: List[str] = []
    if device_cycle:
        for item in device_cycle:
            if isinstance(item, str) and "," in item:
                devices.extend(token.strip() for token in item.split(",") if token.strip())
            elif item:
                devices.append(str(item).strip())
    if not devices:
        devices = [device_map]

    generator_kwargs = {
        "model_path": "Qwen/Qwen2.5-Omni-7B",
        "segment_duration": 6,
        "num_segments": 3,
        "max_batch_size": 3,
        "max_new_tokens": 150,
        "video_fps": 24,
        "ffmpeg_preset": "ultrafast",
        "ffmpeg_crf": 28,
        "use_model_compile": True,
        "use_flash_attention": True,
        "torch_dtype_str": "float16",
        "max_workers_segments": max_workers_segments,
        "max_workers_analysis": max_workers_analysis,
        "ffmpeg_threads": ffmpeg_threads,
        "max_memory": max_memory,
    }

    def make_factory(target_device: str) -> Callable[[], OptimizedMomentQueryGenerator]:
        def _factory(device_name: str = target_device) -> OptimizedMomentQueryGenerator:
            logger.info("GPU 디바이스 %s용 모델 초기화 시작...", device_name)
            instance = OptimizedMomentQueryGenerator(
                device_map=device_name,
                **generator_kwargs,
            )
            logger.info("GPU 디바이스 %s용 모델 초기화 완료", device_name)
            return instance

        return _factory

    generator_pool = GeneratorPool(make_factory(device) for device in devices)

    model_elapsed = time.time() - model_start
    logger.info("모델 초기화 완료 (%s개 인스턴스, %.2fs)", len(generator_pool), model_elapsed)

    overall_start = time.time()
    success_total = 0
    failure_total = 0

    for idx, video_id in enumerate(video_ids, start=1):
        logger.info("====== 비디오 %s/%s: %s ======", idx, len(video_ids), video_id)
        video_comments = grouped.get_group(video_id)
        video_results, video_summary = process_video_comments(
            generator_pool=generator_pool,
            video_id=video_id,
            video_comments=video_comments,
            sequential_ffmpeg=sequential_ffmpeg,
        )

        success_total += int(video_summary["successful_comments"])
        failure_total += int(video_summary["failed_comments"])

        save_video_summary(video_id, video_results, video_summary, captions_dir)

        # GPU 메모리 정리
        try:
            import torch  # noqa: WPS433 (torch는 optional dependency)

            if torch.cuda.is_available():
                current_device = torch.cuda.current_device()
                for device_idx in range(torch.cuda.device_count()):
                    torch.cuda.set_device(device_idx)
                    torch.cuda.empty_cache()
                torch.cuda.set_device(current_device)
        except Exception:  # torch 미설치 또는 GPU 미사용
            pass

    overall_elapsed = time.time() - overall_start
    processed_comments = success_total + failure_total
    save_overall_summary(
        start_idx=start_idx,
        end_idx=end_idx,
        total_comments=total_comments,
        processed_comments=processed_comments,
        success_count=success_total,
        failure_count=failure_total,
        total_videos=len(video_ids),
        elapsed=overall_elapsed,
    )

    logger.info("전체 파이프라인 완료 (성공=%s, 실패=%s, 총시간=%.2fs)", success_total, failure_total, overall_elapsed)


def main() -> None:
    parser = argparse.ArgumentParser(description="지정된 범위의 댓글 캡션 생성")
    parser.add_argument("start", type=int, help="시작 댓글 번호 (1부터 시작)")
    parser.add_argument("--end", type=int, help="끝 댓글 번호 (미지정 시 끝까지)")
    parser.add_argument("--device-map", type=str, default="auto", help="HuggingFace device_map 설정 (기본: auto)")
    parser.add_argument(
        "--device-cycle",
        type=str,
        help="여러 GPU를 순차적으로 사용할 때 쉼표로 구분된 목록 (예: 'cuda:0,cuda:1')",
    )
    parser.add_argument("--max-memory", type=str, help="GPU별 메모리 제한 (예: 'cuda:0=38GiB,cuda:1=38GiB')")
    parser.add_argument("--workers-segments", type=int, default=3, help="세그먼트 생성 ThreadPoolExecutor worker 수 (기본: 3)")
    parser.add_argument("--workers-analysis", type=int, default=2, help="세그먼트 분석 ThreadPoolExecutor worker 수 (기본: 2)")
    parser.add_argument("--ffmpeg-threads", type=int, help="FFmpeg -threads 값 (미설정 시 자동)")
    parser.add_argument(
        "--skip-video",
        action="append",
        help="처리에서 제외할 video_id (여러 번 지정 가능, 쉼표로도 구분 가능)",
    )
    parser.add_argument(
        "--parallel-ffmpeg",
        action="store_true",
        help="FFmpeg 세그먼트 생성을 병렬로 유지 (기본: 순차 처리)",
    )
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="로그 레벨 설정 (기본: INFO)")
    
    args = parser.parse_args()
    logger.setLevel(getattr(logging, args.log_level.upper()))
    
    # 기본값 설정: max_memory는 None (자동 감지)
    max_memory = parse_max_memory(args.max_memory)
    
    skip_videos: List[str] = []
    if args.skip_video:
        for item in args.skip_video:
            skip_videos.extend([token.strip() for token in item.split(",") if token.strip()])

    device_cycle = None
    if args.device_cycle:
        device_cycle = [token.strip() for token in args.device_cycle.split(",") if token.strip()]

    # 기본값으로 간단하게 실행
    logger.info("기본 설정으로 실행합니다:")
    logger.info(f"  - device_map: {args.device_map}")
    logger.info(f"  - workers_segments: {args.workers_segments}")
    logger.info(f"  - workers_analysis: {args.workers_analysis}")
    if max_memory:
        logger.info(f"  - max_memory: {max_memory}")
    else:
        logger.info("  - max_memory: 자동 감지")

    generate_captions_range(
        start_idx=args.start,
        end_idx=args.end,
        device_map=args.device_map,
        device_cycle=device_cycle,
        max_memory=max_memory,
        max_workers_segments=max(args.workers_segments, 1),
        max_workers_analysis=max(args.workers_analysis, 1),
        ffmpeg_threads=args.ffmpeg_threads if (args.ffmpeg_threads and args.ffmpeg_threads > 0) else None,
        skip_videos=skip_videos or None,
        sequential_ffmpeg=not args.parallel_ffmpeg,
    )


if __name__ == "__main__":
    main()
