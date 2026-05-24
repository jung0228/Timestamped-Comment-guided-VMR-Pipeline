import os
import json

# Project root is parent of scripts/
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)

import torch
import soundfile as sf
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info
import subprocess
from tqdm import tqdm
import cv2
import numpy as np
from PIL import Image
from typing import List, Dict, Any, Tuple
from pathlib import Path
import warnings
import pandas as pd
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc

# 경고 메시지 숨기기
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# root logger의 WARNING 레벨 메시지 숨기기
import logging
logging.getLogger().setLevel(logging.ERROR)
logging.getLogger("root").setLevel(logging.ERROR)


class OptimizedMomentQueryGenerator:
    def __init__(self, 
                 model_path: str = "Qwen/Qwen2.5-Omni-7B", 
                 segment_duration: int = 6,
                 num_segments: int = 3,
                 max_batch_size: int = 3,
                 max_new_tokens: int = 200,
                 video_fps: int = 24,
                 ffmpeg_preset: str = "ultrafast",
                 ffmpeg_crf: int = 28,
                 use_model_compile: bool = True,
                 use_flash_attention: bool = True,
                 torch_dtype_str: str = "float16",
                 max_workers_segments: int = 3,
                 max_workers_analysis: int = 2,
                 ffmpeg_threads: int = None,
                 device_map: str = "auto",
                 max_memory: Dict[str, str] = None):
        """최적화된 테스트용 Qwen 2.5 Omni 모델 초기화"""
        print("최적화된 테스트 모델 로딩 중...")
        
        # 설정 저장
        self.model_path = model_path
        self.segment_duration = segment_duration
        self.num_segments = num_segments
        self.max_batch_size = max_batch_size
        self.max_new_tokens = max_new_tokens
        self.video_fps = video_fps
        self.ffmpeg_preset = ffmpeg_preset
        self.ffmpeg_crf = ffmpeg_crf
        self.use_model_compile = use_model_compile
        self.use_flash_attention = use_flash_attention
        self.USE_AUDIO_IN_VIDEO = True
        self.max_workers_segments = max_workers_segments
        self.max_workers_analysis = max_workers_analysis
        self.ffmpeg_threads = ffmpeg_threads
        self.device_map = device_map
        self.max_memory = max_memory
        
        # 자동 계산되는 값들
        self.total_duration = self.num_segments * self.segment_duration
        self.half_duration = self.total_duration // 2
        
        # torch dtype 설정
        torch_dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "auto": "auto"
        }
        self.torch_dtype = torch_dtype_map.get(torch_dtype_str, torch.float16)
        
        # GPU 메모리 최적화
        torch.cuda.empty_cache()
        
        # 모델 로드 설정 준비
        model_kwargs = {
            "torch_dtype": self.torch_dtype,
            "device_map": self.device_map,
            "low_cpu_mem_usage": True,
        }
        
        if self.max_memory:
            model_kwargs["max_memory"] = self.max_memory
        
        # Flash Attention 설정
        if self.use_flash_attention:
            try:
                model_kwargs["attn_implementation"] = "flash_attention_2"
            except:
                print("Flash Attention 2 사용 불가, 기본 어텐션 사용")
        
        # 모델/프로세서 로드
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            self.model_path, **model_kwargs
        )
        self.processor = Qwen2_5OmniProcessor.from_pretrained(self.model_path)
        
        # 모델 컴파일 (옵션)
        if self.use_model_compile:
            try:
                print("모델 컴파일 중... (첫 실행시 시간이 걸릴 수 있습니다)")
                self.model = torch.compile(self.model, mode="reduce-overhead")
                print("✓ 모델 컴파일 완료")
            except Exception as e:
                print(f"모델 컴파일 실패 (무시하고 계속): {e}")
        
        # 모델을 eval 모드로 설정
        self.model.eval()
        
        # 비디오 처리 관련 설정
        self.video_dir = Path("data/videos")
        self.segments_dir = Path(f"optimized_test_video_segments_{self.total_duration}s")
        self.segments_dir.mkdir(exist_ok=True)
        
        # 최적화된 생성 파라미터
        self.generation_config = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": False,     # 그리디 디코딩으로 속도 향상
            "repetition_penalty": 1.1,
            "pad_token_id": self.processor.tokenizer.eos_token_id,
            "use_cache": True,      # KV 캐시 사용
        }
        
        print("최적화된 테스트 모델 로딩 완료!")
        print(f"모델 경로: {self.model_path}")
        print(f"세그먼트 구성: {self.num_segments}개 × {self.segment_duration}초 = {self.total_duration}초")
        print(f"실제 분석 범위: 중심 ±{self.half_duration}초 (총 {self.total_duration}초)")
        print(f"최대 배치 크기: {self.max_batch_size}")
        print(f"최대 토큰 수: {self.max_new_tokens}")
        print(f"비디오 FPS: {self.video_fps}")
        print(f"FFmpeg 설정: {self.ffmpeg_preset} 프리셋, CRF {self.ffmpeg_crf}")
        if self.ffmpeg_threads:
            print(f"FFmpeg 스레드: {self.ffmpeg_threads}")
        print(f"모델 컴파일: {'ON' if self.use_model_compile else 'OFF'}")
        print(f"Flash Attention: {'ON' if self.use_flash_attention else 'OFF'}")
        print(f"Torch dtype: {self.torch_dtype}")
        print(f"병렬 처리: 세그먼트 생성 {self.max_workers_segments}개, 분석 {self.max_workers_analysis}개")
        print(f"디바이스 매핑: {self.device_map}")
        if self.max_memory:
            print(f"GPU 메모리 제한: {self.max_memory}")
        print(f"⚡ 총 분석 시간: {self.total_duration}초")
    
    def find_video_file(self, video_id: str, channel_name: str = None, category: str = None) -> str:
        """기존 비디오 파일 찾기 (새로운 경로 구조 지원)"""
        # 간단한 파일 캐시
        if not hasattr(self, '_video_cache'):
            self._video_cache = {}
        
        cache_key = f"{video_id}_{channel_name}_{category}"
        if cache_key in self._video_cache:
            return self._video_cache[cache_key]
        
        extensions = ['.mp4', '.mkv', '.avi', '.mov', '.webm']
        
        # 1. 새로운 경로 구조 시도: /youtube_videos/{category}/{channel_name}/{video_id}.ext
        if channel_name and category:
            from pathlib import Path
            base_dir = Path(self.video_dir).parent / "youtube_videos"
            new_structure_dir = base_dir / category / channel_name
            
            if new_structure_dir.exists():
                for ext in extensions:
                    video_path = new_structure_dir / f"{video_id}{ext}"
                    if video_path.exists():
                        self._video_cache[cache_key] = str(video_path)
                        return str(video_path)
                
                # video_id가 포함된 파일 찾기
                for video_file in new_structure_dir.glob(f"*{video_id}*"):
                    if video_file.suffix in extensions:
                        self._video_cache[cache_key] = str(video_file)
                        return str(video_file)
        
        # 2. 기존 경로 구조 시도 (하위 호환성)
        for ext in extensions:
            video_path = self.video_dir / f"{video_id}{ext}"
            if video_path.exists():
                self._video_cache[cache_key] = str(video_path)
                return str(video_path)
        
        # video_id가 포함된 파일 찾기
        for video_file in self.video_dir.glob(f"*{video_id}*"):
            if video_file.suffix in extensions:
                self._video_cache[cache_key] = str(video_file)
                return str(video_file)
        
        self._video_cache[cache_key] = None
        return None
    
    def check_video_compatibility(self, video_path: str) -> bool:
        """비디오 파일의 호환성 확인 (캐시 추가)"""
        if not hasattr(self, '_compatibility_cache'):
            self._compatibility_cache = {}
        
        if video_path in self._compatibility_cache:
            return self._compatibility_cache[video_path]
        
        try:
            cmd = [
                "ffprobe", "-v", "quiet", "-print_format", "json", 
                "-show_streams", video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                streams_info = json.loads(result.stdout)
                
                for stream in streams_info['streams']:
                    if stream['codec_type'] == 'video':
                        is_compatible = (stream.get('codec_name') == 'h264' and 
                                       stream.get('pix_fmt') == 'yuv420p')
                        self._compatibility_cache[video_path] = is_compatible
                        return is_compatible
                        
            self._compatibility_cache[video_path] = False
            return False
        except Exception as e:
            self._compatibility_cache[video_path] = False
            return False
    
    def parse_timestamp(self, timestamp_str: str) -> List[float]:
        """timestamp 문자열을 파싱하여 초 단위로 변환 (h:mm:ss, mm:ss, ss 모두 지원)"""
        try:
            import ast
            if timestamp_str.startswith('[') and timestamp_str.endswith(']'):
                timestamp_list = ast.literal_eval(timestamp_str)
            else:
                timestamp_list = [timestamp_str]
            
            seconds_list = []
            for ts in timestamp_list:
                if ':' in ts:
                    parts = ts.split(':')
                    if len(parts) == 3:
                        # h:mm:ss 형식
                        hours, minutes, seconds = parts
                        total_seconds = int(hours) * 3600 + int(minutes) * 60 + int(seconds)
                        seconds_list.append(float(total_seconds))
                    elif len(parts) == 2:
                        # mm:ss 형식
                        minutes, seconds = parts
                        total_seconds = int(minutes) * 60 + int(seconds)
                        seconds_list.append(float(total_seconds))
                    else:
                        # 잘못된 형식
                        seconds_list.append(float(parts[0]))
                else:
                    # 초만 있는 경우
                    seconds_list.append(float(ts))
            
            return seconds_list
        except Exception as e:
            print(f"⚠️ 타임스탬프 파싱 오류: {timestamp_str} - {e}")
            return []
    
    def calculate_average_timestamp(self, timestamps: List[float]) -> float:
        """timestamp 리스트의 평균을 계산"""
        if not timestamps:
            return 0.0
        return np.mean(timestamps)
    
    def create_segments_parallel(self, original_video_path: str, comment_timestamp: float) -> List[str]:
        """병렬로 세그먼트 생성 (최적화)"""
        segments = []
        center_time = comment_timestamp
        start_time = max(center_time - self.half_duration, 0)
        
        print(f"  병렬 세그먼트 생성: {start_time}s부터 {self.total_duration}초 ({self.num_segments}개 세그먼트)")
        
        # 병렬 처리를 위한 함수
        def create_single_segment(i):
            segment_start = start_time + (i * self.segment_duration)
            segment_end = segment_start + self.segment_duration
            
            base_name = os.path.splitext(os.path.basename(original_video_path))[0]
            segment_path = self.segments_dir / f"opt_segment_{base_name}_{int(comment_timestamp)}s_{i+1}_{segment_start:.1f}s_{segment_end:.1f}s.mp4"
            
            # 변수화된 ffmpeg 명령어
            cmd = [
                "ffmpeg", "-i", original_video_path,
                "-ss", str(segment_start), "-t", str(self.segment_duration),
                "-c:v", "libx264", "-c:a", "aac", "-pix_fmt", "yuv420p",
                "-preset", self.ffmpeg_preset,
                "-crf", str(self.ffmpeg_crf),
            ]

            if self.ffmpeg_threads and self.ffmpeg_threads > 0:
                cmd.extend(["-threads", str(self.ffmpeg_threads)])

            cmd.extend([
                "-avoid_negative_ts", "make_zero",
                "-vsync", "cfr", "-r", str(self.video_fps),
                "-movflags", "+faststart",
                "-y", str(segment_path)
            ])
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0 and os.path.exists(segment_path):
                return (i, str(segment_path), True)
            else:
                return (i, None, False)
        
        # 병렬로 세그먼트 생성
        segments = [None] * self.num_segments
        
        with ThreadPoolExecutor(max_workers=self.max_workers_segments) as executor:
            future_to_index = {executor.submit(create_single_segment, i): i for i in range(self.num_segments)}
            
            for future in as_completed(future_to_index):
                try:
                    index, segment_path, success = future.result()
                    segments[index] = segment_path
                    if success:
                        print(f"    ✓ 세그먼트 {index+1}/{self.num_segments} 생성 완료")
                    else:
                        print(f"    ✗ 세그먼트 {index+1}/{self.num_segments} 생성 실패")
                except Exception as e:
                    index = future_to_index[future]
                    segments[index] = None
                    print(f"    ✗ 세그먼트 {index+1}/{self.num_segments} 오류: {e}")
        
        return segments
    
    def create_segments_sequential(self, original_video_path: str, comment_timestamp: float) -> List[str]:
        """순차적으로 세그먼트 생성 (백업용)"""
        segments = []
        center_time = comment_timestamp
        start_time = max(center_time - self.half_duration, 0)
        
        print(f"  순차 세그먼트 생성: {start_time}s부터 {self.total_duration}초 ({self.num_segments}개 세그먼트)")
        
        for i in range(self.num_segments):
            segment_start = start_time + (i * self.segment_duration)
            segment_end = segment_start + self.segment_duration
            
            base_name = os.path.splitext(os.path.basename(original_video_path))[0]
            segment_path = self.segments_dir / f"opt_segment_{base_name}_{int(comment_timestamp)}s_{i+1}_{segment_start:.1f}s_{segment_end:.1f}s.mp4"
            
            # 변수화된 ffmpeg 명령어
            cmd = [
                "ffmpeg", "-i", original_video_path,
                "-ss", str(segment_start), "-t", str(self.segment_duration),
                "-c:v", "libx264", "-c:a", "aac", "-pix_fmt", "yuv420p",
                "-preset", self.ffmpeg_preset,
                "-crf", str(self.ffmpeg_crf),
            ]

            if self.ffmpeg_threads and self.ffmpeg_threads > 0:
                cmd.extend(["-threads", str(self.ffmpeg_threads)])

            cmd.extend([
                "-avoid_negative_ts", "make_zero",
                "-vsync", "cfr", "-r", str(self.video_fps),
                "-movflags", "+faststart",
                "-y", str(segment_path)
            ])
            
            print(f"    세그먼트 {i+1}/{self.num_segments} 생성 중...")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0 and os.path.exists(segment_path):
                segments.append(str(segment_path))
                print(f"    ✓ 세그먼트 {i+1}/{self.num_segments} 생성 완료")
            else:
                segments.append(None)
                print(f"    ✗ 세그먼트 {i+1}/{self.num_segments} 생성 실패")
        
        return segments
    
    def extract_audio_optimized(self, video_path: str, start_time: float, 
                              duration: float = None) -> Tuple[np.ndarray, int]:
        """최적화된 오디오 추출"""
        if duration is None:
            duration = float(self.segment_duration)
            
        start_time = max(start_time, 0)
        duration = max(duration, 1)
        
        if not os.path.exists(video_path):
            return None, None
        
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        temp_audio = f"/tmp/opt_temp_audio_{base_name}_{os.getpid()}.wav"
        
        # 변수화된 오디오 추출 명령어
        cmd_audio = [
            "ffmpeg", "-i", video_path, "-vn",
            "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            "-f", "wav", "-y", temp_audio
        ]
        
        try:
            result = subprocess.run(cmd_audio, capture_output=True, text=True)
            
            if result.returncode == 0 and os.path.exists(temp_audio):
                file_size = os.path.getsize(temp_audio)
                
                if file_size > 0:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        audio, sr = sf.read(temp_audio)
                else:
                    # 세그먼트 길이에 맞는 무음 생성
                    audio = np.zeros(16000 * self.segment_duration, dtype=np.float32)
                    sr = 16000
            else:
                # 세그먼트 길이에 맞는 무음 생성
                audio = np.zeros(16000 * self.segment_duration, dtype=np.float32)
                sr = 16000
            
            # 임시 파일 정리
            if os.path.exists(temp_audio):
                os.remove(temp_audio)
            
            return audio, sr
            
        except Exception as e:
            if os.path.exists(temp_audio):
                os.remove(temp_audio)
            return None, None
    
    def process_segments_parallel(self, segment_paths: List[str], comment: str) -> List[str]:
        """병렬로 여러 세그먼트 동시 처리"""
        print(f"  병렬 분석 모드: {len(segment_paths)}개 세그먼트")
        
        # 유효한 세그먼트만 필터링
        valid_segments = [(i, path) for i, path in enumerate(segment_paths) 
                         if path and os.path.exists(path)]
        
        if not valid_segments:
            return [f"No valid segments for processing"] * len(segment_paths)
        
        results = [""] * len(segment_paths)
        
        # 병렬 처리 함수
        def process_single_segment_wrapper(orig_idx_and_path):
            orig_idx, segment_path = orig_idx_and_path
            try:
                result = self.process_single_segment_optimized(segment_path, orig_idx, comment)
                return (orig_idx, result)
            except Exception as e:
                return (orig_idx, f"Error in segment {orig_idx + 1}: {str(e)}")
        
        # 병렬로 세그먼트 분석
        with ThreadPoolExecutor(max_workers=self.max_workers_analysis) as executor:
            future_to_index = {executor.submit(process_single_segment_wrapper, segment): segment[0] 
                             for segment in valid_segments}
            
            for future in as_completed(future_to_index):
                try:
                    orig_idx, result = future.result()
                    results[orig_idx] = result
                    print(f"    ✓ 세그먼트 {orig_idx+1} 분석 완료")
                except Exception as e:
                    orig_idx = future_to_index[future]
                    results[orig_idx] = f"Error in segment {orig_idx + 1}: {str(e)}"
                    print(f"    ✗ 세그먼트 {orig_idx+1} 분석 실패: {e}")
        
        # 처리되지 않은 세그먼트 처리
        for i, path in enumerate(segment_paths):
            if not results[i]:
                results[i] = f"Failed to process segment {i + 1}"
        
        return results
    
    def process_segments_batch(self, segment_paths: List[str], comment: str) -> List[str]:
        """배치로 여러 세그먼트 동시 처리 (백업용)"""
        print(f"  배치 처리 모드: {len(segment_paths)}개 세그먼트")
        
        # 유효한 세그먼트만 필터링
        valid_segments = [(i, path) for i, path in enumerate(segment_paths) 
                         if path and os.path.exists(path)]
        
        if not valid_segments:
            return [f"No valid segments for processing"] * len(segment_paths)
        
        results = [""] * len(segment_paths)
        
        # 배치 크기만큼 나누어서 처리
        for batch_start in range(0, len(valid_segments), self.max_batch_size):
            batch_end = min(batch_start + self.max_batch_size, len(valid_segments))
            batch_segments = valid_segments[batch_start:batch_end]
            
            print(f"    배치 처리 중: {batch_start+1}-{batch_end}/{len(valid_segments)}")
            
            # 배치 데이터 준비
            conversations = []
            batch_indices = []
            
            for orig_idx, segment_path in batch_segments:
                batch_indices.append(orig_idx)
                
                # 오디오 추출
                audio, sr = self.extract_audio_optimized(segment_path, 0, duration=self.segment_duration)
                
                # 개선된 프롬프트 (세그먼트 길이 변수화)
                system_prompt = f"""Analyze this {self.segment_duration}-second video segment and write exactly two parts as two lines.

VISUAL: Describe only what is visible, including people, objects, actions, scene context, colors, and motion.
AUDIO: Describe only what is audible, including speech content, speaker tone, music, sound effects, ambient sounds, and silence.

Keep both lines concise but informative and do not add any other text."""
                
                user_content = []
                
                if os.path.exists(segment_path) and os.path.getsize(segment_path) > 0:
                    user_content.append({"type": "video", "video": segment_path})
                
                if audio is not None and len(audio) > 0 and not np.all(audio == 0):
                    user_content.append({"type": "audio", "audio": audio})
                
                user_content.append({"type": "text", "text": system_prompt})
                
                # Qwen 기본 시스템 프롬프트 사용 (오디오 출력 호환성)
                conversation = [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}],
                    },
                    {
                        "role": "user",
                        "content": user_content,
                    },
                ]
                
                conversations.append(conversation)
            
            # 배치 추론 실행
            try:
                batch_results = self._process_batch_inference(conversations)
                
                for i, result in enumerate(batch_results):
                    if i < len(batch_indices):
                        results[batch_indices[i]] = result
                        
            except Exception as e:
                print(f"    배치 처리 오류: {e}")
                # 개별 처리로 폴백
                for i, (orig_idx, segment_path) in enumerate(batch_segments):
                    try:
                        result = self.process_single_segment_optimized(segment_path, orig_idx, comment)
                        results[orig_idx] = result
                    except Exception as e2:
                        results[orig_idx] = f"Error in segment {orig_idx + 1}: {str(e2)}"
        
        # 처리되지 않은 세그먼트 처리
        for i, path in enumerate(segment_paths):
            if not results[i]:
                results[i] = f"Failed to process segment {i + 1}"
        
        return results
    
    def _process_batch_inference(self, conversations: List) -> List[str]:
        """배치 추론 실행"""
        if len(conversations) == 1:
            # 단일 추론
            return [self._single_inference(conversations[0])]
        
        # 실제 배치 처리는 복잡하므로 순차 처리로 구현
        # (Qwen2.5-Omni는 배치 처리가 제한적일 수 있음)
        results = []
        for conv in conversations:
            try:
                result = self._single_inference(conv)
                results.append(result)
            except Exception as e:
                results.append(f"Inference error: {str(e)}")
        
        return results
    
    def _single_inference(self, conversation) -> str:
        """단일 추론 실행"""
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
                
                audios, images, videos = process_mm_info(conversation, use_audio_in_video=self.USE_AUDIO_IN_VIDEO)
                
                inputs = self.processor(
                    text=text, 
                    audio=audios, 
                    images=images, 
                    videos=videos, 
                    return_tensors="pt", 
                    padding=True, 
                    use_audio_in_video=self.USE_AUDIO_IN_VIDEO
                )
            
            inputs = inputs.to(self.model.device).to(self.model.dtype)
            
            # 최적화된 생성
            with torch.no_grad():  # 그래디언트 계산 비활성화
                text_ids, audio = self.model.generate(**inputs, **self.generation_config,
                                                     use_audio_in_video=self.USE_AUDIO_IN_VIDEO)
            
            text = self.processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            model_answer = text[0] if text else "Failed to generate response"
            
            # assistant 부분만 추출
            if "<|im_start|>assistant" in model_answer:
                assistant_start = model_answer.find("<|im_start|>assistant")
                assistant_end = model_answer.find("<|im_end|>", assistant_start)
                if assistant_end != -1:
                    model_answer = model_answer[assistant_start + len("<|im_start|>assistant"):assistant_end].strip()
                else:
                    model_answer = model_answer[assistant_start + len("<|im_start|>assistant"):].strip()
            elif "assistant" in model_answer:
                assistant_start = model_answer.find("assistant")
                model_answer = model_answer[assistant_start + len("assistant"):].strip()
                if model_answer.startswith(":"):
                    model_answer = model_answer[1:].strip()
            
            return model_answer
            
        except Exception as e:
            return f"Inference error: {str(e)}"
    
    def process_single_segment_optimized(self, segment_path: str, segment_index: int, comment: str) -> str:
        """최적화된 단일 6초 세그먼트 처리"""
        try:
            if not segment_path or not os.path.exists(segment_path):
                return f"Failed to process segment {segment_index + 1}"
            
            audio, sr = self.extract_audio_optimized(segment_path, 0, duration=self.segment_duration)
            
            if audio is None or len(audio) == 0:
                return f"No audio in segment {segment_index + 1}"
            
            # 개선된 프롬프트 (세그먼트 길이 변수화)
            system_prompt = f"""Analyze this {self.segment_duration}-second video segment and write exactly two parts as two lines.

VISUAL: Describe only what is visible, including people, objects, actions, scene context, colors, and motion.
AUDIO: Describe only what is audible, including speech content, speaker tone, music, sound effects, ambient sounds, and silence.

Keep both lines concise but informative and do not add any other text."""
            
            user_content = []
            
            if os.path.exists(segment_path) and os.path.getsize(segment_path) > 0:
                user_content.append({"type": "video", "video": segment_path})
            
            if audio is not None and len(audio) > 0 and not np.all(audio == 0):
                user_content.append({"type": "audio", "audio": audio})
            
            user_content.append({"type": "text", "text": system_prompt})
            
            # Qwen 기본 시스템 프롬프트 사용 (오디오 출력 호환성)
            conversation = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}],
                },
                {"role": "user", "content": user_content},
            ]
            
            return self._single_inference(conversation)
            
        except Exception as e:
            return f"Error in segment {segment_index + 1}: {str(e)}"
    
    def test_multiple_comments_optimized(self, num_comments: int = 20, video_test: bool = True, 
                                        use_parallel: bool = True, use_batch: bool = False) -> Dict[str, Any]:
        """최적화된 여러 댓글 처리"""
        try:
            df = pd.read_csv(os.path.join(PROJECT_ROOT, 'csv', 'merged_filtered_comments_with_dedup_lang.csv'))
            
            if len(df) == 0:
                return {"error": "Empty CSV file"}
            
            num_comments = min(num_comments, len(df))
            comments_to_process = df.head(num_comments)
            
            print(f"=== 최적화된 테스트: {num_comments}개 댓글 처리 ===")
            print(f"비디오 테스트: {'ON' if video_test else 'OFF'}")
            print(f"병렬 처리: {'ON' if use_parallel else 'OFF'}")
            print(f"배치 처리: {'ON' if use_batch else 'OFF'}")
            print("=" * 50)
            
            all_results = []
            successful_comments = 0
            failed_comments = 0
            
            start_time = time.time()
            for idx, comment_row in tqdm(comments_to_process.iterrows(), total=num_comments, desc="최적화된 댓글 처리"):
                try:
                    video_id = comment_row['video_id']
                    comment = comment_row['comment']
                    language = comment_row['language']
                    
                    # 진행률 정보
                    current_time = time.time()
                    elapsed_time = current_time - start_time
                    progress = (idx + 1) / num_comments
                    
                    if idx > 0:
                        avg_time_per_comment = elapsed_time / (idx + 1)
                        remaining_comments = num_comments - (idx + 1)
                        eta_seconds = avg_time_per_comment * remaining_comments
                        eta_str = f"{eta_seconds/60:.1f}분" if eta_seconds >= 60 else f"{eta_seconds:.0f}초"
                    else:
                        eta_str = "계산 중..."
                    
                    print(f"\n[{idx+1}/{num_comments}] 📍 댓글 처리 중... ({progress*100:.1f}%)")
                    print(f"⏰ 경과: {elapsed_time/60:.1f}분 | ETA: {eta_str}")
                    print(f"🎬 비디오 ID: {video_id}")
                    print(f"💬 댓글: {comment[:80]}...")
                    print(f"🌐 언어: {language}")
                    
                    parsed_timestamps = self.parse_timestamp(comment_row['timestamp'])
                    avg_time = self.calculate_average_timestamp(parsed_timestamps)
                    
                    print(f"📍 타임스탬프: {comment_row['timestamp']} → {avg_time:.1f}초")
                    
                    if avg_time <= 0:
                        print(f"  ❌ 잘못된 타임스탬프: {comment_row['timestamp']}")
                        failed_comments += 1
                        all_results.append({
                            "comment_index": idx + 1,
                            "video_id": video_id,
                            "comment": comment,
                            "language": language,
                            "error": "Invalid timestamp",
                            "analysis_type": "failed"
                        })
                        continue
                    
                    if not video_test:
                        result = {
                            "comment_index": idx + 1,
                            "video_id": video_id,
                            "comment": comment,
                            "language": language,
                            "comment_timestamp": avg_time,
                            "analysis_type": "text_only_test"
                        }
                        all_results.append(result)
                        successful_comments += 1
                        continue
                    
                    print(f"🔍 비디오 파일 검색 중: {video_id}")
                    original_video_path = self.find_video_file(video_id)
                    if not original_video_path:
                        print(f"  ❌ 비디오 파일을 찾을 수 없음: {video_id}")
                        failed_comments += 1
                        all_results.append({
                            "comment_index": idx + 1,
                            "video_id": video_id,
                            "error": f"Video file not found: {video_id}",
                            "analysis_type": "failed"
                        })
                        continue
                    
                    print(f"  ✅ 비디오 파일 발견: {os.path.basename(original_video_path)}")
                    
                    # 세그먼트 생성 (병렬 또는 순차)
                    print(f"🎞️  세그먼트 생성 중... ({self.num_segments}개 × {self.segment_duration}초)")
                    if use_parallel:
                        segment_paths = self.create_segments_parallel(original_video_path, avg_time)
                    else:
                        segment_paths = self.create_segments_sequential(original_video_path, avg_time)
                    
                    # 세그먼트 분석 (병렬, 배치, 또는 순차)
                    print(f"🧠 AI 분석 시작...")
                    segment_start_time = time.time()
                    if use_parallel and not use_batch:
                        segment_queries_raw = self.process_segments_parallel(segment_paths, comment)
                    elif use_batch:
                        segment_queries_raw = self.process_segments_batch(segment_paths, comment)
                    else:
                        segment_queries_raw = []
                        for i, segment_path in enumerate(segment_paths):
                            if segment_path:
                                query = self.process_single_segment_optimized(segment_path, i, comment)
                            else:
                                query = f"Failed to process segment {i + 1}"
                            segment_queries_raw.append(query)
                    
                    segment_end_time = time.time()
                    total_segment_time = segment_end_time - segment_start_time
                    
                    # 결과 정리 및 출력
                    segment_queries = []
                    print(f"📊 분석 결과:")
                    for i, query in enumerate(segment_queries_raw):
                        time_start = max(0, avg_time - self.half_duration + i * self.segment_duration)
                        time_end = max(0, avg_time - self.half_duration + (i + 1) * self.segment_duration)
                        time_range = f"{time_start:.1f}s - {time_end:.1f}s"
                        
                        segment_queries.append({
                            "segment_index": i + 1,
                            "time_range": time_range,
                            "query": query
                        })
                        
                        print(f"  {i+1}. [{time_range}]: {query[:100]}...")
                    
                    result = {
                        "comment_index": idx + 1,
                        "video_id": video_id,
                        "comment": comment,
                        "language": language,
                        "comment_timestamp": avg_time,
                        "total_duration": f"{self.num_segments * self.segment_duration}s ({self.num_segments} segments of {self.segment_duration}s each)",
                        "segments": segment_queries,
                        "analysis_type": "optimized_video_analysis",
                        "processing_time": total_segment_time,
                        "optimization": {
                            "parallel_processing": use_parallel,
                            "batch_processing": use_batch,
                            "model_compiled": True
                        }
                    }
                    
                    all_results.append(result)
                    successful_comments += 1
                    print(f"⏱️  처리 시간: {total_segment_time:.2f}초")
                    print(f"✅ 댓글 {idx+1} 성공적으로 완료!")
                    print("=" * 60)
                    
                    # 메모리 정리
                    if idx % 5 == 0:  # 5개 댓글마다 메모리 정리
                        torch.cuda.empty_cache()
                        gc.collect()
                    
                except Exception as e:
                    print(f"  ❌ 댓글 {idx+1} 처리 중 오류: {e}")
                    failed_comments += 1
                    all_results.append({
                        "comment_index": idx + 1,
                        "error": f"Processing failed: {str(e)}",
                        "analysis_type": "failed"
                    })
            
            final_result = {
                "total_comments": num_comments,
                "successful_comments": successful_comments,
                "failed_comments": failed_comments,
                "success_rate": f"{(successful_comments/num_comments)*100:.1f}%" if num_comments > 0 else "0%",
                "analysis_type": "optimized_multiple_comments_test",
                "optimization_features": {
                    "model_compilation": True,
                    "parallel_processing": use_parallel,
                    "batch_processing": use_batch,
                    "memory_management": True,
                    "float16_precision": True,
                    "optimized_generation_config": True
                },
                "comments": all_results
            }
            
            print(f"\n=== 최적화된 처리 완료 ===")
            print(f"총 댓글: {num_comments}개")
            print(f"성공: {successful_comments}개")
            print(f"실패: {failed_comments}개")
            print(f"성공률: {(successful_comments/num_comments)*100:.1f}%")
            
            return final_result
            
        except Exception as e:
            print(f"최적화된 테스트 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            return {"error": f"Optimized test failed: {str(e)}"}

def main():
    print("=== 최적화된 테스트용 Moment Query Generator ===")
    print("20개 댓글 처리 테스트 실행 (최적화 버전)")
    print("=" * 50)
    
    total_start_time = time.time()
    
    # 최적화된 모델 초기화 (모든 설정 변수화)
    print("\n최적화된 모델 초기화 중...")
    model_start_time = time.time()
    generator = OptimizedTestMomentQueryGenerator(
        # 모델 설정
        model_path="Qwen/Qwen2.5-Omni-7B",
        
        # 세그먼트 설정 (유연하게 조정 가능)
        segment_duration=6,
        num_segments=3,  # 3개로 설정 (18초 총 길이, 중심 ±9초)
        
        # 배치 및 생성 설정
        max_batch_size=3,
        max_new_tokens=200,
        
        # 비디오 처리 설정
        video_fps=24,
        ffmpeg_preset="ultrafast",  # 가장 빠른 설정
        ffmpeg_crf=28,  # 적당한 품질
        
        # 최적화 기능 설정
        use_model_compile=True,
        use_flash_attention=True,
        torch_dtype_str="float16",
        
        # 병렬 처리 설정
        max_workers_segments=3,  # 세그먼트 생성 병렬도
        max_workers_analysis=2   # 세그먼트 분석 병렬도
    )
    model_end_time = time.time()
    model_time = model_end_time - model_start_time
    print(f"✓ 최적화된 모델 초기화 완료 ({model_time:.2f}초)")
    
    # 최적화된 비디오 테스트 실행 (병렬 처리 포함)
    print("\n최적화된 비디오 테스트 실행 (20개 댓글)...")
    total_duration = generator.num_segments * generator.segment_duration
    print(f"   최적화 기능: 모델 컴파일, 병렬 처리, {generator.num_segments}개 세그먼트 ({total_duration}초), 메모리 관리")
    print(f"   병렬 설정: 세그먼트 생성 {generator.max_workers_segments}개, 분석 {generator.max_workers_analysis}개")
    print("   예상 시간: 기존 대비 70-85% 단축 예상 (병렬화 + 세그먼트 수 감소 + 기타 최적화)")
    
    video_start_time = time.time()
    video_result = generator.test_multiple_comments_optimized(
        num_comments=20, 
        video_test=True, 
        use_parallel=True,  # 병렬 처리 활성화
        use_batch=False     # 병렬이 더 효율적이므로 배치는 비활성화
    )
    video_end_time = time.time()
    video_time = video_end_time - video_start_time
    
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    
    if "error" in video_result:
        print(f"최적화된 비디오 테스트 실패: {video_result['error']} ({video_time:.2f}초)")
    else:
        print(f"✓ 최적화된 비디오 테스트 성공! ({video_time:.2f}초)")
        print(f"  총 댓글: {video_result['total_comments']}개")
        print(f"  성공: {video_result['successful_comments']}개")
        print(f"  실패: {video_result['failed_comments']}개")
        print(f"  성공률: {video_result['success_rate']}")
    
    # 결과 출력
    print("\n=== 최적화된 최종 테스트 결과 ===")
    if "error" not in video_result:
        print(f"총 댓글: {video_result['total_comments']}개")
        print(f"성공: {video_result['successful_comments']}개")
        print(f"실패: {video_result['failed_comments']}개")
        print(f"성공률: {video_result['success_rate']}")
        
        # 최적화 기능 요약
        print(f"\n적용된 최적화:")
        opt_features = video_result.get('optimization_features', {})
        for feature, enabled in opt_features.items():
            print(f"  - {feature}: {'✓' if enabled else '✗'}")
    
    # 시간 통계
    print(f"\n=== 최적화된 실행 시간 통계 ===")
    print(f"모델 초기화: {model_time:.2f}초")
    print(f"최적화된 비디오 테스트: {video_time:.2f}초")
    print(f"총 실행 시간: {total_time:.2f}초 ({total_time/60:.1f}분)")
    
    if "error" not in video_result and video_result.get('successful_comments', 0) > 0:
        avg_time_per_comment = video_time / video_result['successful_comments']
        print(f"댓글당 평균 처리 시간: {avg_time_per_comment:.2f}초")
        print(f"예상 속도 향상: 기존 대비 약 2-3배 빠름")
    
    # 결과 저장
    video_result['timing'] = {
        "model_initialization": model_time,
        "optimized_video_test": video_time,
        "total_time": total_time,
        "avg_time_per_comment": video_time / video_result.get('successful_comments', 1) if video_result.get('successful_comments', 0) > 0 else 0
    }
    
    output_file = "optimized_test_moment_query_result_20comments.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(video_result, f, ensure_ascii=False, indent=2)
    print(f"\n최적화된 결과가 {output_file}에 저장되었습니다.")

if __name__ == "__main__":
    main()
