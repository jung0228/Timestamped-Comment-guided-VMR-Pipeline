#!/usr/bin/env python3
"""
video_id_mapping.csv에서 각 채널별로 상위 5개만 선택하여 다운로드합니다.
파일명은 video_id.mp4로 저장됩니다.
"""

import csv
import subprocess
import os
from collections import defaultdict

# Project root is parent of scripts/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
MAPPING_FILE = os.path.join(PROJECT_ROOT, 'csv', 'video_id_mapping.csv')
COMMENTS_DIR = os.path.join(PROJECT_ROOT, 'Comments')


def get_top5_per_channel(mapping_file=None, filter_channels=None):
    """
    각 채널별로 상위 5개 영상만 선택
    
    Args:
        mapping_file: video_id_mapping.csv 파일 경로
        filter_channels: 필터링할 채널 리스트 (None이면 모든 채널)
    """
    if mapping_file is None:
        mapping_file = MAPPING_FILE
    
    if filter_channels:
        print(f"📋 video_id_mapping.csv를 읽는 중 (채널 필터: {', '.join(filter_channels)})...\n")
    else:
        print("📋 video_id_mapping.csv를 읽는 중...\n")
    
    # 채널별로 그룹화
    channel_videos = defaultdict(list)
    
    with open(mapping_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            channel_name = row['channel_name']
            
            # 채널 필터링
            if filter_channels and channel_name not in filter_channels:
                continue
                
            channel_videos[channel_name].append({
                'video_id': row['video_id'],
                'channel_name': channel_name,
                'category': row['category']
            })
    
    # 각 채널별로 상위 5개만 선택
    top5_videos = []
    for channel, videos in channel_videos.items():
        top5 = videos[:5]  # 앞에서 5개 (이미 조회수 순으로 정렬되어 있다고 가정)
        top5_videos.extend(top5)
        print(f"   {channel:<30} : {len(videos)}개 → {len(top5)}개 선택")
    
    if not top5_videos:
        print("⚠️  선택된 영상이 없습니다.")
        if filter_channels:
            print(f"   필터링된 채널: {', '.join(filter_channels)}")
    
    print(f"\n✅ 총 {len(top5_videos)}개 영상을 선택했습니다.\n")
    return top5_videos


def download_videos(videos, base_dir="Videos", resolution="360"):
    """영상들을 다운로드"""
    print(f"📥 영상 다운로드 시작...")
    print(f"⚙️  해상도: {resolution}p")
    print(f"📁 저장 경로: {base_dir}/\n")
    
    # 기본 폴더 생성
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    success_count = 0
    skip_count = 0
    fail_count = 0
    total = len(videos)
    
    for idx, video in enumerate(videos, 1):
        video_id = video['video_id']
        channel_name = video['channel_name']
        category = video['category']
        
        # 카테고리/채널 폴더 생성
        output_dir = os.path.join(base_dir, category, channel_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # 파일명: video_id.mp4
        output_file = os.path.join(output_dir, f"{video_id}.mp4")
        
        print(f"[{idx}/{total}] {channel_name} - {video_id}")
        
        # 댓글 파일이 있으면 이미 다운로드된 것으로 간주하고 스킵
        comment_file = os.path.join(COMMENTS_DIR, f"{video_id}_comments.csv")
        if os.path.exists(comment_file):
            print(f"   ⏭️  댓글이 이미 있어서 스킵합니다. (댓글 파일: {comment_file})")
            skip_count += 1
            continue
        
        # 이미 파일이 존재하는지 확인
        if os.path.exists(output_file):
            print(f"   ⏭️  이미 존재합니다. 건너뜁니다.")
            skip_count += 1
            continue
        
        # yt-dlp 명령어 (해상도 필터를 더 유연하게)
        cmd = [
            'yt-dlp',
            '-f', f'bestvideo[height<={resolution}]+bestaudio/best[height<={resolution}]/best',
            '--no-write-thumbnail',
            '--no-playlist',
            '-o', output_file,
            f'https://www.youtube.com/watch?v={video_id}'
        ]
        
        try:
            print(f"   🔄 다운로드 중...")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0 and os.path.exists(output_file):
                file_size = os.path.getsize(output_file)
                print(f"   ✅ 완료: {file_size / (1024*1024):.1f}MB")
                success_count += 1
            else:
                print(f"   ❌ 다운로드 실패 (종료 코드: {result.returncode})")
                if result.stderr:
                    # 에러 메시지의 마지막 몇 줄만 표시
                    error_lines = result.stderr.strip().split('\n')
                    if len(error_lines) > 5:
                        print(f"   에러 (마지막 5줄):")
                        for line in error_lines[-5:]:
                            print(f"      {line}")
                    else:
                        print(f"   에러: {result.stderr.strip()}")
                if result.stdout:
                    # stdout의 마지막 몇 줄도 표시
                    stdout_lines = result.stdout.strip().split('\n')
                    if len(stdout_lines) > 3:
                        print(f"   출력 (마지막 3줄):")
                        for line in stdout_lines[-3:]:
                            print(f"      {line}")
                fail_count += 1
                
        except subprocess.TimeoutExpired:
            print(f"   ❌ 타임아웃: 다운로드가 10분을 초과했습니다.")
            fail_count += 1
        except Exception as e:
            print(f"   ❌ 예외 발생: {e}")
            import traceback
            print(f"   {traceback.format_exc()}")
            fail_count += 1
        
        print()
    
    # 최종 결과
    print("=" * 60)
    print("🎉 다운로드 완료!")
    print(f"✅ 성공: {success_count}/{total}개")
    print(f"⏭️  건너뜀: {skip_count}/{total}개 (댓글 있음 또는 이미 존재)")
    print(f"❌ 실패: {fail_count}/{total}개")
    print("=" * 60)


def main():
    import argparse
    
    # 기본 저장 경로: data/videos
    default_output_dir = 'data/videos'
    
    parser = argparse.ArgumentParser(description='각 채널별 상위 5개 영상만 다운로드')
    parser.add_argument('--resolution', choices=['360', '480', '720', '1080'], 
                        default='360', help='영상 해상도 (기본: 360p)')
    parser.add_argument('--output-dir', default=default_output_dir, 
                        help=f'저장 폴더 (기본: {default_output_dir})')
    parser.add_argument('--channels', nargs='+', 
                        help='다운로드할 채널 이름들 (예: --channels OfficialGrahamNorton JimmyKimmelLive)')
    
    args = parser.parse_args()
    
    # 1. 각 채널별 상위 5개 선택
    top5_videos = get_top5_per_channel(filter_channels=args.channels)
    
    if not top5_videos:
        print("❌ 다운로드할 영상이 없습니다.")
        return
    
    # 2. 다운로드
    download_videos(top5_videos, args.output_dir, args.resolution)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"❌ 예상치 못한 오류: {e}")
        import traceback
        traceback.print_exc()
