#!/usr/bin/env python3
"""
video_id_mapping.csv에서 각 채널별로 상위 5개 영상의 댓글만 크롤링합니다.
"""

import csv
import os
import time
from youtube_comment_downloader import YoutubeCommentDownloader
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# Project root is parent of scripts/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
MAPPING_FILE = os.path.join(PROJECT_ROOT, 'csv', 'video_id_mapping.csv')

# 병렬 처리를 위한 락
print_lock = Lock()


def get_top5_per_channel(mapping_file=None):
    """각 채널별로 상위 5개 영상만 선택"""
    if mapping_file is None:
        mapping_file = MAPPING_FILE
    print("📋 video_id_mapping.csv를 읽는 중...\n")
    
    # 채널별로 그룹화
    channel_videos = defaultdict(list)
    
    with open(mapping_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            channel_videos[row['channel_name']].append({
                'video_id': row['video_id'],
                'channel_name': row['channel_name'],
                'category': row['category']
            })
    
    # 각 채널별로 상위 5개만 선택
    top5_videos = []
    for channel, videos in channel_videos.items():
        top5 = videos[:5]  # 앞에서 5개
        top5_videos.extend(top5)
        print(f"   {channel:<30} : {len(videos)}개 → {len(top5)}개 선택")
    
    print(f"\n✅ 총 {len(top5_videos)}개 영상의 댓글을 크롤링합니다.\n")
    return top5_videos


def crawl_comments(video_id, output_csv, max_comments=None, video_info=None):
    """댓글을 크롤링하여 CSV 파일에 저장"""
    try:
        downloader = YoutubeCommentDownloader()
        comments = downloader.get_comments_from_url(f"https://www.youtube.com/watch?v={video_id}")
        
        comment_count = 0
        with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['username', 'comment', 'time', 'likes', 'reply_count'])
            
            for comment in comments:
                if max_comments and comment_count >= max_comments:
                    break
                    
                writer.writerow([
                    comment.get('author', ''),
                    comment.get('text', '').replace('\n', ' ').replace('\r', ''),
                    comment.get('time', ''),
                    comment.get('votes', ''),
                    comment.get('reply_count', '')
                ])
                comment_count += 1
        
        return comment_count
        
    except Exception as e:
        with print_lock:
            if video_info:
                print(f"      ❌ {video_info}: {str(e)}")
            else:
                print(f"      ❌ 오류: {str(e)}")
        return 0


def process_video(video, output_dir, max_comments, total, idx):
    """단일 영상 댓글 크롤링 처리"""
    video_id = video['video_id']
    channel_name = video['channel_name']
    category = video['category']
    
    output_csv = os.path.join(output_dir, f"{video_id}_comments.csv")
    video_info = f"{channel_name} - {video_id}"
    
    with print_lock:
        print(f"[{idx}/{total}] {video_info}")
    
    # 이미 파일이 존재하는지 확인
    if os.path.exists(output_csv):
        with print_lock:
            print(f"   ⏭️  이미 존재합니다. 건너뜁니다.")
        return {'status': 'skip', 'video_id': video_id}
    
    # 댓글 크롤링
    with print_lock:
        print(f"   🔄 댓글 크롤링 중...")
    
    comment_count = crawl_comments(video_id, output_csv, max_comments, video_info)
    
    if comment_count > 0:
        with print_lock:
            print(f"   ✅ 완료: {comment_count:,}개 댓글 저장")
        return {'status': 'success', 'video_id': video_id, 'comment_count': comment_count}
    else:
        with print_lock:
            print(f"   ❌ 실패: 댓글을 가져올 수 없습니다")
        return {'status': 'fail', 'video_id': video_id}


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='각 채널별 상위 5개 영상의 댓글만 크롤링')
    parser.add_argument('--max-comments', type=int, default=None, help='영상당 최대 댓글 수 (기본: 무제한)')
    parser.add_argument('--output-dir', default=None, help='댓글 저장 폴더 (기본: 프로젝트 루트의 Comments)')
    parser.add_argument('--workers', type=int, default=5, help='병렬 처리 워커 수 (기본: 5)')
    
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = os.path.join(PROJECT_ROOT, 'Comments')
    
    # 댓글 저장 폴더 생성
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"📁 '{args.output_dir}' 폴더를 생성했습니다.\n")
    
    # 1. 각 채널별 상위 5개 선택
    top5_videos = get_top5_per_channel()
    
    # 2. 댓글 크롤링 (병렬 처리)
    print("=" * 60)
    print(f"💬 댓글 크롤링 시작 (병렬 처리: {args.workers}개 워커)\n")
    
    success_count = 0
    skip_count = 0
    fail_count = 0
    total = len(top5_videos)
    
    # 병렬 처리
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = []
        for idx, video in enumerate(top5_videos, 1):
            future = executor.submit(process_video, video, args.output_dir, args.max_comments, total, idx)
            futures.append(future)
            # API 제한 고려: 약간의 딜레이
            time.sleep(0.5)
        
        # 결과 수집
        for future in as_completed(futures):
            result = future.result()
            if result['status'] == 'success':
                success_count += 1
            elif result['status'] == 'skip':
                skip_count += 1
            else:
                fail_count += 1
    
    # 최종 결과
    print("=" * 60)
    print("🎉 댓글 크롤링 완료!")
    print(f"✅ 성공: {success_count}/{total}개")
    print(f"⏭️  건너뜀: {skip_count}/{total}개 (이미 존재)")
    print(f"❌ 실패: {fail_count}/{total}개")
    print("=" * 60)
    
    # 생성된 파일 목록
    if success_count > 0:
        print(f"\n📁 생성된 파일들 ({args.output_dir}/ 폴더):")
        for video in top5_videos:
            video_id = video['video_id']
            csv_file = os.path.join(args.output_dir, f"{video_id}_comments.csv")
            if os.path.exists(csv_file):
                file_size = os.path.getsize(csv_file)
                with open(csv_file, 'r', encoding='utf-8') as f:
                    line_count = sum(1 for _ in f) - 1  # 헤더 제외
                print(f"   - {video_id}_comments.csv ({line_count:,}개 댓글, {file_size:,} bytes)")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"❌ 예상치 못한 오류: {e}")
        import traceback
        traceback.print_exc()
