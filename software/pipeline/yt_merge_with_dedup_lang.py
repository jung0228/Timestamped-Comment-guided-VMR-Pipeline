import pandas as pd
import glob
import re
import os
from langdetect import detect, LangDetectException

# Project root is parent of scripts/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)


def contains_timestamp(text):
    """
    텍스트에 타임스탬프가 포함되어 있는지 확인합니다.
    0:00, 00:00, 12:34, 1:23:45 등 다양한 타임스탬프 패턴을 지원합니다.
    """
    # NaN이나 None 값 처리
    if pd.isna(text) or not isinstance(text, str):
        return False
    
    pattern = r"(?<!\d)(?:[0-5]?\d:)?[0-5]?\d:[0-5]\d(?!\d)"
    return re.search(pattern, text) is not None

def extract_timestamps(text):
    """
    텍스트에서 타임스탬프 패턴을 추출합니다.
    """
    # NaN이나 None 값 처리
    if pd.isna(text) or not isinstance(text, str):
        return []
    
    pattern = r"(?<!\d)(?:[0-5]?\d:)?[0-5]?\d:[0-5]\d(?!\d)"
    timestamps = re.findall(pattern, text)
    return timestamps

def detect_language(text):
    """
    텍스트의 언어를 감지합니다.
    감지 실패 시 'unknown'을 반환합니다.
    """
    try:
        # NaN이나 None 값 처리
        if pd.isna(text) or not isinstance(text, str):
            return 'unknown'
        
        # 텍스트가 너무 짧으면 감지 불가
        if len(text.strip()) < 3:
            return 'unknown'
        
        # 타임스탬프와 특수문자 제거 후 언어 감지
        clean_text = re.sub(r'[0-9]+:[0-9]+', '', text)  # 타임스탬프 제거
        clean_text = re.sub(r'[^\w\s]', ' ', clean_text)  # 특수문자 제거
        clean_text = clean_text.strip()
        
        if len(clean_text) < 3:
            return 'unknown'
            
        language = detect(clean_text)
        return language
    except LangDetectException:
        return 'unknown'
    except Exception:
        return 'unknown'

def remove_duplicates(df):
    """
    중복된 댓글을 제거합니다.
    comment 컬럼을 기준으로 중복을 찾고, likes가 높은 것을 유지합니다.
    """
    # comment 컬럼을 기준으로 중복 제거 (likes가 높은 것 유지)
    df_no_duplicates = df.sort_values('likes', ascending=False).drop_duplicates(subset=['comment'], keep='first')
    
    return df_no_duplicates

def filter_by_language(df, target_languages=None):
    """
    특정 언어의 댓글만 필터링합니다.
    target_languages가 None이면 모든 언어를 유지합니다.
    """
    if target_languages is None or 'language' not in df.columns:
        return df
    
    # 지정된 언어만 유지
    df_filtered = df[df['language'].isin(target_languages)]
    
    return df_filtered


def load_video_mapping():
    """
    video_id_mapping.csv 파일을 읽어서 비디오 ID와 채널명, 카테고리 정보를 반환합니다.
    """
    try:
        mapping_df = pd.read_csv(os.path.join(PROJECT_ROOT, "csv", "video_id_mapping.csv"))
        # video_id를 키로 하는 딕셔너리 생성
        video_mapping = {}
        for _, row in mapping_df.iterrows():
            if pd.notna(row['video_id']):
                video_mapping[row['video_id']] = {
                    'channel_name': row['channel_name'],
                    'category': row['category']
                }
        return video_mapping
    except FileNotFoundError:
        print("⚠️ csv/video_id_mapping.csv 파일을 찾을 수 없습니다. 기본 매핑을 사용합니다.")
        return {}
    except Exception as e:
        print(f"⚠️ csv/video_id_mapping.csv 파일 읽기 오류: {e}. 기본 매핑을 사용합니다.")
        return {}

def merge_filtered_files_with_dedup_lang(target_languages=None, top_n=20):
    """
    필터링된 파일들을 병합하고 중복 제거, 언어 식별을 수행합니다.
    
    Args:
        target_languages (list): 필터링할 언어 목록 (예: ['ko', 'en']). None이면 모든 언어 유지
        top_n (int): 각 파일에서 추출할 상위 댓글 수
    """
    # csv/video_id_mapping.csv에서 정보 로드
    video_mapping = load_video_mapping()
    
    # Comments folder (under project root)
    comments_dir = os.path.join(PROJECT_ROOT, "Comments")
    if not os.path.exists(comments_dir):
        print(f"❌ {comments_dir} 폴더를 찾을 수 없습니다.")
        return None
    
    # comments 폴더의 모든 CSV 파일 찾기
    filtered_files = glob.glob(os.path.join(comments_dir, "*_comments.csv"))
    
    print(f"📁 찾은 파일들: {filtered_files}")
    
    # 통계를 위한 변수들
    total_stats = {
        'original_comments': 0,
        'timestamp_comments': 0,
        'expanded_comments': 0,
        'after_dedup': 0,
        'after_lang_filter': 0,
        'final_selected': 0,
        'duplicates_removed': 0,
        'lang_filtered': 0
    }
    
    all_data = []
    
    for file_path in filtered_files:
        # 파일명에서 video_id 추출 (예: "myO8fxhDRW0_comments.csv" -> "myO8fxhDRW0")
        filename = os.path.basename(file_path)
        video_id = filename.replace("_comments.csv", "")
        
        # CSV 파일 읽기
        df = pd.read_csv(file_path)
        
        # 통계 누적
        total_stats['original_comments'] += len(df)
        
        # video_id 컬럼 추가
        df['video_id'] = video_id
        
        # 채널명과 카테고리 정보 추가
        if video_id in video_mapping:
            df['channel_name'] = video_mapping[video_id]['channel_name']
            df['category'] = video_mapping[video_id]['category']
        else:
            print(f"⚠️ {video_id}에 대한 매핑 정보를 찾을 수 없습니다.")
            df['channel_name'] = 'Unknown'
            df['category'] = 'Unknown'
        
        # comment 컬럼 처리 (comment가 없으면 다른 컬럼 찾기)
        if 'comment' not in df.columns:
            # username, time, likes, video_id, channel_name, category를 제외한 컬럼을 comment로 사용
            available_cols = [col for col in df.columns if col not in ['username', 'time', 'likes', 'video_id', 'channel_name', 'category']]
            if available_cols:
                df['comment'] = df[available_cols[0]]
        
        # 댓글에서 타임스탬프 추출
        df['timestamp'] = df['comment'].apply(extract_timestamps)
        
        # 타임스탬프가 있는 댓글만 필터링
        df_with_timestamp = df[df['timestamp'].apply(lambda x: len(x) > 0)].copy()
        total_stats['timestamp_comments'] += len(df_with_timestamp)
        
        if len(df_with_timestamp) > 0:
            # timestamp가 여러 개인 경우 각각을 별도 행으로 분리
            expanded_rows = []
            
            for idx, row in df_with_timestamp.iterrows():
                timestamps = row['timestamp']
                
                # 각 timestamp에 대해 별도 행 생성
                for timestamp in timestamps:
                    new_row = row.copy()
                    new_row['timestamp'] = [timestamp]  # 단일 timestamp로 변경
                    expanded_rows.append(new_row)
            
            # 분리된 데이터로 새로운 DataFrame 생성
            df_expanded = pd.DataFrame(expanded_rows)
            total_stats['expanded_comments'] += len(df_expanded)
            
            # 숫자 변환
            df_expanded["likes"] = pd.to_numeric(df_expanded["likes"], errors="coerce")
            
            # 중복 제거
            before_dedup = len(df_expanded)
            df_expanded = remove_duplicates(df_expanded)
            after_dedup = len(df_expanded)
            total_stats['duplicates_removed'] += (before_dedup - after_dedup)
            total_stats['after_dedup'] += after_dedup
            
            # 언어 식별 추가
            df_expanded['language'] = df_expanded['comment'].apply(detect_language)
            
            # 언어 필터링 (옵션)
            if target_languages:
                before_lang_filter = len(df_expanded)
                df_expanded = filter_by_language(df_expanded, target_languages)
                after_lang_filter = len(df_expanded)
                total_stats['lang_filtered'] += (before_lang_filter - after_lang_filter)
                total_stats['after_lang_filter'] += after_lang_filter
            else:
                total_stats['after_lang_filter'] += len(df_expanded)
            
            # 상위 N개 댓글만 추출
            df_top = df_expanded.sort_values(by="likes", ascending=False).head(top_n).copy()
            total_stats['final_selected'] += len(df_top)
            
            # 불필요한 컬럼 제거
            columns_to_remove = ["username", "time", "filename"]
            for col in columns_to_remove:
                if col in df_top.columns:
                    df_top = df_top.drop(col, axis=1)
            
            all_data.append(df_top)
    
    # 모든 데이터 합치기
    if all_data:
        merged_df = pd.concat(all_data, ignore_index=True)
        print(f"\n🎯 최종 합쳐진 데이터: {len(merged_df)} 행")
        
        # 컬럼 순서 재정렬 (존재하는 컬럼만)
        final_columns = []
        for col in ["video_id", "channel_name", "category", "timestamp", "comment", "language", "likes"]:
            if col in merged_df.columns:
                final_columns.append(col)
        
        merged_df = merged_df[final_columns]
        
        # 전체 데이터 언어 통계
        if 'language' in merged_df.columns:
            total_language_counts = merged_df['language'].value_counts()
            print(f"📊 전체 언어별 댓글 수: {dict(total_language_counts)}")
        
        # 채널별 통계
        if 'channel_name' in merged_df.columns:
            channel_counts = merged_df['channel_name'].value_counts()
            print(f"📺 채널별 댓글 수: {dict(channel_counts)}")
        
        # 카테고리별 통계
        if 'category' in merged_df.columns:
            category_counts = merged_df['category'].value_counts()
            print(f"🏷️ 카테고리별 댓글 수: {dict(category_counts)}")
        
        # 결과 저장
        output_file = os.path.join(PROJECT_ROOT, "csv", "merged_filtered_comments_with_dedup_lang.csv")
        os.makedirs(os.path.join(PROJECT_ROOT, "csv"), exist_ok=True)
        merged_df.to_csv(output_file, index=False)
        print(f"💾 결과가 {output_file}에 저장되었습니다.")
        
        # 전체 처리 통계 출력
        print("\n" + "="*60)
        print("📊 전체 처리 통계 요약")
        print("="*60)
        print(f"📝 원본 댓글 수: {total_stats['original_comments']:,}개")
        print(f"⏰ 타임스탬프 포함 댓글: {total_stats['timestamp_comments']:,}개")
        print(f"🔄 타임스탬프 분리 후: {total_stats['expanded_comments']:,}개")
        print(f"🗑️ 중복 제거: {total_stats['duplicates_removed']:,}개 제거")
        print(f"🌍 언어 필터링: {total_stats['lang_filtered']:,}개 제거")
        print(f"📊 상위 {top_n}개 선별: {total_stats['final_selected']:,}개")
        print(f"🏆 최종 결과: {total_stats['final_selected']:,}개")
        print("="*60)
        
        # 첫 몇 행 출력
        print("\n📋 첫 5행:")
        print(merged_df.head())
        
        return merged_df
    else:
        print("❌ 처리할 파일이 없습니다.")
        return None

def main():
    """
    메인 함수 - 영어만 필터링하여 실행
    """
    print("🚀 YouTube 댓글 병합 및 처리 시작 (영어만)")
    print("=" * 50)
    
    # 영어만 필터링 + 중복 제거
    print("\n📝 영어 댓글만 필터링 + 중복 제거")
    merge_filtered_files_with_dedup_lang(
        target_languages=['en'], 
        top_n=20
    )

if __name__ == "__main__":
    main()
