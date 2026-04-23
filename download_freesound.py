"""
freesound.org API로 드럼 샘플 다운로드
- 6클래스: kick / snare / hihat / tom / crash / ride
- 클래스당 최대 400개 목표
- 0.5초(22050 샘플)로 잘라서 dataset/{label}/ 에 저장
"""

import os
import warnings
import requests
import urllib3
import librosa
import soundfile as sf
import numpy as np

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ---------------------------------------------------------------
# 여기에 API key 입력
# ---------------------------------------------------------------
API_KEY = os.environ.get("FREESOUND_API_KEY", "YOUR_API_KEY")

SR        = 22050
N_SAMPLES = int(SR * 0.5)
TARGET    = 400   # 클래스당 목표 개수

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR  = os.path.join(BASE_DIR, 'dataset')
SEARCH_URL   = "https://freesound.org/apiv2/search/text/"

SEARCH_TAGS = {
    'kick' : ['kick drum one shot', 'bass drum hit', 'kick one shot'],
    'snare': ['snare drum one shot', 'snare hit', 'snare one shot'],
    'hihat': ['hi-hat closed one shot', 'hi-hat open one shot', 'hihat hit'],
    'tom'  : ['tom drum hit', 'floor tom hit', 'rack tom one shot'],
    'crash': ['crash cymbal hit', 'crash one shot'],
    'ride' : ['ride cymbal hit', 'ride one shot'],
}


def search_sounds(query, page=1, page_size=150):
    params = {
        'query'    : query,
        'filter'   : 'duration:[0.05 TO 3.0]',
        'fields'   : 'id,name,duration,previews',
        'page_size': page_size,
        'page'     : page,
        'token'    : API_KEY,
    }
    r = requests.get(SEARCH_URL, params=params, timeout=15)
    r.raise_for_status()
    return r.json().get('results', [])


def download_preview(preview_url, save_path):
    """MP3 미리보기 다운로드 후 0.5초 wav로 변환"""
    tmp_path = save_path.replace('.wav', '_tmp.mp3')
    try:
        r = requests.get(preview_url, timeout=20, verify=False)
        r.raise_for_status()
        with open(tmp_path, 'wb') as f:
            f.write(r.content)

        y, _ = librosa.load(tmp_path, sr=SR, mono=True)
        y    = librosa.util.fix_length(y, size=N_SAMPLES)
        sf.write(save_path, y, SR)
        return True
    except Exception as e:
        print(f"    오류: {e}")
        return False
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def download_class(label, queries):
    save_dir = os.path.join(DATASET_DIR, label)
    os.makedirs(save_dir, exist_ok=True)

    # 이미 있는 freesound 파일 수 확인
    existing = len([f for f in os.listdir(save_dir) if f.startswith('fs_')])
    print(f"\n[{label}]  기존 {existing}개")
    if existing >= TARGET:
        print(f"  → 이미 {TARGET}개 이상. 건너뜀.")
        return

    idx    = existing
    ok     = 0
    seen   = set()

    for query in queries:
        if idx >= TARGET:
            break
        print(f"  검색: '{query}'")

        try:
            results = search_sounds(query)
        except Exception as e:
            print(f"  검색 실패: {e}")
            continue

        for sound in results:
            if idx >= TARGET:
                break
            sid = sound['id']
            if sid in seen:
                continue
            seen.add(sid)

            preview = sound.get('previews', {}).get('preview-hq-mp3') or \
                      sound.get('previews', {}).get('preview-lq-mp3')
            if not preview:
                continue

            save_path = os.path.join(save_dir, f"fs_{label}_{idx:04d}.wav")
            if os.path.exists(save_path):
                idx += 1
                continue

            success = download_preview(preview, save_path)
            if success:
                idx += 1
                ok  += 1
                if ok % 20 == 0:
                    print(f"    {idx}개 완료...")

    print(f"  → 완료: {ok}개 추가 (총 {idx}개)")


def main():
    if API_KEY == "YOUR_API_KEY":
        print("API_KEY를 입력해주세요. (이 파일 상단)")
        return

    print(f"저장 위치: {DATASET_DIR}")
    print(f"목표: 클래스당 {TARGET}개\n")

    for label, queries in SEARCH_TAGS.items():
        download_class(label, queries)

    # 최종 현황
    print("\n" + "─" * 40)
    print("최종 데이터 현황:")
    for label in SEARCH_TAGS:
        d = os.path.join(DATASET_DIR, label)
        if os.path.exists(d):
            total = len([f for f in os.listdir(d) if f.endswith('.wav')])
            fs    = len([f for f in os.listdir(d) if f.startswith('fs_')])
            print(f"  {label:<8} 총 {total}개  (freesound {fs}개)")
        else:
            print(f"  {label:<8} 폴더 없음")
    print("─" * 40)


if __name__ == '__main__':
    main()
