"""
e-GMD 데이터셋 다운로드 스크립트
- Acoustic Kit만 받음 (43개 kit 버전 중 하나, 용량 절약)
- drummer1, drummer7만 받음 (충분한 데이터량)
- WAV + MIDI 쌍으로 받음
"""

import csv
import os
import requests
import time

BASE_URL   = "https://storage.googleapis.com/magentadata/datasets/e-gmd/v1.0.0"
CSV_PATH   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "e-gmd-v1.0.0.csv")
SAVE_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "e-gmd")

# 받을 드러머 / kit 필터
TARGET_DRUMMERS = {'drummer1', 'drummer7'}
TARGET_KIT      = 'Acoustic Kit'


def download_file(url, save_path):
    """파일 하나 다운로드. 이미 있으면 건너뜀."""
    if os.path.exists(save_path):
        return 'skip'

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    try:
        r = requests.get(url, timeout=30, stream=True)
        r.raise_for_status()
        with open(save_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        return 'ok'
    except Exception as e:
        print(f"\n  오류: {url}\n  {e}")
        return 'err'


def main():
    # CSV 읽고 필터링
    with open(CSV_PATH, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    targets = [
        r for r in rows
        if r['drummer'] in TARGET_DRUMMERS and r['kit_name'] == TARGET_KIT
    ]

    print(f"다운로드 대상: {len(targets)}개 (WAV + MIDI = {len(targets)*2}개 파일)")
    print(f"저장 위치: {SAVE_DIR}")
    print()

    ok = skip = err = 0

    for i, row in enumerate(targets, 1):
        for key in ('audio_filename', 'midi_filename'):
            rel_path  = row[key]
            url       = f"{BASE_URL}/{rel_path}"
            save_path = os.path.join(SAVE_DIR, rel_path)
            result    = download_file(url, save_path)

            if result == 'ok':   ok   += 1
            elif result == 'skip': skip += 1
            else:                err  += 1

        # 진행률 표시
        if i % 10 == 0 or i == len(targets):
            print(f"  [{i:4d}/{len(targets)}]  완료 {ok}  건너뜀 {skip}  오류 {err}")

    print()
    print(f"완료! 받음 {ok}개 / 건너뜀 {skip}개 / 오류 {err}개")
    print(f"저장 위치: {SAVE_DIR}")


if __name__ == '__main__':
    main()
