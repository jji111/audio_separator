# Freesound 데이터셋 추가 계획

## 목표
freesound.org API로 kick/snare/hihat/tom/crash/ride 6클래스 샘플을 추가로 확보해서 모델 일반화 성능 향상

IDMT는 kick/snare/hihat만 있고 같은 킷으로 녹음된 거라 다양성이 부족함 → freesound로 다양한 킷/녹음 환경 샘플 추가

---

## API Key 발급

1. https://freesound.org 회원가입
2. https://freesound.org/apiv2/apply 접속
3. Application name: drum-extractor, Description: 드럼 분류 모델 학습용
4. 발급된 `Client ID`와 `Client secret` 메모

---

## 검색 태그 전략

| 클래스 | 검색 태그 |
|--------|----------|
| kick   | `kick drum`, `bass drum`, `kick` |
| snare  | `snare drum`, `snare hit`, `snare` |
| hihat  | `hi-hat`, `hihat`, `closed hi-hat`, `open hi-hat` |
| tom    | `tom drum`, `tom hit`, `floor tom`, `rack tom` |
| crash  | `crash cymbal`, `crash` |
| ride   | `ride cymbal`, `ride` |

---

## 필터링 기준

- **duration**: 0.1초 ~ 3.0초 (너무 길면 단일 히트가 아닐 수 있음)
- **license**: Creative Commons (상업적 제한 없는 것)
- **목표 수량**: 클래스당 300~500개
- **포맷**: wav 우선 (없으면 mp3도 OK, librosa가 자동 변환)

---

## 저장 구조

```
dataset/
  kick/
    idmt_001.wav   ← 기존 IDMT 샘플
    ...
    fs_kick_001.wav  ← freesound 샘플 (fs_ 접두사)
    ...
  snare/
  hihat/
  tom/             ← 새로 생성
  crash/           ← 새로 생성
  ride/            ← 새로 생성
```

---

## 코드 스니펫

```python
import freesound
import librosa
import soundfile as sf
import numpy as np
import os

SR       = 22050
DURATION = 0.5
N_SAMPLES = int(SR * DURATION)

client = freesound.FreesoundClient()
client.set_token("YOUR_API_KEY", "token")

SEARCH_TAGS = {
    'kick' : ['kick drum', 'bass drum kick'],
    'snare': ['snare drum', 'snare hit'],
    'hihat': ['hi-hat closed', 'hi-hat open', 'hihat'],
    'tom'  : ['tom drum', 'floor tom', 'rack tom'],
    'crash': ['crash cymbal'],
    'ride' : ['ride cymbal'],
}

def download_and_save(sound, label, save_dir, idx):
    path = os.path.join(save_dir, f"fs_{label}_{idx:04d}.wav")
    if os.path.exists(path):
        return

    sound.retrieve_preview(save_dir, f"tmp_{idx}.mp3")
    tmp = os.path.join(save_dir, f"tmp_{idx}.mp3")

    y, _ = librosa.load(tmp, sr=SR, mono=True)
    y = librosa.util.fix_length(y, size=N_SAMPLES)

    sf.write(path, y, SR)
    os.remove(tmp)

for label, tags in SEARCH_TAGS.items():
    save_dir = os.path.join('dataset', label)
    os.makedirs(save_dir, exist_ok=True)
    idx = 0
    for tag in tags:
        results = client.text_search(
            query=tag,
            filter='duration:[0.1 TO 3.0]',
            fields='id,name,duration,license,previews',
            page_size=150
        )
        for sound in results:
            download_and_save(sound, label, save_dir, idx)
            idx += 1
            if idx >= 400:
                break
```

---

## 주의사항

- `retrieve_preview` 는 MP3 미리보기 (~30초 미만) 다운로드. 전체 파일은 OAuth 필요.
  미리보기도 단일 히트 샘플이면 충분히 쓸 수 있음
- 0.5초로 맞출 때 librosa.util.fix_length 사용 (짧으면 0 패딩, 길면 자름)
- 파일명에 `fs_` 접두사 붙여서 IDMT 샘플과 구분
- 클래스 불균형 주의: 클래스마다 비슷한 수로 맞출 것

---

## 실행 순서

```
1. pip install freesound soundfile
2. API key 발급 (freesound.org)
3. download_freesound.py 실행
4. 데이터 수 확인
5. data_preprocessor.py 수정 (LABELS 6클래스로 확장)
6. train.py 수정 (CLASSES 6클래스, WeightedRandomSampler 추가)
7. inference.py 수정 (CLASSES 6클래스)
8. 재학습
```
