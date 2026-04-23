# e-GMD 데이터셋 추가 계획

## 목표
현재 kick/snare/hihat 3클래스 → kick/snare/hihat/tom/crash/ride 6클래스로 확장

---

## 현재 상태

| 파일 | 현재 클래스 | 변경 필요 |
|------|------------|---------|
| `data_preprocessor.py` | kick/snare/hihat | LABELS 확장, e-GMD 파싱 로직 추가 |
| `train.py` | CLASSES = 3개 | CLASSES 확장 |
| `inference.py` | CNN 3클래스 + 스펙트럼 폴백 | CNN이 직접 crash/tom/ride 분류 |

---

## e-GMD 데이터셋 구조

```
e-gmd-v1.0.0/
  ├── drummer1/
  │     ├── session1/
  │     │     ├── 001.wav   ← 오디오
  │     │     ├── 001.mid   ← MIDI (라벨)
  │     │     └── ...
  │     └── ...
  └── ...
  └── e-gmd-v1.0.0.csv  ← 메타데이터
```

MIDI 파일에 각 히트의 정확한 타이밍과 pitch(악기 번호)가 기록되어 있음.

### GM 드럼 MIDI 번호 매핑
```
36 = Kick
38 = Snare
42 = Closed HiHat
46 = Open HiHat
41 = Low Tom
45 = Mid Tom
50 = High Tom
49 = Crash
51 = Ride
```

---

## 전처리 방식

e-GMD는 IDMT처럼 파일명에 라벨이 없고, **MIDI 파일에서 타이밍을 읽어서 오디오를 자르는 방식**이 필요합니다.

```
MIDI 파일 → 각 히트의 시간(초) + pitch(악기) 읽기
오디오 파일 → 그 시간에서 0.5초 클립 잘라내기
→ pitch로 라벨 결정
→ dataset/{label}/ 에 저장
```

### 필요한 라이브러리
```
pretty_midi  ← MIDI 파싱 (이미 설치됨)
```

---

## 변경할 파일들

### 1. `data_preprocessor.py`
- IDMT 처리 로직 유지
- e-GMD 처리 로직 추가 (MIDI 파싱 → 오디오 클립 추출)
- LABELS에 tom/crash/ride 추가

```python
LABELS = ['kick', 'snare', 'hihat', 'tom', 'crash', 'ride']

# GM pitch → label 매핑
PITCH_MAP = {
    36: 'kick',
    38: 'snare', 40: 'snare',
    42: 'hihat', 46: 'hihat',
    41: 'tom', 43: 'tom', 45: 'tom', 47: 'tom', 48: 'tom', 50: 'tom',
    49: 'crash', 55: 'crash', 57: 'crash',
    51: 'ride', 53: 'ride', 59: 'ride',
}
```

### 2. `train.py`
- CLASSES 확장
```python
CLASSES = ['kick', 'snare', 'hihat', 'tom', 'crash', 'ride']
```
- 클래스 불균형 처리: WeightedRandomSampler 추가 (데이터 수가 클래스마다 다를 수 있음)

### 3. `inference.py`
- CLASSES 확장
- classify_with_spectrum을 CNN이 커버하므로 단순화 가능
- NOTE_MAP 확인 (이미 tom/crash/ride 있음)

---

## 실행 순서

```
1. e-GMD 다운로드 (약 13GB)
2. data_preprocessor.py 실행 (IDMT + e-GMD 동시 처리)
3. train.py 실행 (6클래스 학습)
4. inference.py 실행 (결과 확인)
```

---

## 주의사항

- e-GMD 오디오가 44100Hz로 저장되어 있을 수 있음 → librosa.load(sr=22050)으로 자동 리샘플링
- 클래스 불균형 가능성: kick/snare가 tom/crash보다 훨씬 많음 → WeightedRandomSampler 필요
- 기존 drum_model.pth는 덮어씌워짐 → 백업 권장
