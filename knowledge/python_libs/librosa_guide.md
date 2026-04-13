# librosa 사용 가이드

이 프로젝트(inference.py, test/librosa_test.py)에서 실제로 쓰이는 librosa 함수들을 정리한다.

---

## 1. librosa.load()

오디오 파일을 numpy 배열로 읽는다.

```python
# inference.py:181
y, sr = librosa.load(INPUT_FILE, sr=SR)

# test/librosa_test.py:15
y, sr = librosa.load(AUDIO_FILE, sr=SR, duration=60)
```

**파라미터**

| 파라미터 | 값 | 설명 |
|----------|----|------|
| `sr` | 22050 | 리샘플링할 샘플레이트. 파일 원본 SR이 44100이어도 22050으로 변환해서 읽는다. `None`이면 원본 SR 유지. |
| `duration` | 60 | 읽을 초 수. `test/librosa_test.py`에서는 60초만 읽어서 테스트 속도를 높인다. `inference.py`에서는 생략 → 전체 파일 읽음. |

**반환값**

- `y`: float32 numpy 배열, 모노 파형, 값 범위 약 -1.0 ~ 1.0
- `sr`: 실제 사용된 샘플레이트 (sr 파라미터를 지정했으면 그 값이 돌아온다)

**리샘플링 처리**: 파일 원본 SR != 지정 SR이면 librosa가 자동으로 리샘플링한다. 내부적으로 `resampy` 또는 `soxr`을 사용한다.

```python
# test/librosa_test.py:16
print(f"y shape: {y.shape}  ({len(y)/sr:.1f}초, 샘플 {len(y)}개)")
# 예: y shape: (1323000,)  (60.0초, 샘플 1323000개)
# 60초 * 22050 = 1,323,000개
```

---

## 2. librosa.stft()

Short-Time Fourier Transform. 오디오를 시간 x 주파수 2D 배열로 변환한다.

```python
# inference.py:188 / test/librosa_test.py:35
stft_mag = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH))
```

**파라미터**

| 파라미터 | 값 | 설명 |
|----------|----|------|
| `n_fft` | 2048 | 한 번의 FFT에 사용하는 샘플 수. 주파수 해상도 결정. |
| `hop_length` | 512 | 프레임 이동 간격(샘플 수). 시간 해상도 결정. |

**반환값**: 복소수 numpy 배열, shape = `(n_fft/2 + 1, 프레임수)` = `(1025, 프레임수)`

**왜 `np.abs()`를 씌우나**: `librosa.stft()`는 복소수 배열을 반환한다. 각 원소 `a+bi`에서 `|a+bi| = sqrt(a²+b²)`가 해당 주파수의 진폭(크기)이다. 이 프로젝트는 위상 정보가 필요 없으므로 `np.abs()`로 진폭만 뽑는다.

**shape 계산**

```
주파수 빈 수 = n_fft / 2 + 1 = 2048 / 2 + 1 = 1025
프레임 수    = ceil(len(y) / hop_length)
```

```python
# test/librosa_test.py:39
print(f"stft_mag shape: {stft_mag.shape}")
# 예: (1025, 2583)
```

---

## 3. librosa.amplitude_to_db() vs librosa.power_to_db()

둘 다 dB로 변환하지만 입력 단위가 다르다.

**amplitude_to_db()**: 입력이 진폭(amplitude). 공식: `20 * log10(S / ref)`

```python
# inference.py:189
stft_db = librosa.amplitude_to_db(stft_mag, ref=np.max)
# stft_mag = np.abs(stft) → 진폭 → amplitude_to_db 사용

# inference.py:167
db = librosa.amplitude_to_db(rms, ref=1.0)
# rms도 진폭 단위 → amplitude_to_db 사용
# ref=1.0: 진폭 1.0이 기준(0dB). 드럼 소리는 보통 1.0보다 작으므로 결과가 음수 dB
```

**power_to_db()**: 입력이 파워(power = amplitude²). 공식: `10 * log10(S / ref)`

```python
# inference.py:91 / test/librosa_test.py:54
mel_db = librosa.power_to_db(mel, ref=np.max)
# librosa.feature.melspectrogram()은 파워 스펙트럼을 반환 → power_to_db 사용
```

**핵심 구분**: `librosa.stft()` + `np.abs()` → amplitude → `amplitude_to_db`. `librosa.feature.melspectrogram()` → power → `power_to_db`. 잘못 쓰면 dB 값이 절반이 되거나 두 배가 된다.

---

## 4. librosa.fft_frequencies()

STFT의 각 주파수 빈이 실제로 몇 Hz인지 배열로 반환한다.

```python
# inference.py:190 / test/librosa_test.py:37
freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
```

**반환값**: shape `(n_fft/2 + 1,)` = `(1025,)`. 0Hz부터 sr/2(11025Hz)까지 균일 간격으로 나뉜다.

```python
# test/librosa_test.py:40
print(f"freqs: {freqs[0]:.1f}Hz ~ {freqs[-1]:.1f}Hz  ({len(freqs)}개)")
# freqs: 0.0Hz ~ 11025.0Hz  (1025개)
```

**계산 원리**: 빈 간격 = `sr / n_fft = 22050 / 2048 ≈ 10.77Hz`. `freqs[i] = i * sr / n_fft`.

이 배열이 있어야 "STFT의 몇 번째 행이 몇 Hz인가"를 알 수 있다. 스펙트럼 분류기에서 특정 Hz 대역을 잘라낼 때 쓴다:

```python
# inference.py:107
def _hz_bin(freqs, hz):
    return int(np.argmin(np.abs(freqs - hz)))
# freqs 배열에서 원하는 Hz에 가장 가까운 인덱스를 찾는다
```

---

## 5. librosa.feature.melspectrogram()

멜 스케일 스펙트로그램을 계산한다. 내부적으로 STFT를 수행하고 멜 필터뱅크를 곱한다.

```python
# inference.py:90
mel = librosa.feature.melspectrogram(y=clip, sr=SR, n_mels=N_MELS, hop_length=HOP_LENGTH)

# test/librosa_test.py:53
mel = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS, hop_length=HOP_LENGTH)
```

**파라미터**

| 파라미터 | 값 | 설명 |
|----------|----|------|
| `y` | 오디오 배열 | 파형 입력 |
| `sr` | 22050 | 샘플레이트 |
| `n_mels` | 64 | 멜 필터 수 = 출력의 주파수 축 크기 |
| `hop_length` | 512 | STFT의 hop_length와 동일한 역할 |

**반환값**: 파워 스펙트럼(진폭²), shape = `(n_mels, 프레임수)` = `(64, 프레임수)`. **파워 스펙트럼이므로 `power_to_db()`로 변환해야 한다.**

CNN 입력으로 변환하는 과정:

```python
# inference.py:90~92
mel    = librosa.feature.melspectrogram(y=clip, sr=SR, n_mels=N_MELS, hop_length=HOP_LENGTH)
mel_db = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
tensor = torch.tensor(mel_db).unsqueeze(0).unsqueeze(0).to(device)
# mel_db shape: (64, 프레임수)
# unsqueeze(0): (1, 64, 프레임수)  ← 배치 차원
# unsqueeze(0): (1, 1, 64, 프레임수) ← 채널 차원 (그레이스케일 이미지처럼)
```

```python
# test/librosa_test.py:56~57
print(f"mel_db shape: {mel_db.shape}")
# 예: (64, 2583)
print(f"mel_db 범위: {mel_db.min():.1f}dB ~ {mel_db.max():.1f}dB")
# ref=np.max이므로 최댓값 = 0.0dB, 나머지는 음수
```

---

## 6. librosa.feature.rms()

클립의 프레임별 RMS(Root Mean Square) 에너지를 계산한다.

```python
# inference.py:166 / test/librosa_test.py:138~139
rms = librosa.feature.rms(y=clip)[0].max()
```

**반환값**: shape `(1, n_frames)`. 첫 번째 차원은 항상 1이다 (모노 오디오라서). 각 프레임의 RMS 값이 들어있다.

**`[0].max()`를 쓰는 이유**:
- `[0]`: shape `(1, n_frames)` → `(n_frames,)`. 불필요한 앞 차원 제거.
- `.max()`: 클립 내 프레임 중 가장 큰 RMS를 쓴다. 평균이 아닌 최댓값을 쓰는 이유는 드럼 히트는 onset 직후 한두 프레임에 에너지가 집중되기 때문이다. 평균을 쓰면 긴 클립의 조용한 부분에 희석된다.

velocity 계산 흐름:

```python
# inference.py:165~169
def amplitude_to_velocity(clip, floor, ceil):
    rms = librosa.feature.rms(y=clip)[0].max()       # 클립의 최대 RMS
    db  = librosa.amplitude_to_db(rms, ref=1.0)       # dB로 변환
    v   = (db - floor) / (ceil - floor) * 107 + 20    # floor~ceil 범위를 20~127로 선형 매핑
    return int(np.clip(v, 20, 127))                    # 범위 초과 방지
```

---

## 7. librosa.util.fix_length()

오디오 배열의 길이를 지정한 크기에 맞춘다.

```python
# inference.py:89
clip = librosa.util.fix_length(clip, size=N_SAMPLES)
# N_SAMPLES = 11025 (0.5초)
```

**동작**:
- `len(clip) < N_SAMPLES`: 뒤에 0을 패딩해서 늘린다 (zero-padding)
- `len(clip) > N_SAMPLES`: 뒤를 잘라서 줄인다 (truncation)
- `len(clip) == N_SAMPLES`: 그대로 반환

왜 필요한가: onset이 파일 끝부분에 있으면 `y[s_idx : s_idx + N_SAMPLES]` 슬라이싱이 N_SAMPLES보다 짧은 배열을 반환할 수 있다. 이를 CNN 입력 크기에 맞추기 위해 사용한다.

---

## 8. librosa.onset.onset_strength()

프레임마다 에너지 변화량(ODF, Onset Detection Function)을 계산한다.

```python
# inference.py:194~196 / test/librosa_test.py:70~72
odf_full     = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LENGTH)
odf_high     = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LENGTH, fmin=3000)
odf_combined = odf_full * 0.6 + odf_high * 0.4
```

**파라미터**

| 파라미터 | 값 | 설명 |
|----------|----|------|
| `hop_length` | 512 | 프레임 이동 간격 |
| `fmin` | 3000 (odf_high) | 이 Hz 이상의 주파수만 고려한다. 생략하면 전체 주파수 사용. |

**`fmin=3000`을 따로 계산하는 이유**: 킥과 스네어는 저음이라 `odf_full`에서 잘 잡히지만, 하이햇과 심벌은 고음이라 저음 에너지에 가려질 수 있다. 3kHz 이상만 본 `odf_high`를 추가로 만들어 두 ODF를 합산하면 저음/고음 드럼을 모두 놓치지 않는다.

**반환값**: shape `(n_frames,)`. 각 프레임의 에너지 증가량. 드럼 히트 순간에 값이 뾰족하게 솟는다.

---

## 9. librosa.onset.onset_detect()

ODF에서 실제 onset 위치(프레임 번호)를 찾는다.

```python
# inference.py:198~205 / test/librosa_test.py:74
onset_frames = librosa.onset.onset_detect(
    onset_envelope=odf_combined,
    sr=sr,
    backtrack=True,
    delta=0.3,
    wait=2,
    hop_length=HOP_LENGTH
)
```

**파라미터 각각의 역할**

| 파라미터 | 값 | 역할 |
|----------|----|------|
| `onset_envelope` | odf_combined | 미리 계산한 ODF를 넘긴다. 생략하면 내부에서 자동 계산. |
| `backtrack` | True | 피크 위치가 아니라 에너지가 **올라가기 시작한 지점**을 onset으로 잡는다. 드럼 히트의 실제 시작점을 정확히 찾기 위해 사용. |
| `delta` | 0.3 | 피크로 인정할 최소 높이 (ODF의 표준편차 대비). 노이즈성 작은 변화는 무시하고 확실한 히트만 잡는다. |
| `wait` | 2 | 이전 onset 이후 최소 대기 프레임 수. 같은 히트를 연속으로 두 번 잡는 것을 방지. 2프레임 = 약 47ms. |
| `hop_length` | 512 | ODF와 동일한 hop_length를 명시해야 시간 변환이 정확하다. |

**반환값**: shape `(n_onsets,)`. 각 onset의 프레임 번호.

---

## 10. librosa.beat.beat_track()

오디오에서 BPM(템포)과 박자 위치를 찾는다.

```python
# inference.py:183~184
tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
actual_bpm = float(tempo[0]) if isinstance(tempo, (np.ndarray, list)) else float(tempo)
```

**반환값 타입이 ndarray일 수 있는 이유**: librosa 버전에 따라 `beat_track()`의 반환 타입이 다르다. 구버전은 `float`를 반환하고, 신버전(0.10+)은 `np.ndarray`를 반환한다. 이를 안전하게 처리하기 위해 타입 체크 후 `float()` 변환:

```python
actual_bpm = float(tempo[0]) if isinstance(tempo, (np.ndarray, list)) else float(tempo)
```

`_`(두 번째 반환값)은 박자 프레임 배열이지만 이 프로젝트에서는 사용하지 않는다. BPM 정보만 MIDI 파일 생성에 사용한다.

---

## 11. librosa.frames_to_samples() / frames_to_time()

프레임 번호를 샘플 번호 또는 초 단위 시간으로 변환한다.

```python
# inference.py:207~208 / test/librosa_test.py:75, 112
onset_samples = librosa.frames_to_samples(onset_frames, hop_length=HOP_LENGTH)
onset_times   = librosa.frames_to_time(onset_frames, sr=sr, hop_length=HOP_LENGTH)
```

**변환 공식**

```
샘플 번호 = 프레임 번호 × hop_length
         = frame × 512

시간(초)  = 프레임 번호 × hop_length / sr
          = frame × 512 / 22050
```

**용도 구분**:
- `frames_to_samples()`: `y[s_idx : s_idx + N_SAMPLES]`처럼 파형 배열을 슬라이싱할 때 사용
- `frames_to_time()`: MIDI 노트의 시작 시간(`note.start = t`)을 설정할 때 사용

```python
# inference.py:225, 250
for s_idx, t, f_idx in zip(onset_samples, onset_times, onset_frames):
    clip = y[s_idx: s_idx + N_SAMPLES]   # s_idx: 샘플 번호 (배열 인덱싱)
    ...
    note.start = item['time']             # t: 초 단위 시간 (MIDI 타임스탬프)
```

---

## 12. librosa.display.specshow()

스펙트로그램을 시각화한다. test/librosa_test.py에서 주석 처리된 코드에서 사용된다.

```python
# test/librosa_test.py:44 (주석)
librosa.display.specshow(stft_db, sr=sr, hop_length=HOP_LENGTH, x_axis='time', y_axis='hz')

# test/librosa_test.py:62 (주석)
librosa.display.specshow(mel_db, sr=sr, hop_length=HOP_LENGTH, x_axis='time', y_axis='mel')
```

**파라미터**

| 파라미터 | 값 | 설명 |
|----------|----|------|
| 첫 번째 인자 | stft_db 또는 mel_db | 2D dB 배열 |
| `sr` | 22050 | x축(시간) 눈금 계산에 사용 |
| `hop_length` | 512 | 프레임 번호 → 시간 변환에 사용 |
| `x_axis` | 'time' | x축을 초 단위 시간으로 표시 |
| `y_axis` | 'hz' 또는 'mel' | y축 스케일. 'hz'는 선형, 'mel'은 멜 스케일. |

**주의**: `librosa.display.specshow()`는 matplotlib의 `plt.imshow()`와 달리 y축이 아래서 위로(낮은 주파수 → 높은 주파수) 올라간다. `colorbar(format='%+2.0f dB')`를 붙이면 dB 범례도 표시된다.

테스트 코드에서는 각 섹션 별로 시각화 코드를 주석 처리해뒀다. 필요한 섹션만 주석을 풀면 해당 시각화만 볼 수 있다.
