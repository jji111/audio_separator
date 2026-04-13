# 오디오/신호처리 기초 개념

이 프로그램(inference.py)을 이해하는 데 필요한 개념만 정리한다.

---

## 1. 소리란 무엇인가

소리는 공기 압력의 진동이다. 마이크가 이 진동을 감지하면 전압 신호로 바뀌고, ADC(아날로그-디지털 변환기)가 이를 숫자의 배열로 저장한다.

librosa.load()가 반환하는 `y`가 바로 이 숫자 배열이다.

```python
# inference.py:181
y, sr = librosa.load(INPUT_FILE, sr=SR)
# y = [-0.003, 0.012, -0.001, ...] 형태의 float32 배열
# 값의 범위는 대략 -1.0 ~ 1.0
```

---

## 2. 샘플링

아날로그 신호를 초당 몇 번 측정(샘플링)하느냐를 샘플레이트(SR)라고 한다.

```python
# inference.py:12
SR = 22050
```

- **SR = 22050**: 1초에 22050번 측정한다. 1초짜리 오디오는 숫자 22050개가 된다.
- **Ts (샘플링 간격)**: `Ts = 1 / SR = 1 / 22050 ≈ 0.0000454초` — 이웃한 두 샘플 사이의 시간 간격
- **N_SAMPLES**: 0.5초 클립의 샘플 수

```python
# inference.py:16
N_SAMPLES = int(SR * DURATION)  # int(22050 * 0.5) = 11025개
```

클립 하나가 정확히 11025개 숫자로 표현된다.

---

## 3. 진폭 (Amplitude)

파형의 위아래 높이, 즉 공기 압력의 세기다. `y` 배열의 각 숫자값이 그 순간의 진폭이다.

- 값이 클수록 소리가 크다
- 드럼 히트 순간에는 값이 갑자기 커졌다가 빠르게 줄어든다

---

## 4. 주파수 (Frequency)

1초 동안 진동이 반복되는 횟수, 단위는 Hz다. 소리의 높낮이를 결정한다.

| 소리 | 대략적인 주파수 |
|------|----------------|
| 킥 드럼 | 50 ~ 200 Hz |
| 스네어 | 150 ~ 400 Hz |
| 하이햇 | 5000 ~ 16000 Hz |
| 크래쉬 심벌 | 5000 ~ 16000 Hz (지속) |

inference.py는 이 주파수 대역 차이를 이용해 악기를 구분한다:

```python
# inference.py:128~130
e_low  = _band_max(col, freqs,   80,   400)   # 톰 몸통 대역
e_mid  = _band_max(col, freqs,  400,  2000)   # 톰 어택 / 스네어 잔향
e_high = _band_max(col, freqs, 5000, 16000)   # 심벌류 (crash, ride, hihat)
```

---

## 5. dB (데시벨)

진폭을 로그 스케일로 표현한 단위다. 사람 귀가 소리를 로그적으로 인식하기 때문에 쓴다.

**공식**: `dB = 20 × log10(A / A_ref)`

- `A`: 측정할 진폭
- `A_ref`: 기준값 (보통 `np.max` — 전체에서 가장 큰 값)
- 가장 큰 소리 = 0 dB
- 나머지는 모두 음수 dB (조용할수록 더 큰 음수)

```python
# inference.py:189
stft_db = librosa.amplitude_to_db(stft_mag, ref=np.max)
# 가장 큰 진폭이 0dB, 나머지는 -10, -30, -60 등
```

```python
# inference.py:167
db = librosa.amplitude_to_db(rms, ref=1.0)
# ref=1.0이면 진폭 1.0 = 0dB, 작은 소리는 -20, -40 등 음수
```

스펙트럼 분류기에서 `-50dB` 기준으로 소리가 있는지 없는지를 판단한다:

```python
# inference.py:133
if e_high > -50:   # -50dB보다 크면 고음 성분이 존재한다고 판단
```

---

## 6. 나이퀴스트 정리

**Fmax = SR / 2 = 22050 / 2 = 11025 Hz**

샘플링으로 표현할 수 있는 최대 주파수는 SR의 절반이다.

왜냐면 어떤 주파수를 제대로 복원하려면 1주기 안에 최소 2개의 샘플이 필요하기 때문이다. SR=22050이면 1초에 22050개 샘플 → 11025Hz 신호는 1주기에 정확히 2개 샘플 → 이게 한계다.

`librosa.fft_frequencies()`의 결과가 0Hz부터 11025Hz까지인 이유가 이것이다:

```python
# test/librosa_test.py:37
freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
# freqs: 0.0Hz ~ 11025.0Hz (1025개)
```

---

## 7. FFT (고속 푸리에 변환)

소리(시간 x 진폭)를 주파수별 성분으로 분리하는 알고리즘이다.

**결과가 복소수인 이유**: FFT는 각 주파수의 진폭(크기)뿐 아니라 위상(phase, 파동이 어느 타이밍에 있는지)도 계산한다. 복소수 `a + bi`에서 크기(`|z| = sqrt(a²+b²)`)가 진폭, 각도(`atan2(b,a)`)가 위상이다.

**`np.abs()`로 진폭만 뽑는 이유**: 드럼 분류에는 위상 정보가 필요 없고 각 주파수의 에너지(크기)만 중요하다.

```python
# inference.py:188 / test/librosa_test.py:35
stft_mag = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH))
# librosa.stft() → 복소수 배열
# np.abs() → 진폭만 뽑은 실수 배열
```

---

## 8. STFT (단시간 푸리에 변환)

오디오 전체에 FFT를 한 번 돌리면 "이 곡 전체에 어떤 주파수가 있다"는 정보만 나온다. 언제 그 소리가 났는지 모른다.

STFT는 짧은 구간을 잘라서 FFT를 반복한다. 그래서 **시간 x 주파수** 2D 정보를 얻는다.

```python
# inference.py:17~18
HOP_LENGTH = 512   # 프레임을 512샘플씩 이동
N_FFT      = 2048  # 한 번에 2048샘플을 보고 FFT
```

- **n_fft = 2048**: 한 번의 FFT에 쓸 샘플 수. 클수록 주파수 해상도 좋고, 시간 해상도는 낮아진다.
- **hop_length = 512**: 다음 FFT로 넘어갈 때 이동하는 샘플 수. 작을수록 시간 해상도 높아진다.

**결과 shape 계산**:
- 주파수 축: `n_fft / 2 + 1 = 2048 / 2 + 1 = 1025개`
- 시간 축(프레임 수): `len(y) / hop_length` (반올림)

```python
# test/librosa_test.py:39
print(f"stft_mag shape: {stft_mag.shape}")
# 출력 예: (1025, 2582)
# = 주파수 1025개 x 프레임 2582개
```

---

## 9. 멜스펙트로그램

STFT는 1025개 주파수 빈을 균일 간격으로 나눈다. 그런데 사람 귀는 저음 쪽에서 주파수 변화를 더 잘 감지하고, 고음 쪽에서는 둔감하다.

**멜 스케일**: 사람 귀의 특성에 맞게 저음 쪽은 촘촘하게, 고음 쪽은 성기게 재배치한 주파수 스케일.

멜스펙트로그램은 1025개 주파수 빈을 64개 멜 필터로 압축한다:

```python
# inference.py:90 / test/librosa_test.py:53
mel = librosa.feature.melspectrogram(y=clip, sr=SR, n_mels=N_MELS, hop_length=HOP_LENGTH)
# N_MELS = 64
# 결과 shape: (64, 프레임수)
```

CNN 입력으로 쓰기 좋다. 크기가 작고, 사람이 소리를 인식하는 방식과 비슷하기 때문이다.

---

## 10. Onset (온셋)

드럼 히트처럼 **에너지가 갑자기 커지는 순간**을 onset이라 한다.

ODF(Onset Detection Function): 프레임마다 에너지 변화량을 계산한 배열. 드럼이 치는 순간에 값이 뾰족하게 솟는다.

```python
# inference.py:194~196
odf_full     = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LENGTH)
odf_high     = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LENGTH, fmin=3000)
odf_combined = odf_full * 0.6 + odf_high * 0.4
```

킥/스네어(저음)와 하이햇/심벌(고음)을 모두 잘 잡으려고 전체 ODF와 고음(3kHz 이상) ODF를 6:4 비율로 합산한다.

---

## 11. RMS (Root Mean Square)

클립의 평균 에너지를 나타내는 값이다. 드럼을 세게 쳤는지 약하게 쳤는지를 수치화한다.

**공식**: `RMS = sqrt( mean(y²) )`

```python
# inference.py:166
rms = librosa.feature.rms(y=clip)[0].max()
```

RMS를 dB로 변환해서 MIDI velocity(세기, 20~127)로 매핑한다:

```python
# inference.py:166~169
rms = librosa.feature.rms(y=clip)[0].max()
db  = librosa.amplitude_to_db(rms, ref=1.0)
v   = (db - floor) / (ceil - floor) * 107 + 20
return int(np.clip(v, 20, 127))
```

---

## 12. 잔향 (Decay)

드럼을 치고 나서 소리가 사라질 때까지의 시간이다. 악기마다 특성이 다르다:

| 악기 | 잔향 특성 |
|------|-----------|
| 크래쉬 심벌 | 매우 길다 (쫙 퍼지는 소리) |
| 라이드 심벌 | 중간 (딱딱 울림) |
| 오픈 하이햇 | 짧음 |
| 클로즈드 하이햇 | 아주 짧음 |

inference.py는 고음 대역(5kHz~16kHz)의 에너지가 -52dB 아래로 떨어질 때까지 걸리는 프레임 수로 잔향을 측정한다:

```python
# inference.py:134~145
decay = _decay_frames(stft_db, freqs, f_idx, 5000, 16000)

if decay >= 25:
    return 'crash'    # 잔향 아주 길면 크래쉬
elif decay >= 14:
    return 'ride'     # 중간이면 라이드
else:
    return 'hihat'    # 짧으면 하이햇
```

프레임 1개 = `hop_length / sr = 512 / 22050 ≈ 0.023초`이므로, decay=25프레임은 약 0.58초다.
