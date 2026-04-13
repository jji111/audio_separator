import librosa
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

AUDIO_FILE = '../music_sample/separated/htdemucs/test/drums.wav'
SR = 22050
HOP_LENGTH = 512
N_FFT = 2048
N_MELS = 64
N_SAMPLES = int(SR * 0.5)

y, sr = librosa.load(AUDIO_FILE, sr=SR, duration=60)
print(f"y shape: {y.shape}  ({len(y)/sr:.1f}초, 샘플 {len(y)}개)")
print(f"sr: {sr}  (1초 = {sr}개 샘플)")
print()

# ---------------------------------------------------------------
# 1. 파형 (시간 x 진폭)
# ---------------------------------------------------------------
# plt.figure(figsize=(14, 3))
# times = np.arange(len(y)) / sr
# plt.plot(times, y, linewidth=0.5)
# plt.xlabel('시간 (초)')
# plt.ylabel('진폭')
# plt.title('1. 파형 (Waveform) - 시간축 x 진폭')
# plt.tight_layout()
# plt.show()

# ---------------------------------------------------------------
# 2. STFT - 주파수 x 시간 x 진폭
# ---------------------------------------------------------------
stft_mag = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH))
stft_db  = librosa.amplitude_to_db(stft_mag, ref=np.max)
freqs    = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)

print(f"stft_mag shape: {stft_mag.shape}  (주파수 {stft_mag.shape[0]}개 x 프레임 {stft_mag.shape[1]}개)")
print(f"freqs: {freqs[0]:.1f}Hz ~ {freqs[-1]:.1f}Hz  ({len(freqs)}개)")
print()

# plt.figure(figsize=(14, 4))
# librosa.display.specshow(stft_db, sr=sr, hop_length=HOP_LENGTH, x_axis='time', y_axis='hz')
# plt.colorbar(format='%+2.0f dB')
# plt.title('2. STFT 스펙트로그램 - 주파수 x 시간, 값=dB(진폭)')
# plt.tight_layout()
# plt.show()

# ---------------------------------------------------------------
# 3. 멜스펙트로그램
# ---------------------------------------------------------------
mel    = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS, hop_length=HOP_LENGTH)
mel_db = librosa.power_to_db(mel, ref=np.max)

print(f"mel_db shape: {mel_db.shape}  (멜 필터 {N_MELS}개 x 프레임 {mel_db.shape[1]}개)")
print(f"mel_db 범위: {mel_db.min():.1f}dB ~ {mel_db.max():.1f}dB")
print()

# plt.figure(figsize=(14, 4))
# librosa.display.specshow(mel_db, sr=sr, hop_length=HOP_LENGTH, x_axis='time', y_axis='mel')
# plt.colorbar(format='%+2.0f dB')
# plt.title('3. 멜스펙트로그램 - 멜 주파수 x 시간, 값=dB')
# plt.tight_layout()
# plt.show()

# ---------------------------------------------------------------
# 4. onset (드럼 히트 위치)
# ---------------------------------------------------------------
odf_full = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LENGTH)
odf_high = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LENGTH, fmin=3000)
odf_combined = odf_full * 0.6 + odf_high * 0.4

onset_frames = librosa.onset.onset_detect(onset_envelope=odf_combined, sr=sr, backtrack=True, delta=0.3, wait=2, hop_length=HOP_LENGTH)
onset_times  = librosa.frames_to_time(onset_frames, sr=sr, hop_length=HOP_LENGTH)

print(f"odf shape: {odf_full.shape}  (프레임마다 에너지 변화량)")
print(f"onset 개수: {len(onset_frames)}개")
print(f"onset 시간: {onset_times[:10]}...")
print()

# plt.figure(figsize=(14, 4))
# odf_times = librosa.frames_to_time(np.arange(len(odf_combined)), sr=sr, hop_length=HOP_LENGTH)
# plt.plot(odf_times, odf_combined, label='ODF (합산)')
# plt.plot(odf_times, odf_full, alpha=0.5, label='ODF 전체')
# plt.plot(odf_times, odf_high, alpha=0.5, label='ODF 고음(3kHz+)')
# for t in onset_times:
#     plt.axvline(x=t, color='r', alpha=0.5, linewidth=0.8)
# plt.xlabel('시간 (초)')
# plt.ylabel('에너지 변화량')
# plt.title('4. ODF + onset 위치 (빨간 선)')
# plt.legend()
# plt.tight_layout()
# plt.show()

# ---------------------------------------------------------------
# 5. onset 위치를 파형 위에 표시
# ---------------------------------------------------------------
# plt.figure(figsize=(14, 3))
# plt.plot(times, y, linewidth=0.5)
# for t in onset_times:
#     plt.axvline(x=t, color='r', alpha=0.7, linewidth=0.8)
# plt.xlabel('시간 (초)')
# plt.ylabel('진폭')
# plt.title('5. 파형 + onset 위치 (빨간 선 = 드럼 히트)')
# plt.tight_layout()
# plt.show()

# ---------------------------------------------------------------
# 6. 클립 하나 (첫 번째 onset에서 0.5초)
# ---------------------------------------------------------------
onset_samples = librosa.frames_to_samples(onset_frames, hop_length=HOP_LENGTH)
clip = y[onset_samples[0] : onset_samples[0] + N_SAMPLES]
clip_mel = librosa.feature.melspectrogram(y=clip, sr=SR, n_mels=N_MELS, hop_length=HOP_LENGTH)
clip_mel_db = librosa.power_to_db(clip_mel, ref=np.max)

print(f"클립 shape: {clip.shape}  (0.5초 = {N_SAMPLES}개 샘플)")
print(f"클립 멜스펙트로그램 shape: {clip_mel_db.shape}  → unsqueeze → (1, {N_MELS}, {clip_mel_db.shape[1]})")
print()

# plt.figure(figsize=(6, 4))
# librosa.display.specshow(clip_mel_db, sr=sr, hop_length=HOP_LENGTH, x_axis='time', y_axis='mel')
# plt.colorbar(format='%+2.0f dB')
# plt.title(f'6. 첫 번째 onset 클립 멜스펙트로그램 (CNN 입력)')
# plt.tight_layout()
# plt.show()

# ---------------------------------------------------------------
# 7. 클립별 RMS dB 범위 분석 (velocity 계산용)
# ---------------------------------------------------------------
print("─" * 40)
print("클립별 RMS dB 범위 분석")
print("─" * 40)

rms_db_list = []
for i, s_idx in enumerate(onset_samples):  # 전체
    c = y[s_idx : s_idx + N_SAMPLES]
    rms = librosa.feature.rms(y=c)[0].max()
    db  = librosa.amplitude_to_db(rms, ref=1.0)
    rms_db_list.append(db)
    if i < 20:  # 출력은 앞 20개만
        print(f"  onset {i+1:02d}  RMS dB: {db:.1f}")

print()
print(f"최소: {min(rms_db_list):.1f}dB")
print(f"최대: {max(rms_db_list):.1f}dB")
print(f"평균: {np.mean(rms_db_list):.1f}dB")

# 분포 시각화
plt.figure(figsize=(10, 3))
plt.bar(range(len(rms_db_list)), rms_db_list)
plt.axhline(y=min(rms_db_list), color='b', linestyle='--', label=f'최소 {min(rms_db_list):.1f}dB')
plt.axhline(y=max(rms_db_list), color='r', linestyle='--', label=f'최대 {max(rms_db_list):.1f}dB')
plt.xlabel('onset 번호')
plt.ylabel('RMS dB')
plt.title('7. 클립별 RMS dB 분포 (velocity 범위 결정용)')
plt.legend()
plt.tight_layout()
plt.show()
