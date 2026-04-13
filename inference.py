import os
import datetime
import torch
import torch.nn as nn
import librosa
import numpy as np
import pretty_midi

# ---------------------------------------------------------------
# train.py와 완전히 동일하게 맞춰야 함
# ---------------------------------------------------------------
SR         = 22050
DURATION   = 0.5
HOP_LENGTH = 512
N_MELS     = 64
N_SAMPLES  = int(SR * DURATION)
N_FFT      = 2048

CLASSES = ['kick', 'snare', 'hihat']

# GM 드럼 MIDI 번호
NOTE_MAP = {
    'kick'   : 36,
    'snare'  : 38,
    'hihat'  : 42,
    'tom_lo' : 41,
    'tom_mid': 45,
    'tom_hi' : 50,
    'crash'  : 49,
    'ride'   : 51,
    'other'  : 37,  # Side Stick - 분류 못한 소리는 일단 여기로
}

# 모델이 이 확신도 이상이면 AI 판정을 믿고, 미만이면 모르는 소리로 보고 스펙트럼 분석으로 넘김 ( 탐 찾으려고 )
CONFIDENCE_THRESHOLD = 0.65

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(BASE_DIR, 'music_sample', 'separated', 'htdemucs', 'test', 'drums.wav')
MODEL_PATH = os.path.join(BASE_DIR, 'drum_model.pth')
OUTPUT_DIR = os.path.join(BASE_DIR, 'music_sample', 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)


class DrumCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
        )

        n_frames  = 1 + N_SAMPLES // HOP_LENGTH
        dummy     = torch.zeros(1, 1, N_MELS, n_frames)
        flat_size = self.conv_layers(dummy).flatten(1).shape[1]

        self.fc_layers = nn.Sequential(
            nn.Linear(flat_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, len(CLASSES))
        )

    def forward(self, x):
        x = self.conv_layers(x).flatten(1)
        return self.fc_layers(x)


def load_model(device):
    if not os.path.exists(MODEL_PATH):
        print(f"모델 파일이 없어요: {MODEL_PATH}")
        print("train.py를 먼저 실행해주세요.")
        exit()
    model = DrumCNN().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
    return model


def classify_with_model(model, device, clip):
    clip   = librosa.util.fix_length(clip, size=N_SAMPLES)
    mel    = librosa.feature.melspectrogram(y=clip, sr=SR, n_mels=N_MELS, hop_length=HOP_LENGTH)
    mel_db = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
    tensor = torch.tensor(mel_db).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]
        pred  = probs.argmax().item()

    return CLASSES[pred], float(probs[pred])


# -----------------------------------------------------------
# 스펙트럼 기반 보조 분류기
# AI 모델이 확신하지 못한 소리(crash, tom, ride 등)를 구분하는 규칙
# -----------------------------------------------------------

def _hz_bin(freqs, hz):
    return int(np.argmin(np.abs(freqs - hz)))

def _band_max(col, freqs, lo, hi):
    return float(np.max(col[_hz_bin(freqs, lo): _hz_bin(freqs, hi) + 1]))

def _decay_frames(stft_db, freqs, f_start, lo, hi, threshold=-52, max_frames=50):
    # onset 이후 해당 대역 에너지가 threshold 아래로 떨어질 때까지 걸리는 프레임 수
    # 길수록 소리가 오래 울림 (crash > ride > open hihat > closed hihat)
    lo_idx, hi_idx = _hz_bin(freqs, lo), _hz_bin(freqs, hi)
    for k in range(1, max_frames):
        f = f_start + k
        if f >= stft_db.shape[1]:
            return k
        if np.max(stft_db[lo_idx: hi_idx + 1, f]) < threshold:
            return k
    return max_frames

def classify_with_spectrum(clip, stft_db, freqs, f_idx):

    col = stft_db[:, f_idx]

    e_low  = _band_max(col, freqs,   80,   400)   # 톰 몸통 대역
    e_mid  = _band_max(col, freqs,  400,  2000)   # 톰 어택 / 스네어 잔향
    e_high = _band_max(col, freqs, 5000, 16000)   # 심벌류 (crash, ride, hihat)

    # 고음 성분이 강하면 심벌류
    if e_high > -50:
        decay = _decay_frames(stft_db, freqs, f_idx, 5000, 16000)

        if decay >= 25:
            # 잔향이 아주 길면 crash - 크래쉬는 쫙 퍼지는 소리
            return 'crash'
        elif decay >= 14:
            # 중간 잔향이면 ride - 라이드는 짧게 딱딱 울림
            return 'ride'
        else:
            # 그것보다 짧으면 사실 hihat에 가까운데 모델이 놓친 것
            # other로 처리하지 않고 hihat으로 보냄
            return 'hihat'

    # 저중음 성분이 강하고 고음은 없으면 톰
    if e_low > -48 or e_mid > -48:
        # 피크 주파수 위치로 hi/mid/lo 톰을 구분
        tom_hi  = _band_max(col, freqs, 400,  900)
        tom_mid = _band_max(col, freqs, 200,  400)
        tom_lo  = _band_max(col, freqs,  80,  200)

        if tom_lo >= tom_mid and tom_lo >= tom_hi:
            return 'tom_lo'
        elif tom_mid >= tom_hi:
            return 'tom_mid'
        else:
            return 'tom_hi'

    # 어느 쪽도 아니면 other
    return 'other'


def amplitude_to_velocity(clip, floor, ceil):
    rms = librosa.feature.rms(y=clip)[0].max()
    db  = librosa.amplitude_to_db(rms, ref=1.0)
    v   = (db - floor) / (ceil - floor) * 107 + 20
    return int(np.clip(v, 20, 127))


def run():
    if not os.path.exists(INPUT_FILE):
        print(f"파일이 없어요: {INPUT_FILE}")
        exit()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = load_model(device)

    print(f"파일 로드 중: {INPUT_FILE}")
    y, sr = librosa.load(INPUT_FILE, sr=SR)

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    actual_bpm = float(tempo[0]) if isinstance(tempo, (np.ndarray, list)) else float(tempo)
    print(f"BPM: {actual_bpm:.1f}")

    # 전체 파일의 STFT - 스펙트럼 분류기에서 잔향 분석할 때 씀
    stft_mag = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH))
    stft_db  = librosa.amplitude_to_db(stft_mag, ref=np.max)
    freqs    = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)

    # 킥/스네어처럼 저음 중심인 것과 하이햇/심벌처럼 고음 중심인 것을
    # 둘 다 잡으려고 전체 ODF와 고음 ODF를 합산해서 사용 
    odf_full     = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LENGTH)
    odf_high     = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LENGTH, fmin=3000)
    odf_combined = odf_full * 0.6 + odf_high * 0.4

    onset_frames = librosa.onset.onset_detect(
        onset_envelope=odf_combined,
        sr=sr,
        backtrack=True, #올라가기 시작한 곳으로 알아서
        delta=0.3,
        wait=2,
        hop_length=HOP_LENGTH
    )

    onset_samples = librosa.frames_to_samples(onset_frames, hop_length=HOP_LENGTH)
    onset_times   = librosa.frames_to_time(onset_frames, sr=sr, hop_length=HOP_LENGTH)

    # 전체 onset 클립의 RMS dB 범위 계산 (곡마다 동적으로 velocity 범위 결정)
    all_rms_db = []
    for s_idx in onset_samples:
        clip = y[s_idx : s_idx + N_SAMPLES]
        rms  = librosa.feature.rms(y=clip)[0].max()
        all_rms_db.append(librosa.amplitude_to_db(rms, ref=1.0))
    vel_floor = min(all_rms_db)
    vel_ceil  = max(all_rms_db)
    print(f"velocity 범위: {vel_floor:.1f}dB ~ {vel_ceil:.1f}dB")

    print(f"온셋 {len(onset_frames)}개 탐지. AI + 스펙트럼 하이브리드 분류 시작...")
    print(f"(모델 확신도 {CONFIDENCE_THRESHOLD} 이상 → AI 판정, 미만 → 스펙트럼 분석)")
    print()
    
    results = []
    for s_idx, t, f_idx in zip(onset_samples, onset_times, onset_frames):
        clip              = y[s_idx: s_idx + N_SAMPLES]
        label, confidence = classify_with_model(model, device, clip)

        if confidence >= CONFIDENCE_THRESHOLD:
            # 모델이 충분히 확신하면 그대로 사용
            source = 'AI'
        else:
            # 모델이 애매하면 스펙트럼 분석으로 재분류
            label  = classify_with_spectrum(clip, stft_db, freqs, f_idx)
            source = 'spec'

        velocity = amplitude_to_velocity(clip, vel_floor, vel_ceil)  # AI/스펙트럼 관계없이 실제 진폭으로 계산

        results.append({'time': t, 'label': label, 'velocity': velocity, 'source': source})
        print(f"  [{t:.2f}s]  {label:<10}  {source:<5}  확신도: {confidence:.2f}  vel: {velocity}")

    # MIDI 생성
    drum_proj  = pretty_midi.PrettyMIDI(initial_tempo=actual_bpm)
    drum_track = pretty_midi.Instrument(program=0, is_drum=True)

    for item in results:
        note_num = NOTE_MAP.get(item['label'])
        if note_num is None:
            continue
        note = pretty_midi.Note(
            velocity = item['velocity'],
            pitch    = note_num,
            start    = item['time'],
            end      = item['time'] + 0.08
        )
        drum_track.notes.append(note)

    drum_proj.instruments.append(drum_track)

    timestamp   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(OUTPUT_DIR, f'drum_score_{timestamp}.mid')
    drum_proj.write(output_path)

    # 결과 요약
    label_counts = {}
    for r in results:
        label_counts[r['label']] = label_counts.get(r['label'], 0) + 1

    print()
    print("─" * 40)
    for label in sorted(label_counts):
        pitch = NOTE_MAP.get(label, '?')
        print(f"  {label:<10}  pitch {pitch:<3}  {label_counts[label]}회")
    print("─" * 40)
    print(f"\n저장 완료: {output_path}")


if __name__ == '__main__':
    run()
