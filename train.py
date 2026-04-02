import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import librosa
import numpy as np

# ---------------------------------------------------------------
# 공통 설정값 - inference.py에도 똑같이 맞춰져 있어야 함
# 여기서 뭔가 바꾸면 inference.py도 같이 바꿔야 함
# ---------------------------------------------------------------
SR         = 22050   # 샘플레이트
DURATION   = 0.5     # 클립 길이 (초). 길수록 시간 해상도 올라감
HOP_LENGTH = 512     # STFT hop size. 작을수록 프레임 촘촘해짐
N_MELS     = 64      # 멜 필터 개수
N_SAMPLES  = int(SR * DURATION)  # 샘플 수 (11025)

CLASSES    = ['kick', 'snare', 'hihat']
NUM_EPOCHS = 30
BATCH_SIZE = 32
LR         = 0.001

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
MODEL_PATH  = os.path.join(BASE_DIR, 'drum_model.pth')


class DrumDataset(Dataset):
    def __init__(self, base_dir):
        self.samples = []  # (파일경로, 라벨 인덱스) 튜플 리스트

        for idx, cls in enumerate(CLASSES):
            folder = os.path.join(base_dir, cls)
            if not os.path.exists(folder):
                print(f"경고: {folder} 폴더가 없어요. data_preprocessor.py를 먼저 실행하세요.")
                continue
            for f in os.listdir(folder):
                if f.endswith('.wav'):
                    self.samples.append((os.path.join(folder, f), idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        y, _ = librosa.load(path, sr=SR)
        # 클립 길이를 N_SAMPLES로 맞춤 (짧으면 0으로 채우고, 길면 자름)
        y = librosa.util.fix_length(y, size=N_SAMPLES)

        # 소리를 멜스펙트로그램 이미지로 변환
        mel    = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS, hop_length=HOP_LENGTH)
        mel_db = librosa.power_to_db(mel, ref=np.max).astype(np.float32)

        # CNN 입력 형태: (채널, 주파수, 시간) → unsqueeze로 채널 차원 추가
        return torch.tensor(mel_db).unsqueeze(0), torch.tensor(label, dtype=torch.long)


class DrumCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_layers = nn.Sequential(
            # 첫 번째 Conv 블록
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),   # 학습 안정화
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),   # 과적합 방지

            # 두 번째 Conv 블록 - 채널을 16 → 32로 늘려서 더 복잡한 특징 추출
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
        )

        # Linear 입력 크기를 하드코딩 안 하고 실제 데이터로 계산
        # n_mels, hop_length 바꿔도 자동으로 맞춰짐
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


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"학습 장치: {device}")

    dataset = DrumDataset(DATASET_DIR)
    if len(dataset) == 0:
        print("데이터가 없어요. data_preprocessor.py를 먼저 실행하세요.")
        return

    # 전체 데이터의 80%는 학습, 20%는 검증에 씀
    val_size   = int(len(dataset) * 0.2)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

    model     = DrumCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # 클래스별 데이터 수 출력해서 불균형 있는지 확인
    label_counts = [0] * len(CLASSES)
    for _, label in dataset.samples:
        label_counts[label] += 1
    print(f"데이터: 전체 {len(dataset)}개  (학습 {train_size} / 검증 {val_size})")
    for cls, cnt in zip(CLASSES, label_counts):
        print(f"  {cls}: {cnt}개")
    print()

    best_val_acc = 0.0

    for epoch in range(NUM_EPOCHS):
        # --- 학습 ---
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # --- 검증 ---
        model.eval()
        correct = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                preds = model(inputs).argmax(dim=1)
                correct += (preds == labels).sum().item()

        val_acc  = correct / val_size * 100
        avg_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch+1:02d}/{NUM_EPOCHS}  loss: {avg_loss:.4f}  val_acc: {val_acc:.1f}%")

        # 검증 정확도가 역대 최고일 때만 저장
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  → 모델 저장 (best val_acc: {best_val_acc:.1f}%)")

    print(f"\n학습 완료. 최고 검증 정확도: {best_val_acc:.1f}%")
    print(f"저장 위치: {MODEL_PATH}")
    print(f"다음 단계: inference.py 실행")


if __name__ == '__main__':
    train()
