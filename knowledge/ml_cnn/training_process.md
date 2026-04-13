# 학습 과정 전체 설명

이 문서는 PyTorch 딥러닝 학습의 모든 구성요소를 개념부터 설명하고,
`train.py`의 `train()` 함수가 실제로 어떻게 동작하는지를 단계별로 추적한다.

---

## 1. Dataset과 DataLoader

### Dataset — 데이터 공급자

`Dataset`은 PyTorch의 추상 클래스로, 두 메서드를 반드시 구현해야 한다:

```python
def __len__(self):
    # 데이터셋 전체 샘플 수를 반환
    return len(self.samples)

def __getitem__(self, idx):
    # idx번째 샘플 하나를 반환
    path, label = self.samples[idx]
    ...
    return tensor, label_tensor
```

`DataLoader`가 내부적으로 `len(dataset)`을 호출해 총 개수를 파악하고,
`dataset[i]`를 호출해 개별 샘플을 가져온다.

이 프로젝트의 `DrumDataset.__getitem__`:

```python
# train.py line 45-57
def __getitem__(self, idx):
    path, label = self.samples[idx]
    y, _ = librosa.load(path, sr=SR)
    y = librosa.util.fix_length(y, size=N_SAMPLES)
    mel    = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS, hop_length=HOP_LENGTH)
    mel_db = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
    return torch.tensor(mel_db).unsqueeze(0), torch.tensor(label, dtype=torch.long)
```

매 호출마다 wav 파일을 읽고 멜스펙트로그램으로 변환한다.
이 변환이 학습 루프마다 반복 실행된다.

### DataLoader — 배치 조립기

```python
# train.py line 112-113
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)
```

DataLoader는 Dataset에서 샘플을 꺼내 배치 단위로 묶어준다.
각 이터레이션마다 `(inputs, labels)` 형태의 배치 텐서를 반환한다:

- `inputs` shape: `(32, 1, 64, 22)` — 32개 샘플을 배치 차원으로 쌓음
- `labels` shape: `(32,)`           — 각 샘플의 정수 라벨

---

## 2. random_split — Train/Val 분리

```python
# train.py line 108-110
val_size   = int(len(dataset) * 0.2)
train_size = len(dataset) - val_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])
```

전체 데이터를 무작위로 섞어 80:20 비율로 나눈다.

**왜 분리하는가?**
모델이 학습 데이터에만 특화(과적합)되었는지를 확인하려면,
학습에 사용하지 않은 별도 데이터로 평가해야 한다.
모델이 학습 데이터의 정답을 통째로 외웠다면 훈련 정확도는 높지만 검증 정확도는 낮다.

---

## 3. batch_size, epoch, step 관계

### 정의

- **epoch**: 전체 학습 데이터를 한 번 다 보는 것
- **step (iteration)**: 미니배치 하나를 처리하는 것
- **batch_size**: 한 step에 처리하는 샘플 수

### 관계 공식

```
한 epoch의 step 수 = ceil(전체 샘플 수 / batch_size)
총 step 수 = epoch 수 × (전체 샘플 수 / batch_size)
```

예시: 샘플 800개, batch_size=32, epoch=30
```
한 epoch step 수 = 800 / 32 = 25 step
총 step 수       = 30 * 25 = 750 step
```

이 프로젝트 설정 (`BATCH_SIZE=32`, `NUM_EPOCHS=30`):

```python
# train.py line 21-22
NUM_EPOCHS = 30
BATCH_SIZE = 32
```

### shuffle=True의 역할

```python
# train.py line 112
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
```

매 epoch마다 데이터 순서를 무작위로 섞는다.

**왜 중요한가?**
shuffle 없이 항상 같은 순서로 학습하면:
- 모델이 데이터 순서의 패턴을 학습할 수 있다
- 한 배치 안에 같은 클래스만 몰릴 수 있다

shuffle=True로 하면 배치마다 다양한 클래스가 섞여 gradient 방향이 더 일반화된다.
검증 DataLoader에는 shuffle이 없다. 결과 재현성을 위해서다.

---

## 4. CrossEntropyLoss — 분류 손실 함수

### 수식

CrossEntropyLoss는 내부적으로 LogSoftmax + NLLLoss의 조합이다:

```
CrossEntropy(y, t) = -log( e^(y_t) / sum_j e^(y_j) )
                   = -y_t + log( sum_j e^(y_j) )
```

- `y` : 모델 출력 로짓 벡터 (softmax 이전 값)
- `t` : 정답 클래스 인덱스

정답 클래스의 확률이 1에 가까울수록 loss가 0에 수렴하고,
0에 가까울수록 loss가 무한대로 발산한다.

예시: 3클래스, 로짓=[2.0, 1.0, 0.1], 정답=0(kick)
```
softmax = [0.659, 0.242, 0.099]
loss = -log(0.659) = 0.418
```

### 왜 분류에 CrossEntropyLoss를 쓰는가?

- MSE(평균제곱오차)를 쓰면 클래스 간 거리 개념이 생긴다 (0=kick이 1=snare보다 2=hihat에서 더 멀다는 의미가 생겨 부적절)
- CrossEntropy는 정답 클래스의 확률을 최대화하는 방향으로 자연스럽게 학습된다
- Softmax와 Log의 조합으로 gradient가 안정적이다

```python
# train.py line 116
criterion = nn.CrossEntropyLoss()
```

---

## 5. Adam Optimizer

### SGD와 비교

| 옵티마이저 | 특징                                     | 단점                          |
|-----------|----------------------------------------|-------------------------------|
| SGD       | 단순 gradient descent, lr 고정           | lr 튜닝 어려움, 수렴 느릴 수 있음 |
| Adam      | 적응형 lr, momentum + RMSProp 결합       | 메모리 더 사용                   |

### Adam의 동작 원리

Adam은 파라미터마다 개별적인 학습률을 적용한다:

```
m_t = β1 * m_{t-1} + (1 - β1) * g_t         # 1차 모멘트 (gradient 평균)
v_t = β2 * v_{t-1} + (1 - β2) * g_t²         # 2차 모멘트 (gradient 분산)

m̂_t = m_t / (1 - β1^t)                        # 편향 보정
v̂_t = v_t / (1 - β2^t)

θ_{t+1} = θ_t - lr * m̂_t / (sqrt(v̂_t) + ε)
```

- `β1 = 0.9` (기본값): gradient의 지수이동평균 (관성)
- `β2 = 0.999` (기본값): gradient 제곱의 지수이동평균 (크기 적응)
- `ε = 1e-8` (기본값): 0 나눗셈 방지

### lr(학습률)의 의미

학습률은 한 step에서 파라미터를 얼마나 이동시킬지를 결정한다.

```python
# train.py line 22, 117
LR         = 0.001
optimizer = optim.Adam(model.parameters(), lr=LR)
```

| lr 값       | 결과                                      |
|------------|------------------------------------------|
| 너무 크다   | gradient 방향으로 너무 많이 이동 → loss 발산 |
| 너무 작다   | 수렴은 하지만 매우 느림, 지역 최솟값에 갇힐 수 있음 |
| 적당 (0.001) | 안정적 수렴                               |

---

## 6. 학습 단계별 코드 분석

### optimizer.zero_grad() — Gradient 초기화

```python
# train.py line 136
optimizer.zero_grad()
```

PyTorch는 기본적으로 `.backward()` 호출 시 gradient를 **누적(accumulate)** 한다.
이전 step의 gradient가 남아있으면 현재 step의 gradient와 합산되어
잘못된 방향으로 업데이트된다.
따라서 매 step 시작 전에 반드시 gradient를 0으로 초기화한다.

**gradient 누적이 의도적으로 필요한 경우도 있다:**
메모리 제약으로 batch_size를 작게 설정할 때, 여러 step의 gradient를 모아
큰 배치처럼 흉내내는 "gradient accumulation" 기법에서 쓰인다.

### loss.backward() — 역전파

```python
# train.py line 138
loss.backward()
```

연산 그래프(computational graph)를 역방향으로 traversal하면서
각 파라미터에 대한 `∂loss/∂θ` (gradient)를 계산해 `.grad` 속성에 저장한다.

**역전파(Backpropagation) 원리:**
Chain rule (연쇄법칙)을 반복 적용한다.

```
∂loss/∂W1 = (∂loss/∂output) * (∂output/∂hidden) * (∂hidden/∂W1)
```

각 레이어의 gradient가 출력 레이어부터 입력 레이어 방향으로 전파된다.

### optimizer.step() — 파라미터 업데이트

```python
# train.py line 139
optimizer.step()
```

`.grad`에 저장된 gradient를 사용해 실제로 파라미터를 업데이트한다.
Adam의 경우 위에서 설명한 모멘트 계산 후 업데이트가 일어난다.

**전체 한 step 흐름:**
```
zero_grad() → forward() → loss 계산 → backward() → step()
```

---

## 7. model.train() vs model.eval()

```python
# train.py line 132
model.train()

# train.py line 142
model.eval()
```

| 메서드          | Dropout          | BatchNorm               |
|----------------|------------------|------------------------|
| `model.train()` | 확률적 비활성화 ON | 현재 배치 통계 사용       |
| `model.eval()`  | 비활성화 OFF      | 학습 중 누적된 통계 사용   |

**BatchNorm의 eval 모드:**
학습 중에는 각 미니배치의 평균/분산을 실시간 계산한다.
하지만 추론 시에는 배치 크기가 1이거나 배치가 없을 수 있으므로
학습 과정에서 exponential moving average로 누적한 전체 평균/분산을 사용한다.

---

## 8. torch.no_grad() — Gradient 계산 비활성화

```python
# train.py line 144
with torch.no_grad():
    for inputs, labels in val_loader:
        ...
```

검증/추론 시에는 파라미터를 업데이트하지 않으므로
gradient를 계산할 필요가 없다.

`torch.no_grad()` 블록 안에서는:
- 연산 그래프(computational graph)를 구성하지 않는다
- gradient 저장용 메모리를 할당하지 않는다
- 결과: 메모리 절약 + 약 2~3배 빠른 실행 속도

---

## 9. best_val_acc로 모델 저장하는 이유 (과적합 방지)

```python
# train.py line 129, 155-158
best_val_acc = 0.0
...
if val_acc > best_val_acc:
    best_val_acc = val_acc
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"  → 모델 저장 (best val_acc: {best_val_acc:.1f}%)")
```

### 과적합(Overfitting)이란?

학습이 진행될수록 모델이 학습 데이터의 노이즈까지 외우기 시작한다.
이때 학습 loss는 계속 내려가지만 검증 loss는 오히려 올라간다.

```
epoch 진행 →
훈련 정확도: 85% → 90% → 95% → 99%  (계속 증가)
검증 정확도: 85% → 88% → 87% → 83%  (중간 이후 감소)
                         ↑
                    여기서 저장해야 함
```

마지막 epoch의 모델을 저장하면 이미 과적합된 상태일 수 있다.
검증 정확도가 최고인 시점의 모델을 저장하면 새로운 데이터에 더 잘 일반화된다.

---

## 10. state_dict — 저장/로드 메커니즘

### state_dict란?

모델의 학습 가능한 파라미터(가중치, 편향)를 담은 Python 딕셔너리.

```python
# 예시 state_dict 구조
{
    'conv_layers.0.weight': tensor(...),  # Conv2d 가중치
    'conv_layers.0.bias':   tensor(...),  # Conv2d 편향
    'conv_layers.1.weight': tensor(...),  # BatchNorm 스케일(γ)
    'conv_layers.1.bias':   tensor(...),  # BatchNorm 이동(β)
    'conv_layers.1.running_mean': tensor(...),  # BatchNorm 누적 평균
    ...
    'fc_layers.0.weight': tensor(...),
    'fc_layers.0.bias':   tensor(...),
    ...
}
```

### 저장

```python
# train.py line 157
torch.save(model.state_dict(), MODEL_PATH)
```

state_dict만 저장하는 것이 권장된다.
모델 객체 전체를 저장하면 Python 버전, PyTorch 버전 의존성이 생긴다.

### 로드

```python
# inference.py line 83
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
```

1. `torch.load()`: 파일에서 딕셔너리를 역직렬화(deserialize)
2. `model.load_state_dict()`: 딕셔너리의 값을 모델의 해당 파라미터에 복사

---

## 11. train() 함수 전체 흐름

```python
# train.py line 98-162
def train():
```

### 단계별 흐름도

```
1. 장치 설정
   device = cuda if GPU 있으면, 없으면 cpu

2. 데이터 로드
   DrumDataset 생성 → 전체 wav 파일 경로 + 라벨 인덱스 수집

3. Train/Val 분리
   random_split(80% / 20%)
   DataLoader 두 개 생성 (train: shuffle=True, val: shuffle=False)

4. 모델/손실함수/옵티마이저 초기화
   DrumCNN().to(device)
   CrossEntropyLoss
   Adam(lr=0.001)

5. 데이터 분포 확인 출력
   클래스별 샘플 수 출력

6. 학습 루프 (NUM_EPOCHS=30 반복)
   ┌─────────────────────────────────────────────────────┐
   │ model.train()                                        │
   │                                                     │
   │ for inputs, labels in train_loader:  (25 step/epoch) │
   │     inputs, labels → .to(device)                    │
   │     optimizer.zero_grad()                           │
   │     loss = CrossEntropyLoss(model(inputs), labels)  │
   │     loss.backward()                                 │
   │     optimizer.step()                                │
   │     train_loss 누적                                  │
   │                                                     │
   │ model.eval()                                        │
   │ with torch.no_grad():                               │
   │     for inputs, labels in val_loader:               │
   │         preds = model(inputs).argmax(dim=1)         │
   │         correct += (preds == labels).sum()          │
   │                                                     │
   │ val_acc = correct / val_size * 100                  │
   │ avg_loss = train_loss / step수                       │
   │ 출력: "Epoch XX/30  loss: X.XXXX  val_acc: XX.X%"  │
   │                                                     │
   │ if val_acc > best_val_acc:                          │
   │     best_val_acc = val_acc                          │
   │     torch.save(model.state_dict(), MODEL_PATH)      │
   └─────────────────────────────────────────────────────┘

7. 완료 출력
   최고 검증 정확도, 저장 경로 출력
```

### 학습 완료 후 저장된 것

`drum_model.pth` 파일 하나.
안에는 DrumCNN의 모든 Conv/BN/Linear 파라미터가 딕셔너리로 직렬화되어 있다.
inference.py는 이 파일을 로드해서 같은 구조의 DrumCNN에 파라미터를 복원하고 추론한다.
