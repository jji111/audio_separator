# CNN 기초부터 DrumCNN까지

이 문서는 합성곱 신경망(CNN)의 핵심 개념을 수식과 함께 설명하고,
이 프로젝트의 `DrumCNN`이 실제로 데이터를 어떻게 처리하는지를 레이어별 shape 변화로 추적한다.

---

## 1. Convolution 연산 (합성곱)

### 개념

Convolution은 작은 필터(커널)를 입력 위에서 슬라이딩시키면서
각 위치에서 필터와 입력의 요소별 곱을 합산하는 연산이다.
이미지에서는 "이 위치 주변에 이 패턴이 있는가?"를 수치로 표현한다.

### 수식

입력을 `I`, 필터를 `K`, 출력을 `O`라 할 때:

```
O[i, j] = sum_m sum_n  I[i*s + m, j*s + n] * K[m, n]
```

- `i, j` : 출력의 위치 인덱스
- `m, n` : 필터 내부 인덱스
- `s`    : stride (이동 칸 수)

필터가 2D 이미지 위를 왼쪽→오른쪽, 위→아래 방향으로 한 칸씩(또는 s칸씩) 이동하면서
필터 크기만큼의 영역과 내적(dot product)을 계산한다.
그 결과값 하나가 출력 특징맵(feature map)의 한 픽셀이 된다.

### 시각적 설명

```
입력 5x5          필터 3x3           출력 3x3 (stride=1, padding=0)
┌─────────────┐   ┌────────┐         ┌──────────┐
│ 1  2  3  0  1│  │ 1  0 -1│         │  ?  ?  ? │
│ 0  1  2  3  0│  │ 1  0 -1│  →      │  ?  ?  ? │
│ 1  0  1  2  3│  │ 1  0 -1│         │  ?  ?  ? │
│ 2  1  0  1  2│  └────────┘         └──────────┘
│ 1  2  3  2  1│
└─────────────┘
```

필터가 왼쪽 위 3x3 영역에 놓이면:
`O[0,0] = 1*1 + 2*0 + 3*(-1) + 0*1 + 1*0 + 2*(-1) + 1*1 + 0*0 + 1*(-1) = -4`

---

## 2. kernel_size, padding, stride

### kernel_size

필터의 가로/세로 크기. `kernel_size=3`이면 3×3 필터를 사용한다.

- 작을수록(3×3) 미세한 패턴, 클수록(5×5, 7×7) 넓은 영역의 패턴을 감지
- 이 프로젝트는 모든 Conv 레이어에 `kernel_size=3` 사용

```python
# train.py line 66
nn.Conv2d(1, 16, kernel_size=3, padding=1)
```

### padding

입력 가장자리에 0을 추가하는 픽셀 수.
`padding=1`이면 상하좌우 각 1줄씩 0으로 채운다.

**왜 필요한가?**
padding 없이 3×3 필터를 적용하면 출력이 입력보다 작아진다.
`padding=1`을 쓰면 `kernel_size=3`일 때 출력 크기 = 입력 크기가 유지된다.

### stride

필터가 한 번에 이동하는 칸 수. 기본값 1.
`stride=2`이면 출력 크기가 절반이 된다.

### 출력 크기 계산 공식

```
출력 크기 = floor( (입력 크기 + 2*padding - kernel_size) / stride ) + 1
```

예시: 입력 64, kernel_size=3, padding=1, stride=1
```
출력 = floor( (64 + 2*1 - 3) / 1 ) + 1 = floor(63) + 1 = 64
```
padding=1, kernel_size=3이면 stride=1일 때 크기가 보존된다.

---

## 3. in_channels, out_channels (채널과 특징맵)

### in_channels

입력 텐서의 채널 수.
- 흑백 이미지: 1채널
- RGB 이미지: 3채널
- 이 프로젝트의 멜스펙트로그램: 1채널 (흑백처럼 에너지 값 하나)

### out_channels

이 레이어가 출력하는 특징맵(feature map)의 개수.
필터를 `out_channels`개 독립적으로 학습시키기 때문에,
각 필터는 서로 다른 패턴(수직선, 수평선, 특정 주파수 변화 등)을 감지한다.

```
Conv2d(in_channels, out_channels, ...)
```

```python
# train.py line 66, 73
nn.Conv2d(1,  16, kernel_size=3, padding=1)  # 1채널 입력 → 16개 특징맵
nn.Conv2d(16, 32, kernel_size=3, padding=1)  # 16채널 입력 → 32개 특징맵
```

**두 번째 블록에서 채널을 32로 늘리는 이유:**
첫 번째 레이어가 저수준 패턴(엣지, 주파수 경계)을 감지했다면,
두 번째 레이어는 그 조합으로 더 복잡한 패턴(킥의 저주파 에너지 형태 등)을 감지한다.
채널 수를 늘리면 표현력이 증가한다.

---

## 4. BatchNormalization

### 왜 필요한가?

딥러닝 학습 중 각 레이어의 입력 분포가 계속 변화한다.
앞쪽 레이어의 가중치가 조금 바뀌면 뒤쪽 레이어 입력의 평균/분산이 크게 달라진다.
이를 **Internal Covariate Shift**라고 한다.
이 현상은 학습을 느리게 하고 높은 학습률을 쓰기 어렵게 만든다.

### 수식

미니배치 B = {x_1, ..., x_m} 에 대해:

```
1. 배치 평균:     μ_B = (1/m) * sum(x_i)
2. 배치 분산:     σ²_B = (1/m) * sum((x_i - μ_B)²)
3. 정규화:        x̂_i = (x_i - μ_B) / sqrt(σ²_B + ε)
4. 스케일/이동:   y_i = γ * x̂_i + β
```

- `ε` : 분모가 0이 되지 않도록 더하는 작은 값 (보통 1e-5)
- `γ, β` : 학습 가능한 파라미터 (스케일, 이동)

### 효과

- 각 레이어의 입력을 평균 0, 분산 1 근처로 유지
- 더 높은 학습률 사용 가능 → 빠른 수렴
- 일종의 regularization 효과 → 과적합 억제

```python
# train.py line 67, 74
nn.BatchNorm2d(16)  # 16채널 특징맵 각각을 정규화
nn.BatchNorm2d(32)
```

`BatchNorm2d`는 채널 차원별로 독립적으로 정규화한다.
즉, 16개 특징맵이 있으면 각 특징맵마다 별도의 γ, β를 학습한다.

---

## 5. ReLU 활성화 함수

### 수식

```
ReLU(x) = max(0, x)
```

음수는 0으로, 양수는 그대로 통과시킨다.

### 다른 활성화함수와 비교

| 함수    | 수식                             | 출력 범위    | 문제점                        |
|---------|---------------------------------|------------|-------------------------------|
| Sigmoid | `1 / (1 + e^(-x))`              | (0, 1)     | gradient vanishing, 느린 수렴  |
| Tanh    | `(e^x - e^(-x))/(e^x + e^(-x))` | (-1, 1)    | gradient vanishing (덜하지만) |
| ReLU    | `max(0, x)`                     | [0, ∞)     | Dying ReLU (음수 구역 죽음)    |

### Gradient Vanishing 문제

Sigmoid의 도함수: `σ'(x) = σ(x)(1 - σ(x))`

x가 크거나 작으면 σ(x)는 0 또는 1에 수렴하고, `σ'(x) → 0`이 된다.
역전파 시 이 값이 여러 레이어에 걸쳐 곱해지면 gradient가 0에 가까워져
앞쪽 레이어의 가중치가 거의 업데이트되지 않는다.

ReLU는 양수 구간에서 도함수가 항상 1이므로 gradient vanishing이 없다.

```python
# train.py line 68, 76
nn.ReLU()
```

---

## 6. MaxPooling

### 동작 원리

지정한 크기의 윈도우 안에서 최대값 하나만 남기고 나머지를 버린다.
`MaxPool2d(2)`는 2×2 영역에서 최대값을 뽑고, stride=2로 이동한다.

```
입력 4x4          MaxPool2d(2)      출력 2x2
┌──────────┐                        ┌──────┐
│ 1  3  2  1│   각 2x2 구역에서      │ 3  2 │
│ 4  2  1  3│   최대값 선택    →     │ 4  3 │
│ 1  2  3  2│                        └──────┘
│ 3  1  2  1│
└──────────┘
```

### 크기가 절반이 되는 이유

`MaxPool2d(kernel_size=2, stride=2)`는 2×2 블록을 겹치지 않게 순서대로 처리한다.
가로 64 → 32, 세로 22 → 11 처럼 각 차원이 정확히 절반이 된다.

### 역할

- 공간 해상도를 줄여 계산량과 메모리를 절약
- 위치 불변성(translation invariance): 패턴이 조금 이동해도 같은 출력을 낼 수 있음
- 과적합 억제 효과

---

## 7. Dropout

### 동작 원리

학습 중 각 뉴런을 확률 p로 랜덤하게 비활성화(0으로 만든다).
비활성화된 뉴런은 그 배치에서 순전파와 역전파 모두 참여하지 않는다.

### 학습/추론 모드 차이

| 모드   | 동작                              |
|--------|----------------------------------|
| 학습   | 각 뉴런을 p 확률로 비활성화         |
| 추론   | 모든 뉴런 활성화, 출력에 (1-p) 곱함 |

PyTorch는 `model.train()` / `model.eval()` 호출로 자동 전환된다.

```python
# train.py line 70, 77  (공간 Dropout - 2D 채널 단위)
nn.Dropout2d(0.25)   # 채널 전체를 25% 확률로 0으로 만듦

# train.py line 89  (일반 Dropout - 뉴런 단위)
nn.Dropout(0.5)      # FC 레이어 뉴런을 50% 확률로 비활성화
```

`Dropout2d`는 채널 전체를 한꺼번에 끈다. 특징맵 하나가 통째로 0이 된다.
이는 공간적으로 상관관계가 높은 특징맵에 더 효과적인 정규화가 된다.

---

## 8. Flatten

### 왜 필요한가?

Conv/Pool 레이어들은 3D 텐서 `(채널, 높이, 너비)`를 출력한다.
Linear(FC) 레이어는 1D 벡터를 입력으로 받는다.
이 차원 불일치를 해결하기 위해 Flatten이 필요하다.

### Shape 변화

```
(배치, 채널, 높이, 너비)  →  (배치, 채널*높이*너비)
(1, 32, 16, 5)           →  (1, 2560)
```

```python
# train.py line 94
x = self.conv_layers(x).flatten(1)
```

`.flatten(1)`은 dim=1 이후의 모든 차원을 하나로 합친다.
배치 차원(dim=0)은 그대로 유지한다.

### 동적 크기 계산

이 프로젝트는 Flatten 후의 크기를 하드코딩하지 않고 더미 데이터로 자동 계산한다:

```python
# train.py line 82-84
n_frames  = 1 + N_SAMPLES // HOP_LENGTH   # 1 + 11025 // 512 = 22
dummy     = torch.zeros(1, 1, N_MELS, n_frames)   # (1, 1, 64, 22)
flat_size = self.conv_layers(dummy).flatten(1).shape[1]
```

이렇게 하면 `N_MELS`나 `HOP_LENGTH`를 바꿔도 `fc_layers`가 자동으로 맞춰진다.

---

## 9. Linear(FC) 레이어

### 수식 (행렬 곱 관점)

```
y = x @ W^T + b
```

- `x` : 입력 벡터 (배치, in_features)
- `W` : 가중치 행렬 (out_features, in_features)
- `b` : 편향 벡터 (out_features,)
- `y` : 출력 벡터 (배치, out_features)

입력 벡터의 모든 뉴런이 출력 뉴런 하나하나에 연결되므로 Fully Connected라고 부른다.

```python
# train.py line 87, 90
nn.Linear(flat_size, 128)  # flat_size → 128
nn.Linear(128, len(CLASSES))  # 128 → 3
```

두 번째 Linear의 출력 3은 클래스 수(`kick`, `snare`, `hihat`)에 대응한다.

---

## 10. Softmax

### 수식

입력 벡터 `z = [z_1, z_2, ..., z_K]`에 대해:

```
softmax(z_i) = e^(z_i) / sum_{j=1}^{K} e^(z_j)
```

### 확률로 변환되는 원리

- 지수함수 `e^x`는 항상 양수 → 출력이 모두 양수
- 합산으로 나눔 → 출력의 합 = 1
- 따라서 각 값을 확률로 해석 가능

예시: `z = [2.0, 1.0, 0.1]`
```
e^2.0 = 7.389,  e^1.0 = 2.718,  e^0.1 = 1.105
합계 = 11.212
softmax = [0.659, 0.242, 0.099]  → 합 = 1.0
```

```python
# inference.py line 95
probs = torch.softmax(model(tensor), dim=1)[0]
```

`dim=1`은 클래스 차원(배치 내 각 샘플의 3개 클래스)에 대해 softmax를 계산한다.
`[0]`은 배치 크기 1에서 첫 번째(유일한) 샘플을 꺼낸다.

---

## 11. argmax

가장 큰 값의 **인덱스**를 반환한다.

```python
# inference.py line 96
pred = probs.argmax().item()
```

`probs = [0.659, 0.242, 0.099]`이면 `argmax() = 0` (kick)

```python
# train.py line 147
preds = model(inputs).argmax(dim=1)
```

학습 검증 시에는 softmax 없이 로짓(logit) 값의 argmax를 사용한다.
softmax는 단조증가 변환이므로 argmax 결과는 동일하다.

---

## 12. DrumCNN 전체 데이터 흐름

### 입력 텐서 shape

```
n_frames = 1 + 11025 // 512 = 22
입력: (1, 1, 64, 22)
     (배치, 채널, N_MELS, n_frames)
```

### 각 레이어별 shape 변화

```
입력                         (1,  1, 64, 22)
                              ↓
Conv2d(1→16, 3×3, pad=1)    (1, 16, 64, 22)   # 크기 유지 (pad=1, stride=1)
BatchNorm2d(16)              (1, 16, 64, 22)   # 값만 정규화, shape 불변
ReLU                         (1, 16, 64, 22)   # 값만 clip, shape 불변
MaxPool2d(2)                 (1, 16, 32, 11)   # 가로/세로 각각 절반
Dropout2d(0.25)              (1, 16, 32, 11)   # 훈련 시 일부 채널 0, shape 불변
                              ↓
Conv2d(16→32, 3×3, pad=1)   (1, 32, 32, 11)   # 채널 16→32, 크기 유지
BatchNorm2d(32)              (1, 32, 32, 11)
ReLU                         (1, 32, 32, 11)
MaxPool2d(2)                 (1, 32, 16,  5)   # 32/2=16, floor(11/2)=5
Dropout2d(0.25)              (1, 32, 16,  5)
                              ↓
flatten(1)                   (1, 2560)          # 32*16*5 = 2560
                              ↓
Linear(2560→128)             (1, 128)
ReLU                         (1, 128)
Dropout(0.5)                 (1, 128)
Linear(128→3)                (1, 3)             # 최종 로짓 3개
```

MaxPool2d(2) 두 번째에서 n_frames=22 → 11 → floor(11/2)=5가 된다.
11은 홀수이므로 내림(floor)하여 5가 된다.

### 최종 출력

`(1, 3)` 텐서. 3개 값은 각각 `kick`, `snare`, `hihat`의 로짓(점수)이다.
softmax를 통과하면 3개 값의 합이 1인 확률 분포가 된다.

---

## 13. unsqueeze(0) 두 번의 의미

학습(`train.py`)과 추론(`inference.py`)의 텐서 준비 과정 차이:

### 학습 시 (train.py line 57)

```python
return torch.tensor(mel_db).unsqueeze(0), torch.tensor(label, dtype=torch.long)
```

`mel_db`의 shape: `(64, 22)` (N_MELS × n_frames, 2D 행렬)

`unsqueeze(0)`: dim=0에 차원 추가 → `(1, 64, 22)` (채널 차원 추가)

이것이 `DataLoader`를 통해 배치로 묶이면:
`(배치크기, 1, 64, 22)` ex) `(32, 1, 64, 22)`

DataLoader가 첫 번째 차원을 자동으로 쌓아(stack) 배치를 만든다.

### 추론 시 (inference.py line 92)

```python
tensor = torch.tensor(mel_db).unsqueeze(0).unsqueeze(0).to(device)
```

`mel_db`의 shape: `(64, 22)` (동일)

- 첫 번째 `unsqueeze(0)`: `(64, 22)` → `(1, 64, 22)` (채널 차원)
- 두 번째 `unsqueeze(0)`: `(1, 64, 22)` → `(1, 1, 64, 22)` (배치 차원)

추론 시에는 DataLoader 없이 단일 클립을 바로 모델에 넣기 때문에
배치 차원도 직접 만들어줘야 한다.
모델은 항상 4D 텐서 `(배치, 채널, 높이, 너비)`를 기대하기 때문이다.

| 상황  | mel_db shape | unsqueeze 횟수 | 최종 shape       |
|-------|-------------|---------------|-----------------|
| 학습  | (64, 22)    | 1번 (채널)    | (1, 64, 22) → DataLoader가 배치 추가 |
| 추론  | (64, 22)    | 2번 (채널+배치) | (1, 1, 64, 22) |
