# PyTorch 기초 — 실제 코드 기반 설명

이 문서는 이 프로젝트(`train.py`, `inference.py`)에서 사용한
PyTorch의 핵심 개념들을 실제 코드 라인과 함께 설명한다.

---

## 1. nn.Module 상속

### 왜 상속하는가?

```python
# train.py line 60-61
class DrumCNN(nn.Module):
    def __init__(self):
        super().__init__()
```

`nn.Module`을 상속하면 PyTorch의 파라미터 추적, 저장/로드, GPU 이동 등
모든 인프라를 자동으로 사용할 수 있다.

### `__call__` → `forward` 자동 호출

`nn.Module`은 `__call__` 메서드를 구현하고 있다.
사용자가 모델을 함수처럼 호출하면 `__call__`이 실행되고,
그 내부에서 `forward()`를 호출한다.

```python
# 이렇게 쓰면
output = model(inputs)

# 내부적으로 이렇게 동작한다
# model.__call__(inputs)
#     → hook 처리 (등록된 훅이 있다면)
#     → model.forward(inputs)
#     → hook 처리
#     → output 반환
```

절대로 `model.forward(inputs)`를 직접 호출하지 말것.
`__call__`을 통해야 gradient hook, BatchNorm 통계 업데이트 등이 제대로 동작한다.

```python
# train.py line 93-95
def forward(self, x):
    x = self.conv_layers(x).flatten(1)
    return self.fc_layers(x)
```

---

## 2. nn.Sequential — 레이어 묶기

```python
# train.py line 64-78
self.conv_layers = nn.Sequential(
    nn.Conv2d(1, 16, kernel_size=3, padding=1),
    nn.BatchNorm2d(16),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Dropout2d(0.25),
    nn.Conv2d(16, 32, kernel_size=3, padding=1),
    ...
)
```

`nn.Sequential`은 레이어들을 순서대로 묶어 하나의 모듈로 만든다.
`sequential(x)`를 호출하면 입력이 첫 번째 레이어부터 마지막 레이어까지 차례대로 통과한다.

**파라미터 자동 추적:**
`nn.Sequential` 안에 넣은 모든 레이어는 부모 모듈(`DrumCNN`)의 파라미터로 자동 등록된다.
`model.parameters()`로 가져오면 `conv_layers` 안의 Conv, BN 파라미터도 모두 포함된다.

단순히 Python 리스트에 레이어를 담으면 PyTorch가 그 레이어들을 모듈로 인식하지 못한다.
반드시 `nn.Sequential`, `nn.ModuleList`, `nn.ModuleDict` 등으로 감싸야 한다.

---

## 3. model.parameters() — 가중치 추적 원리

```python
# train.py line 117
optimizer = optim.Adam(model.parameters(), lr=LR)
```

`model.parameters()`는 모델 안의 모든 학습 가능한 파라미터를 iterator로 반환한다.

**추적 원리:**
`nn.Module`에 속성으로 `nn.Parameter` 또는 다른 `nn.Module`을 할당하면
PyTorch가 자동으로 등록(register)한다.

```python
# 내부 동작 (개념적 설명)
# __setattr__이 오버라이드되어 있음
# self.conv_layers = nn.Sequential(...) 할당 시
# PyTorch가 _modules 딕셔너리에 'conv_layers'로 등록
```

`parameters()`는 `_modules` 딕셔너리를 재귀적으로 탐색해 모든 파라미터를 수집한다.
Adam은 이 파라미터 리스트 각각에 대해 모멘트 상태(m, v)를 별도로 유지한다.

---

## 4. torch.tensor vs np.array

| 항목           | `torch.tensor`        | `np.array`           |
|---------------|----------------------|---------------------|
| 연산 추적      | gradient 추적 가능     | gradient 추적 불가    |
| GPU 지원       | .to(device)로 이동 가능 | CPU only             |
| 자동 미분       | autograd 지원          | 미지원               |
| 상호 변환       | .numpy() / torch.from_numpy() | 가능           |

```python
# train.py line 57
return torch.tensor(mel_db).unsqueeze(0), torch.tensor(label, dtype=torch.long)
```

`mel_db`는 `np.float32` 배열이다.
`torch.tensor(mel_db)`는 데이터를 복사해서 PyTorch 텐서를 만든다.
`dtype=torch.long`은 CrossEntropyLoss가 정수 타입의 라벨을 요구하기 때문이다.

**torch.from_numpy()와의 차이:**
`torch.from_numpy(arr)`는 데이터를 복사하지 않고 메모리를 공유한다.
numpy 배열 변경 시 텐서도 바뀐다. `torch.tensor()`는 항상 복사한다.

---

## 5. .to(device) — GPU/CPU 메모리 이동

```python
# train.py line 99-100, 115, 135
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DrumCNN().to(device)
inputs, labels = inputs.to(device), labels.to(device)
```

`.to(device)`는 텐서 또는 모델의 데이터를 지정한 장치의 메모리로 이동시킨다.

**모델과 데이터가 같은 device에 있어야 한다:**
모델이 GPU에 있는데 입력 텐서가 CPU에 있으면 오류가 발생한다.
그래서 매 배치마다 `inputs.to(device)`로 이동시킨다.

**inference.py의 경우:**

```python
# inference.py line 92
tensor = torch.tensor(mel_db).unsqueeze(0).unsqueeze(0).to(device)
```

텐서 생성 즉시 device로 이동한다.

**.to(device)는 in-place가 아니다:**
`tensor.to(device)` 결과를 반드시 변수에 받아야 한다.
`tensor.to(device)` 만 쓰고 결과를 무시하면 이동이 일어나지 않는다.
단, `model.to(device)`는 in-place로 동작한다.

---

## 6. tensor.shape

```python
# train.py line 84
flat_size = self.conv_layers(dummy).flatten(1).shape[1]
```

`shape`은 텐서의 각 차원 크기를 담은 `torch.Size` 객체다.
Python 튜플처럼 인덱스로 접근 가능하다.

```python
t = torch.zeros(1, 32, 16, 5)
t.shape        # torch.Size([1, 32, 16, 5])
t.shape[0]     # 1  (배치 크기)
t.shape[1]     # 32 (채널 수)
t.shape[2]     # 16 (높이)
t.shape[3]     # 5  (너비)
t.ndim         # 4  (차원 수)
```

`flat_size`는 `conv_layers` 통과 후 Flatten한 결과의 두 번째 차원,
즉 `32 * 16 * 5 = 2560`이다.

---

## 7. torch.zeros — 더미 데이터 생성

```python
# train.py line 83
dummy = torch.zeros(1, 1, N_MELS, n_frames)  # (1, 1, 64, 22)
```

모든 원소가 0인 텐서를 생성한다.
이 프로젝트에서는 `conv_layers`를 통과시켜 Flatten 후 크기를 동적으로 계산하는 데 쓴다.

**왜 더미 데이터를 쓰는가?**
Linear 레이어의 `in_features`를 하드코딩하지 않기 위해서다.
`N_MELS`나 `HOP_LENGTH`를 바꾸면 conv 통과 후 크기가 달라지는데,
더미 데이터로 자동 계산하면 설정값이 바뀌어도 코드 수정이 불필요하다.

`torch.zeros`는 gradient 추적을 하지 않는다 (`requires_grad=False`가 기본).
크기 계산 목적이므로 gradient가 필요없어 적합한 선택이다.

다른 초기화:
- `torch.ones(shape)`: 모두 1
- `torch.rand(shape)`: 0~1 균등분포 난수
- `torch.randn(shape)`: 표준정규분포 난수

---

## 8. torch.softmax

```python
# inference.py line 95
probs = torch.softmax(model(tensor), dim=1)[0]
```

### dim 파라미터의 의미

`dim`은 softmax를 적용할 차원을 지정한다.
"이 차원 안의 값들을 확률로 변환한다"는 의미다.

모델 출력 shape: `(1, 3)` (배치 1, 클래스 3)

```python
torch.softmax(output, dim=1)
# dim=1 (클래스 차원)에 대해 softmax
# [0.659, 0.242, 0.099] 이 세 값의 합 = 1.0
```

`dim=0`을 쓰면 배치 차원에 대해 softmax가 적용되어 의미가 없다.
분류 문제에서는 항상 클래스 차원에 대해 softmax를 적용해야 한다.

---

## 9. .argmax()와 .item()

### .argmax()

가장 큰 값의 **인덱스**를 반환한다.

```python
# inference.py line 96
pred = probs.argmax().item()

# train.py line 147
preds = model(inputs).argmax(dim=1)
```

`probs = tensor([0.659, 0.242, 0.099])`
`probs.argmax()` → `tensor(0)` (인덱스 0이 가장 크다)

`dim=1` 인자는 배치 내 각 샘플에 대해 클래스 차원의 argmax를 구한다:
- 입력 shape: `(32, 3)`
- 출력 shape: `(32,)` — 각 샘플의 예측 클래스 인덱스

### .item()

PyTorch 텐서에서 Python 스칼라(기본 숫자 타입)를 추출한다.

```python
pred = probs.argmax().item()  # tensor(0) → int 0
```

`tensor(0)`은 PyTorch 텐서이므로 Python 정수처럼 바로 사용할 수 없다.
`.item()`을 호출하면 Python `int` 또는 `float`로 변환된다.

```python
CLASSES[pred]  # pred가 Python int여야 인덱싱 가능
float(probs[pred])  # Python float로 변환
```

```python
# inference.py line 98
return CLASSES[pred], float(probs[pred])
```

`float(probs[pred])`: 텐서를 Python float으로 변환. `.item()`과 동일한 효과.

---

## 10. torch.save / torch.load / load_state_dict

### torch.save

```python
# train.py line 157
torch.save(model.state_dict(), MODEL_PATH)
```

Python의 `pickle`을 기반으로 객체를 직렬화해 파일에 저장한다.
`state_dict()`는 파라미터 이름(str) → 텐서의 딕셔너리다.

### torch.load

```python
# inference.py line 83
torch.load(MODEL_PATH, map_location=device, weights_only=True)
```

파일에서 직렬화된 객체를 역직렬화해 메모리에 로드한다.
`map_location=device`: 파일이 GPU에서 저장됐더라도 지정한 device로 로드한다.
GPU 없는 환경에서 `map_location='cpu'`를 쓰면 GPU 텐서를 CPU로 로드할 수 있다.

**weights_only=True:**
신뢰할 수 없는 파일을 열 때 임의 코드 실행(RCE) 위험을 방지한다.
`weights_only=True`는 텐서와 딕셔너리 등 안전한 타입만 역직렬화한다.
PyTorch 2.0+ 에서 보안 모범 사례로 권장된다.

### load_state_dict

```python
# inference.py line 83
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
```

로드한 딕셔너리의 각 키(파라미터 이름)를 모델의 해당 파라미터에 복사한다.

**strict=True (기본값):**
딕셔너리의 키와 모델의 파라미터 이름이 정확히 일치해야 한다.
inference.py의 DrumCNN 정의가 train.py와 완전히 같아야 하는 이유다.
설정값(`N_MELS`, `HOP_LENGTH`, `CLASSES`)도 동일해야 하며,
두 파일 상단에 "완전히 동일하게 맞춰야 함"이라는 주석이 있다:

```python
# inference.py line 9-10
# ---------------------------------------------------------------
# train.py와 완전히 동일하게 맞춰야 함
# ---------------------------------------------------------------
```

---

## 11. CUDA 관련 — torch.cuda.is_available()

```python
# train.py line 99, inference.py line 177
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

`torch.cuda.is_available()`: CUDA 지원 GPU가 있으면 `True`, 없으면 `False`.
이 조건식으로 GPU가 있으면 GPU를 쓰고, 없으면 CPU를 쓰는 코드가 된다.

### torch.device 객체

`torch.device('cuda')` 또는 `torch.device('cpu')`.
`.to(device)` 호출 시 이 객체를 인자로 전달한다.

**여러 GPU가 있는 경우:**
`torch.device('cuda:0')`, `torch.device('cuda:1')` 으로 특정 GPU를 지정한다.
이 프로젝트는 단일 GPU만 고려하므로 `'cuda'`(기본적으로 `cuda:0`)를 사용한다.

### 학습 vs 추론의 device 흐름

**학습 (train.py):**
```
1. device 결정 (cuda/cpu)
2. model.to(device)          → 모델 파라미터를 device 메모리로
3. inputs.to(device)         → 배치 데이터를 device 메모리로
4. labels.to(device)
5. loss 계산, backward, step → 모두 device에서 수행
```

**추론 (inference.py):**
```
1. device 결정 (cuda/cpu)
2. model = DrumCNN().to(device)
3. model.load_state_dict(...)  → 저장된 파라미터 복원
4. model.eval()
5. tensor = ...unsqueeze(0).unsqueeze(0).to(device)  → 입력도 device로
6. with torch.no_grad(): model(tensor)               → device에서 추론
7. probs.argmax().item()       → Python 숫자로 가져오기
```

결과값(`.item()`, `float()`)은 CPU 메모리로 자동 복사되므로
`tensor.cpu().item()` 같이 명시적으로 CPU로 옮길 필요가 없다.
