import os
import zipfile
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ZIP_PATH = os.path.join(BASE_DIR, 'IDMT-SMT-DRUMS-V2.zip')
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
LABELS = ['kick', 'snare', 'hihat']

if not os.path.exists(ZIP_PATH):
    print(f"IDMT-SMT-DRUMS-V2.zip 파일이 없어요")
    exit()

# 이미 데이터가 있으면 굳이 다시 안 해도 됨
if os.path.exists(DATASET_DIR):
    existing = [
        f for label in LABELS
        for f in os.listdir(os.path.join(DATASET_DIR, label))
        if f.endswith('.wav') and os.path.exists(os.path.join(DATASET_DIR, label))
    ]
    if existing:
        print(f"데이터셋이 이미 있어요 (총 {len(existing)}개). 다시 만들려면 dataset 폴더를 지우고 실행하세요.")
        exit()
    shutil.rmtree(DATASET_DIR)

for label in LABELS:
    os.makedirs(os.path.join(DATASET_DIR, label), exist_ok=True)

print("압축 푸는 중...")
temp_dir = os.path.join(BASE_DIR, 'temp_extract')
with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
    zip_ref.extractall(temp_dir)

# 파일명에 #KD#, #SD#, #HH# 가 붙어 있어서 그걸로 분류
print("kick / snare / hihat 폴더로 분류 중...")
count = {label: 0 for label in LABELS}

for root, dirs, files in os.walk(temp_dir):
    for file in files:
        if not file.endswith('.wav'):
            continue
        src = os.path.join(root, file)
        if '#KD#' in file:
            target = 'kick'
        elif '#SD#' in file:
            target = 'snare'
        elif '#HH#' in file:
            target = 'hihat'
        else:
            continue
        shutil.move(src, os.path.join(DATASET_DIR, target, file))
        count[target] += 1

shutil.rmtree(temp_dir)

print("완료!")
for label in LABELS:
    print(f"  {label}: {count[label]}개")
print(f"\n다음 단계: train.py 실행")
