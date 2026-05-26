# Saved Models Guide

이 폴더는 팀이 학습한 모델 체크포인트를 함께 저장하고 공유하는 공간입니다.

## 목적

- 학습된 모델 파라미터를 저장한다.
- Adam optimizer 상태를 함께 저장해 이어서 학습할 수 있게 한다.
- 팀원이 같은 형식으로 저장하고 같은 방식으로 불러올 수 있게 한다.

## 권장 파일 구성

- `latest.pkl`: 가장 최근 학습 상태
- `best.pkl`: 가장 성능이 좋았던 학습 상태
- `epoch_010.pkl`, `epoch_020.pkl`: 중간 보관용 체크포인트

필요하면 나중에 `archive/` 폴더를 추가해 epoch별 파일을 따로 모아도 됩니다.

## 체크포인트에 저장할 내용

체크포인트는 아래 구조를 따르는 것을 권장합니다.

```python
checkpoint = {
    "model_params": model.params,
    "optimizer_state": {
        "m": optimizer.m,
        "v": optimizer.v,
        "t": optimizer.t,
        "lr": optimizer.lr,
    },
    "train_state": {
        "epoch": current_epoch,
        "loss_history": loss_history,
    },
    "model_config": {
        "use_batchnorm": True,
        "use_dropout": True,
        "dropout_ratio": 0.5,
    },
}
```

## 저장 방법

학습이 끝난 뒤 또는 epoch 중간 저장 시 아래처럼 `pickle`로 저장합니다.

```python
import pickle

checkpoint = {
    "model_params": model.params,
    "optimizer_state": {
        "m": optimizer.m,
        "v": optimizer.v,
        "t": optimizer.t,
        "lr": optimizer.lr,
    },
    "train_state": {
        "epoch": current_epoch,
        "loss_history": loss_history,
    },
    "model_config": {
        "use_batchnorm": True,
        "use_dropout": True,
        "dropout_ratio": 0.5,
    },
}

with open("saved_models/latest.pkl", "wb") as f:
    pickle.dump(checkpoint, f)
```

성능이 가장 좋을 때는 `best.pkl`로도 함께 저장하면 됩니다.

```python
with open("saved_models/best.pkl", "wb") as f:
    pickle.dump(checkpoint, f)
```

## 불러오기 방법

같은 구조의 모델과 optimizer를 먼저 만든 뒤 체크포인트를 읽어옵니다.

```python
import pickle
from network import NeuralNetwork
from optimizers import Adam

with open("saved_models/latest.pkl", "rb") as f:
    checkpoint = pickle.load(f)

config = checkpoint["model_config"]

model = NeuralNetwork(
    use_batchnorm=config["use_batchnorm"],
    use_dropout=config["use_dropout"],
    dropout_ratio=config["dropout_ratio"],
)
optimizer = Adam(lr=checkpoint["optimizer_state"]["lr"])

model.params = checkpoint["model_params"]
optimizer.m = checkpoint["optimizer_state"]["m"]
optimizer.v = checkpoint["optimizer_state"]["v"]
optimizer.t = checkpoint["optimizer_state"]["t"]

current_epoch = checkpoint["train_state"]["epoch"]
loss_history = checkpoint["train_state"]["loss_history"]
```

이후에는 `current_epoch` 다음부터 다시 학습을 이어가면 됩니다.

## 팀 공용 규칙

- 모든 팀원은 `saved_models/` 폴더에 같은 형식으로 저장합니다.
- 체크포인트 key 이름은 `model_params`, `optimizer_state`, `train_state`, `model_config`로 통일합니다.
- 이어서 학습하려면 저장할 때와 불러올 때 모델 구조가 같아야 합니다.
- `latest.pkl`은 가장 최근 상태, `best.pkl`은 가장 성능이 좋은 상태로 사용합니다.
- 동시에 같은 파일을 덮어쓰지 않도록 주의합니다.

## 주의 사항

- `model.params`만 저장하면 추론은 가능하지만 Adam 상태가 없어서 학습을 자연스럽게 이어가기 어렵습니다.
- BatchNorm, Dropout, 층 수 같은 모델 구조가 달라지면 이전 체크포인트를 그대로 쓰기 어렵습니다.
- 노트북에서 학습 중이라면 셀이 끝난 뒤 저장 코드를 실행하는 것이 안전합니다.
