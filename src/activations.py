# -*- coding: utf-8 -*-
"""
활성화 함수 모음.

학생 구현 대상:
- ReLU.forward, ReLU.backward
- Softmax.forward, Softmax.backward
"""

import numpy as np


class ReLU:
    """
    ReLU(Rectified Linear Unit) 활성화 함수.

    은닉층에서 음수 값은 0으로 막고, 양수 값은 그대로 통과시킵니다.
    forward에서 만든 mask는 backward 때 "어느 위치로 gradient를 흘릴지" 결정하는 데 사용됩니다.
    
    mask = 어떤 위치를 통과시키고 어떤 위치를 막을지 표시하는 True/False 배열 (입력 x 에서 양수였던 위치 True/ 0이나 음수였던 위치 False)
    Ture/ False는 누가 바꿈? 
    -> NumPy 배열에 조건을 걸면, NumPy가 그 조건을 배열의 원소마다 하나씩 적용해서 True/False 배열로 만들어준다.

    np.where(조건, 조건이 True일 때 쓸 값, 조건이 False일 때 쓸 값)
    """

    def forward(self, x):
        """
        Args:
            x: 임의 shape의 입력 배열
        Returns:
            x와 같은 shape. x > 0인 위치만 원래 값을 유지합니다.
        """
        self.mask = x > 0
        out = np.where(self.mask,x,0)
        return out
        # TODO: x > 0 위치를 self.mask에 저장하고, 음수/0 위치는 0으로 바꾸세요.
        raise NotImplementedError("ReLU.forward를 구현하세요.")

    def backward(self, dout):
        """
        Args:
            dout: 다음 층에서 넘어온 gradient

        Returns:
            ReLU 입력 x에 대한 gradient. forward 때 x <= 0이었던 위치는 0입니다.
        """
        # TODO: forward에서 저장한 self.mask를 이용해 gradient가 흐를 위치만 남기세요.
        # 써야하는 변수 : dx
        dx = np.where(self.mask,dout,0)
        return dx

        raise NotImplementedError("ReLU.backward를 구현하세요.")


class Softmax:
    """
    Softmax 출력층.

    각 샘플의 로짓(logit)을 클래스별 확률로 바꿉니다.
    exp 계산 전에 행별 최댓값을 빼면 큰 숫자에서 overflow가 나는 것을 줄일 수 있습니다.
    """

    def forward(self, x):
        """
        Args:
            x: (batch_size, num_classes) 로짓

        Returns:
            (batch_size, num_classes) 확률. 각 행의 합은 1입니다.
        """
        # TODO: 수치 안정성을 위해 row별 max를 뺀 뒤 softmax 확률을 계산하세요.
        # 힌트: np.max(..., axis=1, keepdims=True), np.exp, np.sum을 사용합니다.
        raise NotImplementedError("Softmax.forward를 구현하세요.")

    def backward(self, dout):
        """
        Softmax와 Cross Entropy를 함께 미분한 gradient를 train()에서 직접 만들기 때문에
        여기서는 받은 gradient를 그대로 통과시킵니다.
        """
        # TODO: train()에서 만든 gradient를 그대로 반환하세요.
        raise NotImplementedError("Softmax.backward를 구현하세요.")
