# 2020GraduateProjectStudy
- Python의 Tensorflow 및 Keras 정리
- Machine Learning 정리
- Computer Vision 기말 대비!!

## Regularization (4-1)
- Overfitting을 피하기 위해 사용하는 방법<br>
- Overfitting
  모델이 학습 데이터에 특화되어 테스트 성능이 낮아지는 것, 일반화(generalization) 성능이 감소<br>
  학습 데이터에 비해 학습모델이 너무 복잡해서 발생<br>
- 모델을 제한해 단순하게 만드는 것, 가중치가 작은 값을 가지게 하는 것(가중치의 분포가 균일해짐)<br>
- J(θ)= J_D(θ)+ αJ_pen(θ), α: Regularization Coefficient(클수록 그래프 단순해짐)<br>

1. L1(Lasso) Regularization: 가중치의 절대값에 비례하는 비용 추가

2. L2(Ridge) Regularization:: 가중치의 제곱에 비례하는 비용 추가

3. Lp Regularization(Elastic Net): L1 + L2

```bash
tf.keras.layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l1(0.01))
tf.keras.layers.Dense(64, bias_regularizer=tf.keras.regularizers.l2(0.01))
```

## Training
```bash
model.compile(optimizer='Adam', loss='sparce_categorical_crossentropy', metrics = ['accuracy'])
```
- Categorical Cross Entropy: multi class classification에 사용
- accuracy: 예측이 label과 얼마나 자주 일치하는지 계산

