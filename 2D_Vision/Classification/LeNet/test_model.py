from LeNet_model.LeNet import *

model = LeNet_5(n_classes=10)
print(model)

# 모델을 CUDA로 전달합니다.
model.to("cuda")
print(next(model.parameters()).device)

# 모델 summary를 확인합니다.
from torchsummary import summary
summary(model, input_size=(1, 32, 32))