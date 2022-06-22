from VggNet_model.VggNet import *

model = VGGNet(n_classes=10)
print(model)

# 모델을 CUDA로 전달합니다.
model.to("cuda")
print(next(model.parameters()).device)

# 모델 summary를 확인합니다.
from torchsummary import summary
summary(model, input_size=(3, 224, 224))