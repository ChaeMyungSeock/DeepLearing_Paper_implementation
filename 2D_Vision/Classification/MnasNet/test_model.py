
from MnasNet_model.MnasNet import SE_MB_bottleneck, MnasNet #, #SE_BasicBlock
from torchsummary import summary



#
def MnasNet_A1():
    model = MnasNet(SE_MB_bottleneck, n_classes=10)
    print(model)

    # 모델을 CUDA로 전달합니다.
    model.to("cuda")
    print(next(model.parameters()).device)

    # 모델 summary를 확인합니다.

    return summary(model, input_size=(3, 224, 224))


MnasNet_A1()
# resnetX101()