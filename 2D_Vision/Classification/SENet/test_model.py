
from SENet_model.SENet import SE_ResNext_bottleneck, SE_ResNext #, #SE_BasicBlock
from torchsummary import summary



#
def se_resnetX50():
    model = SE_ResNext(SE_ResNext_bottleneck, [3, 4, 6, 3], n_classes=10)
    print(model)

    # 모델을 CUDA로 전달합니다.
    model.to("cuda")
    print(next(model.parameters()).device)

    # 모델 summary를 확인합니다.

    return summary(model, input_size=(3, 224, 224))

#
#
def se_resnetX101():
    model = SE_ResNext(SE_ResNext_bottleneck, [3, 4, 23, 3], n_classes=10)
    print(model)

    # 모델을 CUDA로 전달합니다.
    model.to("cuda")
    print(next(model.parameters()).device)

    # 모델 summary를 확인합니다.

    return summary(model, input_size=(3, 224, 224))

#
# def resnet152():
#     model = ResNet50(Res_bottleneck, [3, 8, 36, 3], n_classes=10)
#     print(model)
#
#     # 모델을 CUDA로 전달합니다.
#     model.to("cuda")
#     print(next(model.parameters()).device)
#
#     # 모델 summary를 확인합니다.
#
#     return summary(model, input_size=(3, 224, 224))


se_resnetX50()
# resnetX101()