# from ResNeXt_model.ResNeXt34 import *
# from ResNeXt_model.ResNeXt50 import *

from ResNeXt_model.ResNeXt import ResNext_bottleneck, ResNext
from torchsummary import summary



#
def resnetX50():
    model = ResNext(ResNext_bottleneck, [3, 4, 6, 3], n_classes=10)
    print(model)

    # 모델을 CUDA로 전달합니다.
    model.to("cuda")
    print(next(model.parameters()).device)

    # 모델 summary를 확인합니다.

    return summary(model, input_size=(3, 224, 224))

#
#
def resnetX101():
    model = ResNext(ResNext_bottleneck, [3, 4, 23, 3], n_classes=10)
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


resnetX50()
# resnetX101()