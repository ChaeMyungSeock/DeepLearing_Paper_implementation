#from ResNet_model.ResNet import *
# from ResNet_model.ResNet50 import *

from ResNet_model.ResNet import BasicBlock, Res_bottleneck, ResNet
from torchsummary import summary



def resnet18():
    model = ResNet(BasicBlock,  [2, 2, 2, 2], n_classes=10)
    print(model)

    # 모델을 CUDA로 전달합니다.
    model.to("cuda")
    print(next(model.parameters()).device)

    # 모델 summary를 확인합니다.

    return summary(model, input_size=(3, 224, 224))


def resnet34():
    model = ResNet(BasicBlock, [3, 4, 6, 3], n_classes=10)
    print(model)

    # 모델을 CUDA로 전달합니다.
    model.to("cuda")
    print(next(model.parameters()).device)

    # 모델 summary를 확인합니다.

    return summary(model, input_size=(3, 224, 224))


#
def resnet50():
    model = ResNet(Res_bottleneck, [3, 4, 6, 3], n_classes=10)
    print(model)

    # 모델을 CUDA로 전달합니다.
    model.to("cuda")
    print(next(model.parameters()).device)

    # 모델 summary를 확인합니다.

    return summary(model, input_size=(3, 224, 224))


#
def resnet101():
    model = ResNet(Res_bottleneck, [3, 4, 23, 3], n_classes=10)
    print(model)

    # 모델을 CUDA로 전달합니다.
    model.to("cuda")
    print(next(model.parameters()).device)

    # 모델 summary를 확인합니다.

    return summary(model, input_size=(3, 224, 224))

#
def resnet152():
    model = ResNet(Res_bottleneck, [3, 8, 36, 3], n_classes=10)
    print(model)

    # 모델을 CUDA로 전달합니다.
    model.to("cuda")
    print(next(model.parameters()).device)

    # 모델 summary를 확인합니다.

    return summary(model, input_size=(3, 224, 224))


resnet50()
# resnet152()