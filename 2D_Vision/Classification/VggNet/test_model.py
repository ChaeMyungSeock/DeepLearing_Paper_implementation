from VggNet_model.VggNet import *
export = True
def export_classificaton_model(size, model):
    w = size[0]
    h = size[1]

    x = torch.rand(1,3,h,w)

    if torch.cuda.is_available():
        x = x.cuda()
    filename = 'TensorRT_python/test_vgg16_.onnx'
    print('dumping network to %s' % filename)
    torch.onnx.export(model, x, filename)

model = VGGNet(n_classes=10)
print(model)

# 모델을 CUDA로 전달합니다.
model.to("cuda")
print(next(model.parameters()).device)

# 모델 summary를 확인합니다.
from torchsummary import summary
summary(model, input_size=(3, 224, 224))


if export:
    export_classificaton_model([224,224], model)

