from torch import nn
import torch.nn.functional as F
from torch import Tensor
import torch
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size, stride, padding):
        super(BasicConv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,stride=stride,padding=padding,bias = False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1)

        )


    def forward(self, x : Tensor) -> Tensor:

        return self.conv(x)


class Residaul_Block(nn.Module):
    def __init__(self, in_channels):
        super(Residaul_Block, self).__init__()

        self.resdial = nn.Sequential(
            BasicConv(in_channels, in_channels//2, kernel_size=1, stride=1, padding=0),
            BasicConv(in_channels//2, in_channels, kernel_size=3, stride=1, padding=1)

        )

        self.shortcut = nn.Sequential()

    def forward(self, x):
        x_shortcut = self.shortcut(x)
        x_resdial = self.resdial(x)

        return x_shortcut + x_resdial


# FPN의 Top_down layer 입니다.
# lateral connection과 Upsampling이 concatate 한 뒤에 수행합니다.
# https://viso.ai/deep-learning/yolov3-overview/ 여기 아키텍쳐 참고
# https://csm-kr.tistory.com/11
class Top_down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(
            BasicConv(in_channels, out_channels, 1, stride=1, padding=0),
            BasicConv(out_channels, out_channels*2, 3, stride=1, padding=1),
            BasicConv(out_channels*2, out_channels, 1, stride=1, padding=0),
            BasicConv(out_channels, out_channels*2, 3, stride=1, padding=1),
            BasicConv(out_channels*2, out_channels, 1, stride=1, padding=0)
        )

    def forward(self, x):
        return self.conv(x)

# YOLO Layer를 정의합니다.
# YOLO Layer는 13x13, 26x26, 52x52 피쳐맵에서 예측을 수행합니다.
class YOLOLayer(nn.Module):
    def __init__(self, channels, anchors, num_classes=20, img_dim=416):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors # three anchors per YOLO Layer
        self.num_anchors = len(anchors) # 3
        self.num_classes = num_classes # VOC classes 20
        self.img_dim = img_dim # 입력 이미지 크기 416
        self.grid_size = 0

        # 예측을 수행하기 전, smooth conv layer 입니다.
        self.conv = nn.Sequential(
            BasicConv(channels, channels*2, 3, stride=1, padding=1),
            nn.Conv2d(channels*2, (self.num_anchors*(num_classes+5)), 1, stride=1, padding=0)
        )

    def forward(self, x):
        x = self.conv(x)

        # prediction
        # x: batch, channels, W, H
        batch_size = x.size(0)
        grid_size = x.size(2) # S = 13 or 26 or 52
        self.device = x.device

        prediction = x.view(batch_size, self.num_anchors, self.num_classes+5, grid_size, grid_size) # shape (batch, anchor 갯수, 클래스 + bbox, class, h w 사이즈
        # shape change (batch, 3(self.num_anchors), 25(self.num_classes+5), S, S) -> (batch, 3, S, S, 25)
        prediction = prediction.permute(0, 1, 3, 4, 2)
        prediction = prediction.contiguous() # 메모리에 데이터를 순서대로 저장

        obj_score = torch.sigmoid(prediction[..., 4])  # objectness Confidence: 1 if object, else 0
        pred_cls = torch.sigmoid(prediction[..., 5:])  # class score

        # grid_size 갱신
        if grid_size != self.grid_size:
            # grid_size를 갱신하고, transform_outputs 함수를 위해 anchor 박스를 전처리 합니다.
            self.compute_grid_offsets(grid_size, cuda=x.is_cuda)

        # calculate bounding box coordinates
        pred_boxes = self.transform_outputs(prediction)

        # output shape(batch, num_anchors x S x S, 25)
        # ex) at 13x13 -> [batch, 507, 25], at 26x26 -> [batch, 2028, 25], at 52x52 -> [batch, 10647, 25]
        # 최종적으로 YOLO는 10647개의 바운딩박스를 예측합니다.
        output = torch.cat((pred_boxes.view(batch_size, -1, 4),
                            obj_score.view(batch_size, -1, 1),
                            pred_cls.view(batch_size, -1, self.num_classes)), -1)
        return output

    def compute_grid_offsets(self, grid_size, cuda=True):
        self.grid_size = grid_size # ex) 13, 26, 52
        self.stride = self.img_dim / self.grid_size # ex) 32, 16, 8

        # cell index 생성
        # 1, 1, S, 1
        self.grid_x = torch.arange(grid_size, device=self.device).repeat(1, 1, grid_size, 1).type(torch.float32)
        # 1, 1, 1, S
        self.grid_y = torch.arange(grid_size, device=self.device).repeat(1, 1, grid_size, 1).transpose(3, 2).type(torch.float32)

        # anchors를 feature map 크기로 정규화, [0~1] 범위
        # ex) (10, 13), (16, 30), (33, 23) / stride
        scaled_anchors = [(a_w / self.stride, a_h / self.stride) for a_w, a_h in self.anchors]
        # tensor로 변환
        self.scaled_anchors = torch.tensor(scaled_anchors, device=self.device)

        # shape=(3,2) -> (1,1,3,1)
        self.anchor_w = self.scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1))
        # shape=(3,2) -> (1,1,3,1)
        self.anchor_h = self.scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))

    # 예측한 바운딩 박스 좌표를 계산하는 함수입니다.
    def transform_outputs(self, prediction):
        # prediction = (batch, num_anchors, S, S, coordinates + classes)
        device = prediction.device
        x = torch.sigmoid(prediction[..., 0]) # sigmoid(box x), 예측값을 sigmoid로 감싸서 [0~1] 범위
        y = torch.sigmoid(prediction[..., 1]) # sigmoid(box y), 예측값을 sigmoid로 감싸서 [0~1] 범위
        w = prediction[..., 2] # 예측한 바운딩 박스 너비
        h = prediction[..., 3] # 예측한 바운딩 박스 높이

        pred_boxes = torch.zeros_like(prediction[..., :4]).to(device)
        pred_boxes[..., 0] = x.data + self.grid_x # sigmoid(box x) + cell x 좌표
        pred_boxes[..., 1] = y.data + self.grid_y # sigmoid(box y) + cell y 좌표
        pred_boxes[..., 2] = torch.exp(w.data) * self.anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * self.anchor_h

        return pred_boxes * self.stride


class DarkNet(nn.Module):
    def __init__(self, anchors, num_blocks=[1,2,8,8,4], num_classes=20):
        super().__init__()

        # feature extractor
        self.conv1 = BasicConv(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.res_block1 = self._make_residual_block(64, num_blocks[0])
        self.res_block2 = self._make_residual_block(128, num_blocks[1])
        self.res_block3 = self._make_residual_block(256, num_blocks[2])
        self.res_block4 = self._make_residual_block(512, num_blocks[3])
        self.res_block5 = self._make_residual_block(1024, num_blocks[4])

        self.upsample = nn.Upsample(scale_factor=2)

        # FPN Top down, conv + upsampling을 수행합니다.
        self.topdown_1 = Top_down(1024, 512)
        self.topdown_2 = Top_down(768, 256)
        self.topdown_3 = Top_down(384, 128)

        # FPN lateral connection
        # 차원 축소를 위해 사용합니다.
        self.lateral_1 = BasicConv(512, 256, 1, stride=1, padding=0)
        self.lateral_2 = BasicConv(256, 128, 1, stride=1, padding=0)


        # prediction, 13x13, 26x26, 52x52 피쳐맵에서 예측을 수행합니다.
        self.yolo_1 = YOLOLayer(512, anchors=anchors[2]) # 13x13
        self.yolo_2 = YOLOLayer(256, anchors=anchors[1]) # 26x26
        self.yolo_3 = YOLOLayer(128, anchors=anchors[0]) # 52x52

    def _make_residual_block(self,in_channels, num_block):
        blocks = []

        # down sample
        blocks.append(BasicConv(in_channels=in_channels//2, out_channels=in_channels,kernel_size=3, stride=2, padding=1))

        for i in range(num_block):
            blocks.append(Residaul_Block(in_channels=in_channels))

        return nn.Sequential(*blocks)

    def forward(self, x):
        # feature extractor
        x = self.conv1(x)
        c1 = self.res_block1(x)
        c2 = self.res_block2(c1)
        c3 = self.res_block3(c2)
        c4 = self.res_block4(c3)
        c5 = self.res_block5(c4)
        ## darknet backbone

        p5 = self.topdown_1(c5) # 1024 -> 512
        p4 = self.topdown_2(torch.cat((self.upsample(p5), self.lateral_1(c4)), 1)) # 768 -> 256
        p3 = self.topdown_3(torch.cat((self.upsample(p4), self.lateral_2(c3)), 1)) # 384 -> 128

        # prediction
        yolo_1 = self.yolo_1(p5)
        yolo_2 = self.yolo_2(p4)
        yolo_3 = self.yolo_3(p3)


        return torch.cat((yolo_1, yolo_2, yolo_3), 1), [yolo_1, yolo_2, yolo_3]
