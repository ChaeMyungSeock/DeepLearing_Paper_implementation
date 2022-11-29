import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import utils
from torchsummary import summary
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import to_pil_image
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import cv2
import os
import copy
import numpy as np
import pandas as pd
import random
# import albumentations as A
# from albumentations.pytorch import ToTensor
from model.models import DarkNet
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# test
anchors = [[(10,13),(16,30),(33,23)],[(30,61),(62,45),(59,119)],[(116,90),(156,198),(373,326)]]
x = torch.randn(1, 3, 416, 416)
with torch.no_grad():
    model = DarkNet(anchors)
    output_cat , output = model(x)
    print(output_cat.size())
    print(output[0].size(), output[1].size(), output[2].size())