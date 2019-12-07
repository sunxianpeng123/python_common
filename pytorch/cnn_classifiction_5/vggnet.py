# encoding: utf-8

"""
@author: sunxianpeng
@file: vggnet.py
@time: 2019/11/18 13:59
"""

import torch
import torch.nn as nn

class VggNet16(nn.Module):
    def __init__(self,num_classes):
        super(VggNet16, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,64,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(64,128,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.classifier = nn.Sequential(
            nn.Linear(512*7*7,4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),

            nn.Linear(4096,4096),
            nn.ReLU(True),
            nn.Dropout(),

            nn.Linear(4096,num_classes)
        )
    def forward(self,x):
        x = self.features(x)
        x = x.view(x.shape[0],-1)
        print(x)
        x = self.classifier(x)
        print(x)
        return x


if __name__ == '__main__':
    # 代入数据检验
    size = 512
    y = torch.randn(1, 3, size, size)
    model = VggNet16(num_classes=10)
    model(y)