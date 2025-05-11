import torch
from torch import nn
import torchvision
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models import mobilenet_v3_large
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
def conv_bn_relu(in_channels,out_channels,kernel_size=3,padding='same',dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_channels,out_channels,kernel_size,padding='same',dilation=dilation),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),  
    )

class DehazingModule(nn.Module): # Fig 3 原图->去雾图
    def __init__(self):
        super().__init__()
        self.init_conv = nn.Sequential(
            conv_bn_relu(3,3,3),
            conv_bn_relu(3,3,3),
        )
        self.branch1 = nn.Sequential(
            conv_bn_relu(3,3,3),
            conv_bn_relu(3,3,3),
        )
        self.branch2 = nn.Sequential(
            conv_bn_relu(3,3,3),
            conv_bn_relu(3,3,3,dilation=2),
        )
        self.branch3 = nn.Sequential(
            conv_bn_relu(3,3,3,dilation=2),
            conv_bn_relu(3,3,3,dilation=2),
        )
        self.final_conv = nn.Sequential(
            conv_bn_relu(9,3,3),
            conv_bn_relu(3,3,1),
        )
        self.b=1
   
    def forward(self, x):
        x = self.init_conv(x)
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        concat = torch.cat([b1, b2, b3],1)
        k = self.final_conv(concat)
        return torch.relu(k * x - k + self.b)

class AttentionFusionModule(nn.Module): #Fig 5 F_haze+F_dehaze->F_fuse
    def __init__(self,in_channels,reduction=16):
        super().__init__()
        self.gap_w=nn.AdaptiveAvgPool2d((None,1))
        self.gap_h=nn.AdaptiveAvgPool2d((1,None))
        self.reduce_conv=conv_bn_relu(in_channels,in_channels//reduction,1)
        self.expand_conv=conv_bn_relu(in_channels//reduction,in_channels,1)


    def forward(self,f_haze,f_dehaze):
        _,_,H,W=f_haze.shape
        x1=f_haze+f_dehaze                              # B,C,H,W
        x2_1=self.gap_w(x1).permute(0,1,3,2)            # B,C,1,H
        x2_2=self.gap_h(x1)                             # B,C,1,W
        x2=torch.concat((x2_1,x2_2),3)                  # B,C,1,H+W
        f_tmp=self.reduce_conv(x2)                      # B,C/r,1,H+W
        x_h,x_w=f_tmp.split((H,W),3)                    # B,C/r,1,H  B,C/r,1,W
        att_h = self.expand_conv(x_h).permute(0,1,3,2)  # B,C,H,1
        att_w = self.expand_conv(x_w)                   # B,C,1,W
        f_att = torch.sigmoid(att_h @ att_w)            # B,C,H,W
        return f_haze * f_att + f_dehaze * (1-f_att)
    
class BADBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.mobilenet = mobilenet_v3_large(weights='DEFAULT').features
        self.dehazing_module = DehazingModule()
        self.attention_fusion = AttentionFusionModule(in_channels=960)
        self.out_channels = 960  # FasterRCNN要求
        self.hr_loss=None

    def forward(self, x):
        f_haze = self.mobilenet(x)
        # return f_haze
        x_dehaze = self.dehazing_module(x)
        f_dehaze = self.mobilenet(x_dehaze)
        self.hr_loss=self.hrloss(f_haze,f_dehaze)
        return self.attention_fusion(f_haze, f_dehaze)
    def hrloss(self,f_haze,f_dehaze):
        f_haze = torch.relu(torch.mean(f_haze, dim=(2, 3)))
        f_dehaze = torch.relu(torch.mean(f_dehaze, dim=(2, 3)))
        # eps = 1e-8
        # f_haze = torch.clamp(f_haze, min=eps)  # 限制最小值为eps
        # f_dehaze = torch.clamp(f_dehaze, min=eps)  # 限制最小值为eps
        # kl_loss = torch.mean(f_haze * (torch.log(f_haze) - torch.log(f_dehaze)))
        return torch.nn.functional.kl_div(
            f_dehaze.log(), f_haze, reduction='batchmean'
        )
class BADNet(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.backbone = BADBackbone()
        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),),
            aspect_ratios=((0.5, 1.0, 2.0),)
        )
        box_roi_pool = MultiScaleRoIAlign(
            featmap_names=['0'],
            output_size=7,
            sampling_ratio=2
        )
        self.detector = FasterRCNN(
            backbone=self.backbone,
            num_classes=num_classes,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=box_roi_pool,
            image_mean=[0.485, 0.456, 0.406],
            image_std=[0.229, 0.224, 0.225]
        )

    def forward(self, images, targets=None):
        return self.detector(images, targets)


if __name__ == '__main__':
    # 测试输入
    images = torch.randn(2, 3, 224, 224)
    targets = [
        {'boxes': torch.tensor([[10, 10, 50, 50]]), 'labels': torch.tensor([1])},
        {'boxes': torch.tensor([[20, 20, 60, 60]]), 'labels': torch.tensor([2])} 
    ]
    model = BADNet(num_classes=6)
    model.train()
    losses = model(images, targets)
    print("训练损失:", losses)

    model.eval()
    detections = model(images)
    print("推理结果:", detections)