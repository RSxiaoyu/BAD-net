import torch
import torchvision
from tqdm import tqdm
from model import BADNet
from pprint import pprint
from dataset import VOCDetection
from torchvision.transforms import v2
from torch.utils.data import DataLoader,ConcatDataset,Subset
from torchvision.datasets import wrap_dataset_for_transforms_v2


def voc_loader(root='./dataset',year='2007',image_set='test',download=True):
    transforms=v2.Compose([
        v2.ToImage(),
        # v2.RandomPhotometricDistort(p=1),
        # v2.RandomZoomOut(fill={tv_tensors.Image: (123, 117, 104), "others": 0}),
        # v2.RandomIoUCrop(),
        # v2.RandomHorizontalFlip(p=1),
        # v2.SanitizeBoundingBoxes(),
        v2.Resize((224,224)),
        v2.ToDtype(torch.float32, scale=True),
    ])

    voc_dataset=VOCDetection(
        root=root,
        year=year,
        image_set=image_set,
        download=download,
        transforms=transforms,
    )
    voc_dataset=wrap_dataset_for_transforms_v2(voc_dataset)

    pprint(len(voc_dataset))

    voc_loader = DataLoader(
        Subset(voc_dataset,indices=range(100)),
        batch_size=10,
        shuffle=True,
        collate_fn=lambda batch: tuple(zip(*batch)),
    )
    return voc_loader
voc_tv=voc_loader(image_set=['person_trainval','bicycle_trainval','motorbike_trainval','bus_trainval','car_trainval'])
voc_test=voc_loader(image_set=['person_test','bicycle_test','motorbike_test','bus_test','car_test'])

def train_loop(model, data_loader, optimizer, epoch, alpha):
    model.train()
    loss = 0
    for images, targets in tqdm(data_loader):
        images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        det_loss = model(images, targets)
        hr_loss = model.backbone.hr_loss
        loss = sum(det_loss.values()) + alpha * hr_loss
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print(f"\nEpoch {epoch}, Total Loss: {loss.item():.4f}")


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = 6
epochs = 10
batch_size = 16
lr = 1e-4
weight_decay = 1e-4
alpha = 0.001


model = BADNet(num_classes=num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay) 
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)


for epoch in tqdm(range(epochs)):
    train_loop(model, voc_test, optimizer, epoch, alpha=alpha)
    scheduler.step()  # 更新学习率
    
    # 间隔迭代训练策略（论文中的Interval Iterative Strategy，简化实现）
    # if epoch % 2 == 1:  # 奇数epoch生成精炼数据集（示例逻辑，需根据论文Algorithm 1完整实现）
    #     # 此处需添加生成精炼数据集的逻辑（如对去雾图像添加不同程度的雾）
    #     # 简化示例：仅使用原始数据集，完整实现需参考论文数据增强算法
    #     pass
