import torch
from tqdm import tqdm
from model import BADNet
from dataset import train_loader,test_loader

def train_loop(model, data_loader, optimizer, epoch, alpha):
    model.train()
    loss = 0
    scaler = torch.amp.GradScaler()
    for images, targets in tqdm(data_loader):
        images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        with torch.amp.autocast('cuda'):
            det_loss = model(images, targets)
            hr_loss = model.backbone.hr_loss
            loss = sum(det_loss.values()) + alpha * hr_loss
        
        # 反向传播
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
        
    print(f"\nEpoch {epoch}, Total Loss: {loss.item():.4f}")


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = 6
epochs = 1
batch_size = 16
lr = 1e-4
weight_decay = 1e-4
alpha = 0.001


model = BADNet(num_classes=num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay) 
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

print(device)
for epoch in tqdm(range(epochs)):
    train_loop(model, test_loader(), optimizer, epoch, alpha=alpha)
    scheduler.step()  # 更新学习率
    
    # 间隔迭代训练策略（论文中的Interval Iterative Strategy，简化实现）
    # if epoch % 2 == 1:  # 奇数epoch生成精炼数据集（示例逻辑，需根据论文Algorithm 1完整实现）
    #     # 此处需添加生成精炼数据集的逻辑（如对去雾图像添加不同程度的雾）
    #     # 简化示例：仅使用原始数据集，完整实现需参考论文数据增强算法
    #     pass
