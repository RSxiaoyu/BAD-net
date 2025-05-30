from matplotlib import pyplot as plt
import torch
from tqdm import tqdm
from dataset import test_loader, train_loader,refined_loader
from model import BADNet


def train_loop(model, data_loader, optimizer, epoch, alpha):
    if epoch%2==0:
        loader=data_loader
    else:
        loader=refined_loader(data_loader,model.backbone.dehazing_module,device,)

    model.train()
    loss=0
    for images, targets in tqdm(loader):
        images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        optimizer.zero_grad()
        
        det_loss = model(images, targets)
        hr_loss = model.backbone.hr_loss
        tmp_loss = sum(det_loss.values()) + alpha * hr_loss
        loss+=tmp_loss

        tmp_loss.backward()
        optimizer.step()
    train_loss.append(loss.item())
    print(f"\nEpoch {epoch}, Train Loss: {loss.item():.4f}")

def test_loop(model, data_loader, alpha):
    # model.eval()
    loss=0
    with torch.no_grad():
        for images, targets in tqdm(data_loader):
            images = images.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            det_loss = model(images, targets)
            hr_loss = model.backbone.hr_loss
            tmp_loss = sum(det_loss.values()) + alpha * hr_loss
            loss+=tmp_loss
    test_loss.append(loss.item())
    print(f"\nEpoch {epoch}, Test Loss: {loss.item():.4f}")

def draw_loss(train_loss,test_loss):
    plt.cla()
    plt.plot(train_loss, label='Train Loss', color='blue')
    plt.plot(test_loss, label='Test Loss', color='red')  # 虚线区分
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.pause(0.1)
    
if __name__=='__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = 6
    epochs = 20
    batch_size = 8
    lr = 1e-4
    weight_decay = 1e-4
    alpha = 0.001
    train_loss=[]
    test_loss=[]

    model = BADNet(num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay) 
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    # scaler = torch.amp.GradScaler()

    trainloader=train_loader(batch_size)
    testloader=test_loader(batch_size)
    print(f'Model running on {device}')
    plt.ion()
    for epoch in tqdm(range(epochs)):
        train_loop(model, trainloader, optimizer, epoch, alpha=alpha)
        test_loop(model,testloader,alpha=alpha)
        scheduler.step()  # 更新学习率
        draw_loss(train_loss,test_loss)
    plt.ioff()
    plt.show()