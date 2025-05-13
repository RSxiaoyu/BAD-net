import torch
from pprint import pprint
from torchvision.transforms import v2
from torch.utils.data import DataLoader,ConcatDataset,Subset,Dataset
from torchvision.datasets import VOCDetection,wrap_dataset_for_transforms_v2


classes = [
    '','aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
] # index = class + 1
target_classes=['','person','bicycle','motorbike','bus','car']
collate_fn=lambda batch: (torch.stack([x[0] for x in batch]),[x[1] for x in batch])
def add_haze(dehazed_image, level=0.05):
    C, H, W = dehazed_image.shape
    y, x = torch.meshgrid(torch.arange(H,device=dehazed_image.device), torch.arange(W,device=dehazed_image.device), indexing='ij')
    d = torch.sqrt((x - W/2)**2 + (y - H/2)**2) / max(H, W)
    td = torch.exp(-level * d)
    hazed_image = dehazed_image * td + 0.5 * (1 - td)
    return hazed_image

class FilterCategories(torch.nn.Module):
    def __init__(self, target_classes):
        super().__init__()
        self.target_index = [classes.index(name) for name in target_classes]        # target classes -> index in all classes
        self.index_map = {index: i  for i, index in enumerate(self.target_index)}   # index in all classes ->index in 5+1 classes

    def forward(self, image, target):
        mask = torch.tensor([label in self.target_index for label in target['labels']], dtype=torch.bool)
        target['boxes'] = target['boxes'][mask]
        target['labels'] = torch.tensor([self.index_map[item.item()] for item in target['labels'][mask]],dtype=torch.int64)
        return image,target
    
class RandomHazeTransform(torch.nn.Module):
    def __init__(self, max_level=0.2, add_prob=2/3):
        super().__init__()
        self.max_level = max_level
        self.add_prob = add_prob
    
    def forward(self, image, target):
        if torch.rand(1) < self.add_prob:
            image=add_haze(image)
        return image, target  # 返回加雾后的图像和原标签


def voc_dataset(root='./dataset',year='2007',image_set='test'):
    transforms=v2.Compose([
        FilterCategories(target_classes),
        v2.ToImage(),
        v2.Resize((224,224)),
        v2.ToDtype(torch.float32,scale=True)
    ])
    voc_dataset=VOCDetection(root=root,year=year,image_set=image_set,download=True,transforms=transforms)
    return wrap_dataset_for_transforms_v2(voc_dataset)


def train_loader(batch_size=8,subset=None):
    voc2007_trainval=voc_dataset(image_set='trainval')
    print('VOC 2007 trainval loaded.')
    voc2012_trainval=voc_dataset(year='2012',image_set='trainval')
    print('VOC 2012 trainval loaded.')
    voc_trainval=ConcatDataset([voc2007_trainval,voc2012_trainval])
    if subset:
        voc_trainval=Subset(voc2007_trainval,subset)
    print('VOC trainval ready.')
    return DataLoader(
        voc_trainval,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

def test_loader(batch_size=8,subset=None):
    voc_test=voc_dataset()
    if subset:
        voc_test=Subset(voc_test,subset)
    print('VOC test ready.')
    return DataLoader(
        voc_test,
        batch_size=batch_size,
        collate_fn=collate_fn,
    )

    
class refined_dataset(Dataset):
    def __init__(self, ori_dataset, dehazing_module, device, num_levels):
        self.ori_dataset = ori_dataset
        self.dehazing_model = dehazing_module.eval()
        self.num_levels = num_levels
        self.device=device
    def __len__(self):
        return len(self.ori_dataset) * self.num_levels
    
    def __getitem__(self, index):
        ori_index = index // self.num_levels
        level_idx = index % self.num_levels
        level = 0.05 + 0.01 * level_idx
        image, target = self.ori_dataset[ori_index]
        image = image.to(self.device)
        with torch.no_grad():
            dehazed_image = self.dehazing_model(image.unsqueeze(0))[0]  # 输出(C, H, W)
        hazed_image = add_haze(dehazed_image, level)
        return hazed_image, target

    

def refined_loader(ori_loader,dehazing_module,device,batch_size=8,num_levels=10):
    dataset=refined_dataset(ori_loader.dataset,dehazing_module,device,num_levels)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )