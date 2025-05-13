import torch
from pprint import pprint
from torchvision.transforms import v2
from torch.utils.data import DataLoader,ConcatDataset,Subset
from torchvision.datasets import VOCDetection,wrap_dataset_for_transforms_v2


classes = [
    '','aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
] # index = class + 1
target_classes=['','person','bicycle','motorbike','bus','car']


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
        voc_trainval=Subset(voc_trainval,subset)
    print('VOC trainval ready.')
    return DataLoader(
        voc_trainval,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: (torch.stack([x[0] for x in batch]),[x[1] for x in batch]),
    )
def test_loader(batch_size=8,subset=None):
    voc_test=voc_dataset()
    if subset:
        voc_test=Subset(voc_test,subset)
    print('VOC test ready.')
    return DataLoader(
        voc_test,
        batch_size=batch_size,
        collate_fn=lambda batch: (torch.stack([x[0] for x in batch]),[x[1] for x in batch]),
    )