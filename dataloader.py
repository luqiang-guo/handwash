import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


def get_images_and_labels(dir_path)
    print("")
    # 读取label


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

class HandWashDataset(Dataset):
    def __init__(self, dir_path, transform=None, loader=pil_loader):
        self.dir_path = dir_path
        self.transform = transform
        self.images, self.labels = get_images_and_labels(self.dir_path)
 
    def __len__(self):
        # 返回数据集的数据数量
        return len(self.images)
 
    def __getitem__(self, index):
        img_path = self.images[index]
        label = self.labels[index]

        imgs = self.loader(img_names)
        if self.transform is not None:
            imgs = self.transform(imgs)
        return imgs, label

def HandWashDataloader(train_paths, val_paths, batch_size)
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.3456, 0.2281, 0.2233], [0.2528, 0.2135, 0.2104])
    ])

    # 确定imagenet test transforms
    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])
    
    train_dataset = HandWashDataset(train_paths, train_transforms)
    val_dataset =   HandWashDataset(val_paths, test_transforms)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, val_loader