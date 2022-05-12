import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


def get_images_and_labels(root_path, label_files):
 
    root_path = root_path+"/"
    filenames = []
    labels = []
    with open(root_path+label_files) as f:
        dir_labels = [line.strip().split(' ') for line in f.readlines()] 

    for filename, label in dir_labels:
        filenames.append(root_path+filename)
        labels.append(label)
        # print(root_path+filename, label)
 
    return filenames, labels

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

class HandWashDataset(Dataset):
    def __init__(self, dir_path, label_file, transform=None, loader=pil_loader):
        self.dir_path = dir_path
        self.label_file = label_file
        self.transform = transform
        self.loader = loader
        self.images, self.labels = get_images_and_labels(self.dir_path, self.label_file )
 
    def __len__(self):
        return len(self.images)
 
    def __getitem__(self, index):
        img_path = self.images[index]
        label = self.labels[index]

        imgs = self.loader(img_path)
        if self.transform is not None:
            imgs = self.transform(imgs)
        return imgs, torch.tensor(int(label))

def HandWashDataloader(root_paths, batch_size):

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

    # 确定imagenet test transforms
    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])
    
    train_dataset = HandWashDataset(root_paths, "train_list.txt", train_transforms)
    val_dataset =   HandWashDataset(root_paths, "val_list.txt", test_transforms)

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