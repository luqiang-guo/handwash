from dataloader import HandWashDataset, HandWashDataloader




train_dataset = HandWashDataset("/mnt/nvme2n1/handwash_dataset", "train_list.txt")
train_dataset = HandWashDataset("/mnt/nvme2n1/handwash_dataset", "val_list.txt")



train_loader, val_loader = HandWashDataloader("/mnt/nvme2n1/handwash_dataset",
                                               64)

for img, label in train_loader:
    print(img)
    print(label)