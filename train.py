import torch

from dataloader import HandWashDataloader
from model import HandWashModel




def train(train_paths, val_paths, batch_size):
    print("start -->")
    
    # dataloader
    train_loader, val_loader = HandWashDataloader(train_paths, val_paths, batch_size)

    # model
    model = HandWashModel()

    if use_gpu:
        model = model.cuda()

    # optimizer
    optimizer = optim.SGD([
                {'params': model.module.share.parameters()},
                {'params': model.module.fc.parameters(), 'lr': learning_rate},
            ], lr=learning_rate / 10, momentum=momentum, dampening=dampening,
                weight_decay=weight_decay, nesterov=use_nesterov)
    

    # train
    for epoch in range(epochs):
        #
        
        for data in train_loader:
            img, label = data
            
            out = model.forward(img)

            # 



