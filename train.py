import torch

from dataloader import HandWashDataloader
from model import HandWashModel
from utils import accuracy



def train(datasete_paths, batch_size, epochs, use_gpu):
    print("start -->")
    
    # dataloader
    train_loader, val_loader = HandWashDataloader(datasete_paths, batch_size)

    # model
    
    model = HandWashModel()

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    if use_gpu:
        model = model.to(device)

    # optimizer
    learning_rate = 0.0005
    momentum = 0.9
    dampening = 0.0
    weight_decay = 0.0
    use_nesterov = False

    optimizer = torch.optim.SGD([
                {'params': model.share.parameters()},
                {'params': model.fc.parameters(), 'lr': learning_rate},
            ], lr=learning_rate / 10, momentum=momentum, dampening=dampening,
                weight_decay=weight_decay, nesterov=use_nesterov)
    
    criterion = torch.nn.CrossEntropyLoss().to(device)
    

    # train
    model.train()

    for epoch in range(epochs):
        #
        
        for imgs, target in train_loader:
            # print(imgs, target)
            if use_gpu:
                imgs = imgs.to(device)
                target = target.to(device)
            output = model.forward(imgs)

            loss = criterion(output, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(loss)

            prec1 = accuracy(output.data, target)
            print("top1 : ", prec1)

