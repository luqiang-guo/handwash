import torch

from dataloader import HandWashDataloader
from model import HandWashModel




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
    learning_rate = 1e-3
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
        
        for imgs, label in train_loader:
            # print(imgs, label)
            if use_gpu:
                imgs = imgs.to(device)
                label = label.to(device)
            out = model.forward(imgs)

            loss = criterion(out, label)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(loss)

