import torch

from dataloader import HandWashDataloader
from model import HandWashModel
from utils import accuracy
import copy


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
        # train
        for input, target in train_loader:
            #
            if use_gpu:
                input = input.to(device)
                target = target.to(device)
            output = model.forward(input)

            loss = criterion(output, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            prec1 = accuracy(output.data, target)
            # print("Epoch:{} train :top1 ")
            print("train :top1 ", prec1, "loss : ", loss)

        # validation
        for i, (input, target) in enumerate(val_loader):
            if use_gpu:
                input = input.to(device)
                target = target.to(device)
            
            with torch.no_grad():
                output = model.forward(input)
                prec1 = accuracy(output.data, target)
                print("validation : top1 ", prec1)
        
        # save models

        model_wts = copy.deepcopy(model.state_dict())

        model_name = "./models/" + "handwash_" + str(epoch) + ".pth"
        torch.save(model_wts, model_name)


