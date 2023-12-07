import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
import random
import numpy as np
from torch import autocast
from torch.cuda.amp import GradScaler
import argparse
import wandb
from models import vgg, layers


def manual_seed(seed):
    np.random.seed(seed) #1
    random.seed(seed) #2
    torch.manual_seed(seed) #3
    torch.cuda.manual_seed(seed) #4.1
    torch.cuda.manual_seed_all(seed) #4.2
    torch.backends.cudnn.benchmark = False #5 
    torch.backends.cudnn.deterministic = True #6


def validate(model, test_loader):
    model.eval()
    val_loss = 0.0
    val_acc = 0
    with torch.no_grad():
        for data in tqdm(test_loader,leave=True):
            imgs = torch.cat([data[0], torch.zeros((data[0].size(0),1,224,224))], 1)
            imgs, target = imgs.to(device), data[1].to(device)
            output = model(imgs)
            loss = criterion(output, target)

            val_loss += loss.item()
            _, preds = torch.max(output.data, 1)
            val_acc += (preds==target).sum().item()
    
    val_loss = val_loss/len(test_loader)
    val_acc = 100. * val_acc/len(test_loader.dataset)

    return val_loss, val_acc


def get_args():
    parser = argparse.ArgumentParser(description="training")
    # model or layer
    # pytorch(int,float) conv1d
    parser.add_argument("--lr", type=float,default=1e-3, help="select learning rate")
    parser.add_argument("--optim", choices=['adam','sgd'],default='sgd', help="select optimizer")
    parser.add_argument("--epoch", type=int, default= 50, help='select number of epoch')
    parser.add_argument("--custom", action='store_true')
    parser.add_argument("--bias", action='store_true')
    parser.add_argument('--batch', type=int, default=32)
    return parser.parse_args()


if __name__=="__main__":
    args = get_args()

    name = f"vgg16_{args.optim}_{args.lr}_{args.batch}_{args.bias}"

    wandb.init(
            project = "cifar100_vgg16",
            name = name,
            config = args
        )
    manual_seed(42)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(device)

    if args.custom or args.bias==False:
        model = vgg.vgg16(bias=args.bias)
        model.weight_init()
    else:
        model = torchvision.models.vgg.vgg16(pretrained=True)
        model.features[0] = nn.Conv2d(4,64,3,1,1)
        model.classifier[6] = nn.Linear(in_features=4096, out_features=100, bias=True)
    # freeze convolution weights
    # for i,param in enumerate(model.features.parameters()):
    #     if i !=0:
    #         param.requires_grad = False
        
    # print(model)

    EPOCHS = args.epoch
    BATCH = args.batch
    VAL_BATCH=256
    LR = args.lr
    MOMENTUM = 0.9
    DECAY=5e-4

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomCrop((224,224),padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        # transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std= (0.2675, 0.2565, 0.2761)),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std= (0.2675, 0.2565, 0.2761)),
    ])

    train_data = torchvision.datasets.CIFAR100(root="./dataset", train=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH,
                                            shuffle=True,pin_memory=True,num_workers=4)
    val_data = torchvision.datasets.CIFAR100(root="./dataset", train=False, transform=test_transform)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=VAL_BATCH,
                                            shuffle=False,pin_memory=True,num_workers=4)

    # optimizerdmf aksemfusmsep
    if args.optim =="adam":
        optimizer = optim.Adam(model.classifier.parameters(), lr=LR)
    else:
        optimizer = optim.SGD(model.classifier.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=DECAY)

    scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr = LR, max_lr=LR*10,step_size_up=20, cycle_momentum=False)
    # loss function
    criterion = nn.CrossEntropyLoss()

    model.to(device)
    zero_tensor = torch.zeros((BATCH,1,224,224))


    start_loss, start_acc = validate(model, val_loader)
    # print(f"START LOSS {start_loss:.4f}, ACC {start_acc:.2f}")
    best = 0
    scaler = GradScaler()
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        train_acc = 0
        for data in tqdm(train_loader,leave=True):
            imgs = torch.cat([data[0], torch.zeros((data[0].size(0),1,224,224))], 1)
            imgs, target = imgs.to(device), data[1].to(device)
            optimizer.zero_grad()

            with autocast(device_type='cuda', dtype=torch.float16):
                output = model(imgs)
                loss = criterion(output, target)

            scaler.scale(loss).backward()

            scaler.step(optimizer)

            # Updates the scale for next iteration.
            scaler.update()

            _, preds = torch.max(output.data, 1)

            train_loss += loss.item()
            train_acc += (preds ==target).sum().item()

        train_loss = train_loss/len(train_loader)
        train_acc = 100. * train_acc/len(train_loader.dataset)
        # print(f" TRAIN_LOSS : {train_loss:.4f}, TRAIN_ACC : {train_acc:.2f}")

        val_loss, val_acc = validate(model, val_loader)
        # print(f" VAL_LOSS : {val_loss:.4f}, VAL_ACC : {val_acc:.2f}")
        if best < val_acc:
            best = val_acc
            checkpoint = {
                'model' : model,
                'model_state_dict' : model.state_dict(),
                'best_acc': val_acc
            }
            torch.save(checkpoint, f'./checkpoint/{name}.pth')
            # print(f"save best acc {best:.2f}")

        metric_info = {
                'lr/lr' : optimizer.param_groups[0]['lr'],
                'train/loss' : train_loss,
                'train/acc' : train_acc,
                'val/loss' : val_loss,
                'val/acc' : val_acc,
            }
        wandb.log(metric_info, step=epoch)
