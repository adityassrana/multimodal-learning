#!/usr/bin/env python
# coding: utf-8
import itertools
import torch
from torch import nn
from torch.utils.data import DataLoader, dataloader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms, models
from torchvision.utils import make_grid
from torchsummary import summary
from pathlib import Path
import os, sys, argparse
import matplotlib.pyplot as plt
import logging
from torchsummary import summary
import RGBDutils

def parse_args(args = sys.argv[1:]):
    """
    Utility function for parsing command line arguments
    """
    parser = argparse.ArgumentParser(description='A simple script for training an image classifier')
    parser.add_argument('--exp_name',type=str, default='baseline', help='name of experiment - for saving model and tensorboard dir')
    parser.add_argument("--data_path", default="/home/adityassrana/MCV_UAB/m5-vr/homework/Multimodal-learning-LuisHerranz/sunrgbd_lite", help = "path to Dataset")
    parser.add_argument("--image_size", type=int, default=224, help="image size used for training")
    parser.add_argument("--max_epochs", type=int, default=2, help="number of epochs to train the model for")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="base learning rate to use for training")
    parser.add_argument("--plot_stats", action="store_true", help = "to save matplotlib plots of train-val loss and accuracy")
    args = parser.parse_args(args)
    return args

def get_dataloaders(path: str, img_size: int = 224, batch_size: int = 128, num_workers: int = 2):
    """
    Get train,val and test dataloaders from the dataset
    Args:
        path: path to dataset
        modality: rgb or hha
        img_size: size of image to be used fo training (bilinear interpolation is used)
        batch:size: batch size for dataloaders
        num_workers: workers to be used for loading data
    Returns:
        train and test dataloaders
    """
    # normalization is important when using models pretrained on ImageNet
    # Data augmentation and normalization for training
    RGB_AVG = [0.485, 0.456, 0.406] # Default ImageNet ILSRVC2012
    RGB_STD = [0.229, 0.224, 0.225] # Default ImageNet ILSRVC2012
    DEPTH_AVG = [0.485, 0.456, 0.406] # Default ImageNet ILSRVC2012
    DEPTH_STD = [0.229, 0.224, 0.225] # Default ImageNet ILSRVC2012

    data_transforms = {
        'train': RGBDutils.Compose([
            RGBDutils.RandomResizedCrop(227),
            RGBDutils.RandomHorizontalFlip(),
            RGBDutils.ToTensor(),
            RGBDutils.Normalize(RGB_AVG, RGB_STD, DEPTH_AVG, DEPTH_STD)]),
        'val': RGBDutils.Compose([
            RGBDutils.Resize(256),
            RGBDutils.CenterCrop(227),
            RGBDutils.ToTensor(),
            RGBDutils.Normalize(RGB_AVG, RGB_STD, DEPTH_AVG, DEPTH_STD)]),
        'test': RGBDutils.Compose([
            RGBDutils.Resize(256),
            RGBDutils.CenterCrop(227),
            RGBDutils.ToTensor(),
            RGBDutils.Normalize(RGB_AVG, RGB_STD, DEPTH_AVG, DEPTH_STD)])
            }
    
    partitions = ['train', 'val', 'test']

    # prepare datasets
    image_datasets = {x: RGBDutils.ImageFolder(os.path.join(path, x), transform=data_transforms[x]) for x in partitions}
    dataset_sizes = {x: len(image_datasets[x]) for x in partitions}
    class_names = image_datasets['train'].classes
    logging.info([f"{x}: {len(image_datasets[x])} images" for x in partitions])
    # prepare dataloaders
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=(x == 'train'), num_workers=num_workers) for x in partitions}
    return dataloaders

# In PyTorch every network is implementd as a nn.Module
class RGBDnet(nn.Module):
    # The parameters are initialized in __init__(self, ...)
    def __init__(self, num_classes):
        super(RGBDnet, self).__init__()
        
        # RGB branch
        model_rgb = models.alexnet(pretrained=True)
        self.rgb_convs = model_rgb.features
        c = model_rgb.classifier
        self.rgb_fcs = nn.Sequential(c[0],c[1],c[2],c[3],c[4],c[5])
        num_ftrs_rgb = c[4].out_features

        # HHA branch
        model_hha = models.alexnet(pretrained=True)
        self.hha_convs = model_hha.features
        c = model_hha.classifier
        self.hha_fcs = nn.Sequential(c[0],c[1],c[2],c[3],c[4],c[5])
        f = model_hha.features
        c = model_hha.classifier
        num_ftrs_hha = c[4].out_features

        # Classifier
        self.classifier = nn.Linear(num_ftrs_rgb+num_ftrs_hha, num_classes)

    # The data flow is defined in forward. No need to specify backward operations (PyTorch takes care of them)
    def forward(self, x):
        x_rgb = self.rgb_convs(x[0])
        x_rgb = x_rgb.view(x_rgb.size(0), 256 * 6 * 6)
        x_hha = self.hha_convs(x[1])
        x_hha = x_hha.view(x_hha.size(0), 256 * 6 * 6)
        x_rgb = self.rgb_fcs(x_rgb)
        x_hha = self.hha_fcs(x_hha)
        x = torch.cat((x_rgb, x_hha), 1)
        x = self.classifier(x)
        return x

    # The data flow is defined in forward. No need to specify backward operations (PyTorch takes care of them)
    def forward(self, x):
        x_rgb = self.rgb_convs(x[0])
        x_rgb = x_rgb.view(x_rgb.size(0), 256 * 6 * 6)
        x_hha = self.hha_convs(x[1])
        x_hha = x_hha.view(x_hha.size(0), 256 * 6 * 6)
        x_rgb = self.rgb_fcs(x_rgb)
        x_hha = self.hha_fcs(x_hha)
        x = torch.cat((x_rgb, x_hha), 1)
        x = self.classifier(x)
        return x

def get_model(num_classes):
    return RGBDnet(num_classes) 
    
def save_hist(train_stat, test_stat, stat_name='accuracy', baseline=None, xmax=20, location='lower right'):
    """
    create matplotlib figures for training statistics
    """
    plt.plot(train_stat)
    plt.plot(test_stat)
    if baseline is not None:
        plt.hlines(baseline, 0, xmax, 'g')
    plt.title(f"Model {stat_name}")
    plt.ylabel(f"{stat_name}")
    plt.xlabel('epoch')
    if baseline is not None:
        plt.legend(['train', 'validation', 'baseline - keras'], loc=location)
    else:
        plt.legend(['train', 'validation'], loc=location)
    plt.savefig(f"{stat_name}.png")
    plt.close()  

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] - %(message)s')
    args = parse_args()
    logging.info(args)

    # Get Dataloaders
    dataloaders = get_dataloaders(args.data_path,img_size=args.image_size,batch_size=args.batch_size)
    logging.info(f"Dataset Found and Loaded")

    # CUDA settings
    if torch.cuda.is_available():
        logging.info('CUDA is available, setting device to CUDA')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    num_classes = len(dataloaders['train'].dataset.classes)
    model = get_model(num_classes)
    print(model)
    #summary(model, ((3, 224,224), (3, 224,224)), device='cpu')

    train_acc_hist = []
    val_acc_hist = []
    train_loss_hist = []
    val_loss_hist = []

    # Training and Validation Loop
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    perlayer_optim = [
    {'params': model.rgb_convs[0].parameters(), 'lr': 0.00}, # conv1 RGB
    {'params': model.rgb_convs[3].parameters(), 'lr': 0.00}, # conv2 RGB
    {'params': model.rgb_convs[6].parameters(), 'lr': 0.00}, # conv3 RGB
    {'params': model.rgb_convs[8].parameters(), 'lr': 0.00}, # conv4 RGB
    {'params': model.rgb_convs[10].parameters(), 'lr': 0.00}, # conv5 RGB
    {'params': model.rgb_fcs[1].parameters(), 'lr': 0.00}, # fc6 RGB
    {'params': model.rgb_fcs[4].parameters(), 'lr': 0.00}, # fc7 RGB
    {'params': model.hha_convs[0].parameters(), 'lr': 0.00}, # conv1 HHA
    {'params': model.hha_convs[3].parameters(), 'lr': 0.00}, # conv2 HHA
    {'params': model.hha_convs[6].parameters(), 'lr': 0.00}, # conv3 HHA
    {'params': model.hha_convs[8].parameters(), 'lr': 0.00}, # conv4 HHA
    {'params': model.hha_convs[10].parameters(), 'lr': 0.00}, # conv5 HHA
    {'params': model.hha_fcs[1].parameters(), 'lr': 0.00}, # fc6 HHA
    {'params': model.hha_fcs[4].parameters(), 'lr': 0.00}, # fc7 HHA
    {'params': model.classifier.parameters(), 'lr': 3e-4} # fc8
    ]

    for param in itertools.chain(model.rgb_convs[0].parameters(),model.rgb_convs[3].parameters(),
                                model.rgb_convs[6].parameters(),model.rgb_convs[8].parameters(),
                                model.rgb_convs[10].parameters(),model.rgb_fcs[1].parameters(),
                                model.rgb_fcs[4].parameters(),
                                model.hha_convs[0].parameters(),model.hha_convs[3].parameters(),
                                model.hha_convs[6].parameters(),model.hha_convs[8].parameters(),
                                model.hha_convs[10].parameters(),model.hha_fcs[1].parameters(),
                                model.hha_fcs[4].parameters(),
                                model.classifier.parameters()):
        param.requires_grad = True

    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    #scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, args.lr, epochs=args.max_epochs, steps_per_epoch=len(train_loader))
    logging.info("Starting Training")
    for epoch in range(args.max_epochs):
        model.train()

        # training statistics
        losses, acc, count = [],[],[]
        for batch_idx, (xb,db,yb) in enumerate((train_loader)):
            #transfer data to GPU
            xb, db, yb = xb.to(device), db.to(device), yb.to(device)
            preds = model((xb,db))
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            #scheduler.step()

            # calculating this way to account for the fact that the
            # last batch may have different batch size
            bs = xb.shape[0]
            # get number of right predictions
            correct_predictions = (preds.argmax(dim=1)==yb).float().sum()
            # add to list
            losses.append(bs*loss.item()), count.append(bs), acc.append(correct_predictions)

        # accumulate/average statistics
        n = sum(count)
        train_loss_epoch = sum(losses)/n
        train_acc_epoch = sum(acc)/n

        train_loss_hist.append(train_loss_epoch)
        train_acc_hist.append(train_acc_epoch)

        model.eval()
        with torch.no_grad():
            losses, acc, count = [],[],[]
            for batch_idx, (xb,db,yb) in enumerate((val_loader)):
                #transfer data to GPU
                xb,db,yb = xb.to(device), db.to(device), yb.to(device)
                preds = model((xb,db))
                loss = criterion(preds, yb)
                bs = xb.shape[0]
                # get number of right predictions
                correct_predictions = (preds.argmax(dim=1)==yb).float().sum()
                # add to list
                losses.append(bs*loss.item()), count.append(bs), acc.append(correct_predictions)

        # accumulate/average statistics
        n = sum(count)
        val_loss_epoch = sum(losses)/n
        val_acc_epoch = sum(acc)/n

        val_loss_hist.append(val_loss_epoch)
        val_acc_hist.append(val_acc_epoch)

    
        print(f"Epoch{epoch}, train_accuracy:{train_acc_epoch:.4f}, val_accuracy:{val_acc_epoch:.4f}, train_loss:{train_loss_epoch:.4f}, val_loss:{val_loss_epoch:.4f}")
    
    logging.info("Finished Training")
    
    save_hist(train_acc_hist, val_acc_hist, 'accuracy', xmax=args.max_epochs, location='lower right')
    save_hist(train_loss_hist, val_loss_hist, 'loss', xmax=args.max_epochs, location='upper right')
    logging.info("Saved training plots")

    logging.info("Evaluating Model on Test Set")
    
    test_loader = dataloaders['test']
    with torch.no_grad():
        losses, acc, count = [],[],[]
        for batch_idx, (xb,db,yb) in enumerate((test_loader)):
            #transfer data to GPU
            xb,db,yb = xb.to(device),db.to(device),yb.to(device)
            preds = model((xb,db))
            loss = criterion(preds, yb)
            bs = xb.shape[0]
            # get number of right predictions
            correct_predictions = (preds.argmax(dim=1)==yb).float().sum()
            # add to list
            losses.append(bs*loss.item()), count.append(bs), acc.append(correct_predictions)

    # accumulate/average statistics
    n = sum(count)
    test_loss = sum(losses)/n
    test_acc = sum(acc)/n
    print(f"test_accuracy:{test_acc:.4f}, test_loss:{train_loss_epoch:.4f}")