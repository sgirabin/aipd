import json
import numpy as np
import torch
import argparse
import os

from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from collections import OrderedDict
from time import time

DIRECTORY_FORMAT={'test','train','valid'}


def is_directory_exist(path):
    if(not os.path.isdir(path)):
        return False;
    return True

def validate_data_dir(data_path):
    if (not is_directory_exist(data_path)):
        raise Exception("Data Directory does not exist")
    else:
        subdir = os.listdir(data_path)
        if (not set(subdir).issubset(DIRECTORY_FORMAT)):
            raise Exception('Missing required directory (test, train, valid)')

def validate_save_dir(save_path):
    if (not is_directory_exist(save_path)):
        raise Exception("Save Directory does not exist")

def parse_arguments():
    parser = argparse.ArgumentParser(prog='train', description='AIPD - Image Classifier Parameters')
    parser.add_argument('data_dir', type=str, default='flowers', action='store', help='Data Directory Location (mandatory, default=flowers)')
    parser.add_argument('--save_dir', type=str, dest='save_dir', default='.', action='store', help='Save Directory (optional, default=current directory)')
    parser.add_argument('--arch', type=str, dest='arch', default='vgg13', choices=['vgg13', 'densenet121'], action='store' , help='vgg13 or densenet121 (optional, default=vgg16)')
    parser.add_argument('--epochs', type=int, dest='epochs', default='3', action='store' , help='Epochs/Cycle for Training (optional, default=3)')
    parser.add_argument('--learning_rate', type=float,dest='learning_rate', default='0.001', action='store', help='Learning Rate (optional, default=0.01')
    parser.add_argument('--hidden_units', type=int, dest='hidden_units', default='512', action='store', help='Hidden Units (optional, default=512)')
    parser.add_argument('--gpu', dest='gpu',  default=True, action='store_true', help='Use GPU (default=True)')

    return parser.parse_args()

def load_images(path):
    print("--- loading images started ---")

    train_path = path + '/train'
    valid_path = path + '/valid'
    test_path  = path + '/test'

    train_data_transformer = transforms.Compose([
                                transforms.RandomRotation(30),
                                transforms.Resize(255),
                                transforms.CenterCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                                ])
    valid_data_transformer = transforms.Compose([
                                transforms.Resize(255),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                                ])

    image_datasets = {
        'train': datasets.ImageFolder(train_path, transform=train_data_transformer),
        'valid': datasets.ImageFolder(valid_path, transform=valid_data_transformer),
        'test' : datasets.ImageFolder(test_path, transform=valid_data_transformer)
    }

    dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
        'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size=64, shuffle=True),
        'test' : torch.utils.data.DataLoader(image_datasets['test'] , batch_size=64, shuffle=True)
    }

    images = { 'dataset': image_datasets, 'loader': dataloaders }

    print("--- loading images finished ---")

    return images

def build_model(arch, layers, learning_rate):
    print("--- start building model ---")

    if arch == 'densenet121':
        model = models.densenet121(pretrained=True)
        fc1 = 1024
    elif arch == 'vgg13':
        model = models.vgg13(pretrained = True)
        fc1 = 25088
    else:
        raise ValueError('Model arch error.')

    for param in model.parameters():
        param.requires_grad = False
    model.classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(fc1, layers)),
                              ('relu', nn.ReLU()),
                              ('dropout', nn.Dropout(0.2)),
                              ('fc2', nn.Linear(layers, 102)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    print("--- building model completed ---")

    return { 'model': model, 'criterion': criterion, 'optimizer': optimizer }

def train_model(model, trainloader, validloader, epochs, print_steps, criterion, optimizer, device='cpu'):
    print("--- model training started ---")

    steps = 0
    model.to(device)

    for e in range(epochs):
        t1 = time()
        running_loss = 0
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_steps == 0:
                correct = 0
                total = 0

                with torch.no_grad():
                    for data in validloader:
                        images, labels = data[0].to(device), data[1].to(device)
                        outputs = model(images)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()

                valid_accuracy = correct / total

                print(" Epoch: {}/{} --> ".format(e+1, epochs),
                      " Training Loss: {:.4f}".format(running_loss/print_steps),
                      " ; ",
                      " Validation Accuracy: {}".format(round(valid_accuracy,4)))

                running_loss = 0

        t2 = time()
        print("Elapsed Time for epoch {}: {}s.".format(epochs+1, t2-t1))

    print("--- model training completed ---")
    return model

def test_model(testloader, model, device='cpu'):
    print("--- model testing started ---")

    correct = 0
    total = 0

    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test accuracy: %d %%" % (100 * correct / total))

    print("--- model testing finished ---")

    return correct / total


def save_model(save_dir, image_loader, model):
    print("--- saving model ---")
    model.class_to_idx = image_loader['train'].class_to_idx

    checkpoint = {
        'model': model,
        'class_to_idx': model.class_to_idx,
        'classifier': model.classifier,
        'state_dict': model.state_dict()
    }

    checkpoint_fn = 'checkpoint.pth'
    if not os.path.isdir(save_dir): os.makedirs(save_dir)
    if os.path.isfile(save_dir+checkpoint_fn): os.remove(save_dir+checkpoint_fn)

    torch.save(checkpoint, save_dir+checkpoint_fn)
    print(" ---  model ('{}') is saved at '{}'---".format(checkpoint_fn, save_dir))

    return (save_dir+checkpoint_fn)

def main():

    ## parse arguments
    args = parse_arguments()

    ## validate input arguments
    validate_data_dir(args.data_dir)
    validate_save_dir(args.save_dir)
    if args.gpu:
        device = 'cuda'
    else:
        device ='cpu'

    ## load image
    images = load_images(args.data_dir)

    ## build model
    model = build_model(args.arch, args.hidden_units, args.learning_rate)

    ## train model
    print_steps = 10
    model['model'] = train_model(model['model'],
                        images['loader']['train'],
                        images['loader']['valid'],
                        args.epochs,
                        print_steps,
                        model['criterion'],
                        model['optimizer'],
                        device)

    ## test model
    test_model(images['loader']['test'], model['model'], device)

    ## save model
    save_model(args.save_dir, images['dataset'], model['model'])

    return None

if __name__ == "__main__":
    main()
