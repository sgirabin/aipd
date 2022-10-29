import matplotlib.pyplot as plt
import numpy as np

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, models, transforms

import time
from PIL import Image
import argparse
import os
import json


def parse_arguments():
  parser = argparse.ArgumentParser(prog='Predict', description='AIPD - Image Classifier Parameters')

  parser.add_argument('image_file', type=str, action='store', help='Image File (required)')
  parser.add_argument('checkpoint_file', type=str, default='.', action='store', help='Checkpoint File (required)')
  parser.add_argument('--category_names', type=str, dest='category_names', default='cat_to_name.json', action='store' , help='Category Filename (optional, default=cat_to_name.json)')
  parser.add_argument('--top_k', type=int, dest='top_k', default='5', action='store' , help='Result Top N (optional, default=5)')
  parser.add_argument('--gpu', dest='gpu',  default=True, action='store_true', help='Use GPU (default=True)')

  return parser.parse_args()

def validate_input(filename):
  if (not os.path.exists(filename)):
      raise Exception("File {file} does not exist".format(filename))

def load_model_from_checkpoint(filename):
  print("--- loading model from checkpoint started ---")
  checkpoint = torch.load(filename)

  model = checkpoint['model']
  for param in model.parameters():
      param.requires_grad = False

  model.classifier = checkpoint['classifier']
  model.class_to_idx = checkpoint['class_to_idx']
  model.load_state_dict(checkpoint['state_dict'])

  print("--- loading model completed ---")

  return model

def load_image(image_file):
  print("--- loading images started ---")
  image = Image.open(image_file)
  transform = transforms.Compose([transforms.Resize(255),
                                  transforms.CenterCrop(224),
                                  transforms.ToTensor(),
                                  transforms.Normalize([0.485, 0.456, 0.406],
                                                       [0.229, 0.224, 0.225])])
  image = transform(image)
  image = np.array(image)

  print("--- loading images completed ---")
  return image

def predict(image, model, device, topk):
  print("--- start prediction ---")
  image = torch.from_numpy(image).type(torch.FloatTensor)
  image = image.unsqueeze_(0).float()
  image = image.to(device)

  model = model.to(device)
  model.eval()
  with torch.no_grad():
    output = model.forward(image)

  probs, indeces = torch.exp(output).topk(topk)

  probs   =   probs.to('cpu').numpy().tolist()[0]
  indeces = indeces.to('cpu').numpy().tolist()[0]

  mapping = {val: key for key, val in model.class_to_idx.items()}
  classes = [mapping[item] for item in indeces]

  print("--- prediction completed ---")
  return probs, classes


def main():

    ## parse arguments
    args = parse_arguments()

    ## validate arguments
    validate_input(args.image_file)
    validate_input(args.checkpoint_file)
    validate_input(args.category_names)
    if args.gpu:
        device = 'cuda'
    else:
        device ='cpu'

    ## load checkpoint - pre-trained model
    model = load_model_from_checkpoint(args.checkpoint_file)

    ## load and process image
    image = load_image(args.image_file)

    ## predict
    probs, classes = predict(image, model, device, topk=args.top_k)

    ## load category name (mapping)
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    ## print prediction
    names = [cat_to_name[key] for key in classes]
    print("Prediction Result:")
    for idx, item in enumerate(probs):
      probs[idx] = round(item*100, 2)
      print("No {}. Class {} ({}) - probability = {:.2f}%".format(str(idx+1), names[idx], classes[idx], probs[idx]))

    return None

if __name__ == "__main__":
    main()
