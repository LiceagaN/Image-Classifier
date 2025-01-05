import argparse
import torch
import json
import os
import random
from torch import nn
from torch import optim
from PIL import Image
import numpy as np
import torch.nn.functional as F
from torchvision import datasets, transforms, models



def load_checkpoint(file):
    model = models.densenet169(pretrained=True)
    checkpoint = torch.load(file)
    learning_rate = checkpoint['learning_rate']
    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    state_dict = checkpoint['optimizer_state_dict']
    optimizer.load_state_dict(state_dict)
    epoch = checkpoint['epoch']
    inputsize = checkpoint['input_size']
    outputsize = checkpoint['output_size']
    
    return model, optimizer, inputsize, outputsize, epoch

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    preprocess = transforms.Compose([
        transforms.Resize(256),             # Resize the image to 256x256 pixels
        transforms.CenterCrop(224),         # Crop the center to 224x224 pixels
        transforms.ToTensor(),              # Convert the PIL image to a PyTorch tensor
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],      # Normalize the image with pre-defined mean values
            std=[0.229, 0.224, 0.225]        # Normalize the image with pre-defined standard deviation values
        ),
    ])

    image = Image.open(path_image)
    input_tensor = preprocess(image)
    return input_tensor
 
def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    image = Image.open(path_image)
    image = process_image(image)
    
    with torch.no_grad():
        model.eval()
        image = image.view(1,3,224,224)
        image = image.to(device)
        model = model.to(device)
        
        predictions = model.forward(image)
        
        predictions = torch.exp(predictions)
        top_ps, top_class = predictions.topk(topk, dim=1)
    
    return top_ps, top_class


parser = argparse.ArgumentParser(
        description = 'Model training program made in pytorch to classify flowers',
        prog= 'pred'
)

parser.add_argument('path_image', nargs='?', default='flowers/test/11/image_03141.jpg', help='Format: path/to/image')
parser.add_argument('path_checkpoint', nargs='?', default='checkpoint.pth')
parser.add_argument('--top_k', default=5, type=int, help='Number of top classes')
parser.add_argument('--categories', default='cat_to_name.json', help='The file were are stored all the classes')
parser.add_argument('--gpu', default=True, help='The gpu is enabled by default, if you do not want to use it please set it as False')

args = parser.parse_args()
path_image = args.path_image
path_checkpoint = args.path_checkpoint
top_k = args.top_k
categories = args.categories
gpu = args.gpu

if args.gpu == False:
    print('\nWe are going to use CPU for the prediction!!!')
    gpu = False
else:
    if torch.cuda.is_available():
        gpu = True
        print('\nWe are going to use GPU for the prediction!!!')
    else:
        gpu = False
device = torch.device("cuda" if gpu else "cpu")

print('Loading in a mapping from category label to category name')
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
model, optimizer, input_size, output_size, epoch  = load_checkpoint(path_checkpoint)
model.eval()
probs, classes = predict(path_image, model)
probs = probs.data.cpu()
probs = probs.numpy().squeeze()
classes = classes.data.cpu()
classes = classes.numpy().squeeze()
classes = [cat_to_name[str(clas)].title() for clas in classes]
print('The model predictions for this flower are: \n' )
for x in range(top_k):
    print("{:<20}\t {:.3f}%".format(classes[x] + ":", probs[x]*100))

        