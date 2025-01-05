import argparse
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import os

parser = argparse.ArgumentParser(
        description = 'Model training program made in pytorch to classify flowers',
        prog= 'train'
)
parser.add_argument('data_directory', default='flowers',nargs='?')
parser.add_argument('--save_dir', action='store', dest='data_save', help='where the checkpoint will be saved')
parser.add_argument('--arch', action='store', default='densenet169', choices=['densenet169','alexnet'], 
                    help='model architecture, you can choose betwwen densenet169 and alexnet')

#Hyperparameters
parser.add_argument('--learning_rate', default='0.002', type=float, help='Hyperparameter for the model training', dest='learning_rate')
parser.add_argument('--epochs', default=15, type=int, help='Number of epochs that the model will be trained', dest='epochs')
parser.add_argument('--hidden_unit_per_layer', nargs=2, default=[832,416], type=int, 
                    help='This is for the units for each hidden unit, please enter 2 number in decrease order between 1664 and 102',
                    dest='hidden_unit_per_layer')
#Set the gpu parameter
parser.add_argument('--gpu', default=True, help='The gpu is enabled by default, if you do not want to use it please set it as False',
                    dest='gpu')

args = parser.parse_args()

data_dir =  args.data_directory
save_dir =  args.data_save
arch = args.arch
learning_rate = args.learning_rate
hidden_unit_per_layer = args.hidden_unit_per_layer
epochs = args.epochs
gpu = args.gpu  
if args.gpu == False:
    print('\nWe are going to use CPU for the training!!!')
    gpu = False
else:
    if torch.cuda.is_available():
        gpu = True
        print('\nWe are going to use GPU for the training!!!')
    else:
        gpu = False

if 102 <= hidden_unit_per_layer[1] <= hidden_unit_per_layer[0]<= 1664:
    print('The units per layer are ok')
else:
    hidden_unit_per_layer = [832, 416]
    print('\nThe units per layer are incompatible, we are going to use the default valuels {}'.format([832,416]))
    
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.5, 0.5, 0.5],
                                                            [0.5, 0.5, 0.5])])
test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor()])
validation_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor()])

train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
validation_data = datasets.ImageFolder(valid_dir, transform=validation_transforms)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)
validationloader = torch.utils.data.DataLoader(validation_data, batch_size=64, shuffle=True)

#Building the network
if arch == 'densenet169':
    model = models.densenet169(pretrained= True)
    print('MODEL DENSENET169:')
    print(model)
else:
    model = models.alexnet(pretrained=True)
    print('MODEL ALEXNET:')
    print(model)

device = torch.device("cuda" if gpu else "cpu")
print(device)
#Freeze the parameters 
for param in model.parameters():
    param.requires_grad = False
if arch == 'densenet169':
    model.classifier = nn.Sequential(nn.Linear(1664, hidden_unit_per_layer[0]),
                                    nn.ReLU(),
                                    nn.Dropout(0.2),
                                    nn.Linear(hidden_unit_per_layer[0], hidden_unit_per_layer[1]),
                                    nn.ReLU(),
                                    nn.Dropout(0.2),
                                    nn.Linear(hidden_unit_per_layer[1],102),
                                    nn.LogSoftmax(dim=1))
else:
    model.classifier = nn.Sequential(nn.Linear(9216, hidden_unit_per_layer[0]),
                                    nn.ReLU(),
                                    nn.Dropout(0.2),
                                    nn.Linear(hidden_unit_per_layer[0], hidden_unit_per_layer[1]),
                                    nn.ReLU(),
                                    nn.Dropout(0.2),
                                    nn.Linear(hidden_unit_per_layer[1],102),
                                    nn.LogSoftmax(dim=1))

print('\nClassifier layers of the model:   \n')
print(model.classifier)

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
model.to(device)

print('Training has started')
print('\n---------------------------------------------------')
steps = 0
train_losses, validation_losses = [], []
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        log_ps = model(images)
        loss = criterion(log_ps, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    else:
        validation_loss = 0
        accuracy = 0
        with torch.no_grad():
            for images, labels in validationloader:
                images, labels = images.to(device), labels.to(device)
                log_ps = model(images)
                validation_loss += criterion(log_ps, labels)
                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor))
        train_losses.append(running_loss/len(trainloader))
        validation_losses.append(validation_loss/len(validationloader))
        print("Epoch: {}/{}.. ".format(e+1, epochs),
              "Training Loss: {:.3f}.. ".format(running_loss/len(trainloader)),
              "Validation Loss: {:.3f}.. ".format(validation_loss/len(validationloader)),
              "Validation Accuracy: {:.3f}".format(accuracy/len(validationloader)))

print('Testing has started')
print('--------------------------------------------------------------------')
steps = 0
running_loss = 0
for e in range(epochs):
    steps += 1
    test_loss = 0
    accuracy = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)                  
            test_loss += batch_loss.item()
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()                    
    print("Epoch {}/{}.. ".format(e+1, epochs),
            "Test loss: {:.3f}.. ".format(test_loss/len(testloader)),
            "Test accuracy: {:.3f}".format(accuracy/len(testloader)))
    running_loss = 0
    model.train()

print('Compleating the process\n')

if save_dir:
    if not os.path.exists(save_dir):
        os.mkdir(savedir)
    save_model = saved_dir + '/checkpoint.pth' 
else:
    save_model = 'checkpoint.pth'

image_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
class_to_idx = image_datasets.class_to_idx
model.class_to_idx = class_to_idx
checkpoint = {'input_size': 1664,
              'output_size': 102,
              'classifier': model.classifier,
              'class_to_idx': model.class_to_idx,
              'learning_rate': learning_rate,
              'epoch': epochs,
              'optimizer_state_dict': optimizer.state_dict(),
              'model_state_dict': model.state_dict()}

torch.save(checkpoint, save_model)
print('The model has been saved')