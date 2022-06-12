import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from plots import plot
from tqdm import tqdm

def get_model(name,mode,config,train_loader,val_loader):

  if mode == 'train_model':

    if name == 'vgg11':
      model = models.vgg11_bn(pretrained=True)
    elif name == 'googlenet':
      model = models.googlenet(pretrained=True)
    elif name == 'squeezenet':
      model = models.squeezenet1_1(pretrained = True)
    elif name == 'wideresnet':
      model =torch.hub.load('pytorch/vision:v0.10.0', 'wide_resnet50_2', pretrained=True)
      # model = models.wideresnet_50_2(pretrained=True)
    else:
      raise ValueError('{} model not found'.format(name))

    model = trainer(root=config['root'], model=model, name=name, device=config['device'],
                    train_loader=train_loader, val_loader=val_loader)
  elif mode == 'load_model':
    # path = config['root'] + "/" + name + ".pt"
    path = config['root'] + "/" + "models/" + name + "/" + name +".pt"
    model = torch.load(path)
    # model.eval()

  return model


def trainer(root,model,name,device,train_loader,val_loader):
  train_dataloader, train_classes, train_size=train_loader[0],train_loader[1],train_loader[2]
  val_dataloader, val_classes, val_size=val_loader[0],val_loader[1],val_loader[2]

  num_classes = len(train_classes)
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.99)
  step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

  if name == 'vgg11':
    num_ftrs = model.classifier[0].in_features
    model.classifier = nn.Linear(num_ftrs, num_classes)

  elif name == 'googlenet':
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

  elif name == 'squeezenet':
    model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))

  elif name == 'wideresnet':
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

  model = model.to(device)
  train_dict = dict(model=model, criterion=criterion, optimizer=optimizer,
                    scheduler=step_lr_scheduler, epochs=40, device=device, model_name=name)
  dataset_dict = dict(train=dict(dataloader=train_dataloader, datasize=train_size),
                      val=dict(dataloader=val_dataloader, datasize=val_size), root=root)
  model = train_model(train_dict, dataset_dict)
  return model



def train_model(train_dict,dataset_dict):
  val_loss_gph , train_loss_gph, val_acc_gph, train_acc_gph = [], [], [], []
  best_model_wts = copy.deepcopy(train_dict['model'].state_dict())
  best_acc = 0.0
  print('='*10,'{}'.format(train_dict['model_name']),'='*10)

  for epoch in range(train_dict['epochs']):

    for phase in ['train', 'val']:
      if phase == 'train':
        train_dict['model'].train()
        dataset = dataset_dict['train']
      else:
        train_dict['model'].eval()
        dataset = dataset_dict['val']

      running_loss,running_corrects = 0.0,0

      for inputs, labels in dataset['dataloader']:
        inputs,labels = inputs.to(train_dict['device']),labels.to(train_dict['device'])
        with torch.set_grad_enabled(phase == 'train'):
          outputs = train_dict['model'](inputs)
          _, preds = torch.max(outputs, 1)  # was (outputs,1) for non-inception and (outputs.data,1) for inception
          loss = train_dict['criterion'](outputs, labels)
          if phase == 'train':
            train_dict['optimizer'].zero_grad()
            loss.backward()
            train_dict['optimizer'].step()

        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

      if phase == 'train':
        train_dict['scheduler'].step()

      epoch_loss = running_loss / dataset['datasize']
      epoch_acc = running_corrects.double() / dataset['datasize']

      # print("epoch loss",epoch_loss,"epoch acc",epoch_acc)

      if phase == 'train':
        train_loss_gph.append(epoch_loss)
        train_acc_gph.append(epoch_acc.item())
      if phase == 'val':
        val_loss_gph.append(epoch_loss)
        val_acc_gph.append(epoch_acc.item())

      print('Epoch {} of {}----  {} Loss: {:.4f} Acc: {:.4f}'.format(epoch+1,train_dict['epochs'], phase, epoch_loss, epoch_acc))

      # deep copy the model
      if phase == 'val' and epoch_acc >= best_acc:
        best_acc = epoch_acc
        best_model_wts = copy.deepcopy(train_dict['model'].state_dict())
        path=dataset_dict['root'] + "/"+"models/" + train_dict['model_name']+"/"+train_dict['model_name']
        torch.save(train_dict['model'].state_dict(), path + ".pth")
        torch.save(train_dict['model'], path + ".pt")
        # print('==>Model Saved')

  plot(val_loss_gph, train_loss_gph, "Loss",dataset_dict['root'],train_dict['model_name'] )
  plot(val_acc_gph, train_acc_gph, "Accuracy",dataset_dict['root'],train_dict['model_name'] )

  # model.classifier = nn.Linear(num_ftrs, num_classes)

  print('Best val Acc: {:4f}'.format(best_acc))
  # load best model weights
  train_dict['model'].load_state_dict(best_model_wts)
  return train_dict['model']