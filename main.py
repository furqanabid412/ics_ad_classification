from dataloader import imgLoader
from trainer import get_model
from predictions import test_predictions
from sugeno import ensemble_sugeno

import warnings
warnings.filterwarnings('ignore')
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
import os
import pandas as pd
import numpy as np
from sklearn.metrics import *

# data_dir = '/media/furqan/Terabyte/Semester/Spring2022/Intelligent control/Project/IntelligentControl/data'
# data_dir = 'C:/Users/Furqan\Desktop\FuzzyIntegralBasedEnsemble\data'

data_dir = 'C:/Users/Furqan/Desktop/FuzzyIntegralBasedEnsemble/Alzdata'
model_name=['vgg11','googlenet','squeezenet','wideresnet']

def get_predictions(mode='load_model'):
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  # configuration for dataloader
  config = dict(root=data_dir, dataset_stats=dict(mean=[0.1413, 0.1413, 0.1414], std=[0.1821, 0.1820, 0.1821]),
                img_resize=(224, 224), batch_size=16, num_workers=0, device=device, )

  return_mode = 'dataloader'  # 1. dataset : 2. dataloader
  # Dataloader for train and val
  train_loader = imgLoader(config=config, mode='train', return_mode='dataloader')
  val_loader = imgLoader(config=config, mode='val', return_mode='dataloader')

  mode = 'train_model'  # 1.'train_model' : 2.'load_model'

  vgg11 = get_model(name='vgg11', mode=mode, config=config, train_loader=train_loader, val_loader=val_loader)
  googlenet = get_model(name='googlenet', mode=mode, config=config, train_loader=train_loader, val_loader=val_loader)
  squeezenet = get_model(name='squeezenet', mode=mode, config=config, train_loader=train_loader, val_loader=val_loader)
  wideresnet = get_model(name='wideresnet', mode=mode, config=config, train_loader=train_loader, val_loader=val_loader)

  models = {'vgg11': vgg11, 'googlenet': googlenet, 'squeezenet': squeezenet, 'wideresnet': wideresnet}
  val_dataset, val_classes = imgLoader(config=config, mode='val', return_mode='dataset')
  test_predictions(val_dataset=val_dataset, num_classes=len(val_classes), data_dir=data_dir, models=models)

def load_predictions():
  from sugeno import metrics
  classes = ['AD', 'CN']
  pred = dict()
  for filename in model_name:
    file = os.path.join(data_dir, filename+'.csv')
    data = np.asarray(pd.read_csv(file,header=None))
    pred[filename],labels=data[:,:2],data[:,-1].astype(np.int32)
    argmax=np.argmax(pred[filename],axis=1)
    metrics(labels, argmax, classes,name=filename)
  return pred,labels

# Get prediction probabilities on test set
# 1.'train_model' : 2.'load_model'

# get_predictions(mode='load_model')

# Load prediction probabilities on test set
pred,labels = load_predictions()

ensemble_sugeno(names=model_name,pred=pred,labels=labels)


print('program-finished')