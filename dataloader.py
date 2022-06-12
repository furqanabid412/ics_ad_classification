import torch
from torchvision import datasets, transforms
import os
from torch.utils.data import Dataset

def imgLoader(config,mode='train',return_mode='dataloader'):
  data_transforms = transforms.Compose([
      transforms.Resize(config['img_resize']),
      transforms.ToTensor(),
      transforms.Normalize(config['dataset_stats']['mean'], config['dataset_stats']['std'])
    ])
  image_dataset = datasets.ImageFolder(os.path.join(config['root'], mode),data_transforms)
  class_names = image_dataset.classes
  if return_mode == 'dataset':
    return image_dataset,class_names
  elif return_mode == 'dataloader':
    dataloaders = torch.utils.data.DataLoader(image_dataset, batch_size=config['batch_size'],shuffle=True, num_workers=config['num_workers'])
    return dataloaders,class_names,len(image_dataset)
