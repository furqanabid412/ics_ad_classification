import torch
import numpy
from dataloader import imgLoader
from tqdm import tqdm


if __name__ == "__main__":
    #  computed means

    # only training set
    # mean : tensor([0.1416, 0.1416, 0.1417]) std : tensor([0.1823, 0.1823, 0.1824])

    # training+test set
    # mean : tensor([0.1413, 0.1413, 0.1414]) std : tensor([0.1821, 0.1820, 0.1821])


    data_dir = 'C:/Users/Furqan/Desktop/FuzzyIntegralBasedEnsemble/Alzdata'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config = dict(root=data_dir, dataset_stats=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                  img_resize=(224, 224), batch_size=8, num_workers=4, device=device, )

    return_mode = 'dataloader'  # 1. dataset : 2. dataloader
    # Dataloader for train and val
    train_loader = imgLoader(config=config, mode='train', return_mode='dataloader')
    val_loader = imgLoader(config=config, mode='val', return_mode='dataloader')

    cnt = 0
    fst_moment = torch.empty(3)
    snd_moment = torch.empty(3)

    loader={0:train_loader,1:val_loader}

    for i in range(2):
        for images,labels in tqdm(loader[i][0]):

            b, c, h, w = images.shape
            nb_pixels = b * h * w
            sum_ = torch.sum(images, dim=[0, 2, 3])
            sum_of_square = torch.sum(images ** 2,
                                      dim=[0, 2, 3])
            fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
            snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)
            cnt += nb_pixels

    mean, std = fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)
    print("mean :", mean,"std :",std)
