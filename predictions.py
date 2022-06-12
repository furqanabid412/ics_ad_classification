import torch
import csv
import numpy as np

def test_predictions(val_dataset,num_classes, data_dir,models):
  for model_name, model in models.items():
    print('=' * 10, '{}'.format(model_name), '=' * 10)
    model = model.eval()
    testloader = torch.utils.data.DataLoader(val_dataset, batch_size=1)
    correct,total = 0,0

    f = open(data_dir + '/' + model_name + ".csv", 'w+', newline='')
    writer = csv.writer(f)
    with torch.no_grad():
      num = 0
      save = np.zeros((len(testloader), num_classes+1))
      for data in testloader:
        images, labels = data
        labels = labels.cuda()
        outputs = model(images.cuda())
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels.cuda()).sum().item()
        prob = torch.nn.functional.softmax(outputs, dim=1)

        prob_and_gt = prob[0].tolist()[0:num_classes]+labels.tolist()

        save[num] = np.asarray(prob_and_gt)
        num += 1
    # print("Accuracy on trainset = ", 100 * correct / total)

    for i in range(len(testloader)):
      writer.writerow(save[i].tolist())

    f.close()
