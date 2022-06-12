import matplotlib.pyplot as plt
import os

def plot(val_loss, train_loss, typ,data_dir,name):
  plt.title("{} after epoch: {}".format(typ, len(train_loss)))
  plt.xlabel("Epoch")
  plt.ylabel(typ)
  plt.plot(list(range(len(train_loss))), train_loss, color="r", label="Train " + typ)
  plt.plot(list(range(len(val_loss))), val_loss, color="b", label="Validation " + typ)
  plt.legend()

  path = os.path.join(data_dir,'models',name,name+'_' +typ + ".png")
  plt.savefig(path)
  # plt.figure()
  plt.show()
  plt.close()