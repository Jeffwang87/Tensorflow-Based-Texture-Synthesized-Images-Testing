"""
Load texture synthesized images to test accuracy of vgg net prediction
authors: Jianxin Wang

"""

from torchvision import models
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import os
from PIL import Image
from torchvision import models
from skimage import io
import numpy as np

class ImageData(Dataset):
    def __init__(self, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.image = []
        self.label = []
        for image in sorted(os.listdir(directory)):
          self.image.append(directory+'/'+image)
        f = open("/content/drive/MyDrive/Human_AI/val_2.txt", "r")
        for x in f:
          self.label.append(x.split()[1])
        self.transform = transform
    def __len__(self):
      return len(self.image)
    def __getitem__(self, idx):
      image = self.image[idx]
      label = self.label[idx]
      if self.transform:
        img = self.transform(Image.open(image))
      else:
        img = io.imread(image)
      target = torch.tensor(int(label))
      return img, target
      
vggnet = models.vgg19(pretrained=True)
vggnet.cuda()

test = ImageData(transform=transforms.Compose([
                                               transforms.Resize([224, 224]),
                                              transforms.ToTensor()]))
test_data = DataLoader(test, batch_size=64,shuffle=False, num_workers=1,pin_memory=True)

vggnet.eval()
score = []
for i, data in enumerate(test_data):
    images, labels = data
    tmp = []
    tmp = torch.squeeze(labels.long())
    images, labels = images.cuda(),  tmp.cuda()
    outputs = vggnet(images)
    outputs_numpy = outputs.cpu().data.numpy()
    outputs_argmax = np.argmax(outputs_numpy,axis=1)
    #print(outputs_argmax)
    labels_numpy = labels.cpu().data.numpy()
    #print(labels_numpy)
    score = np.concatenate((score,(labels_numpy==outputs_argmax).astype(int)),axis=0)
    #break

meanAccuracy3 = sum(score)/len(score)
