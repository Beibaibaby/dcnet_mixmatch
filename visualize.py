import os
import torch
import numpy as np
import argparse
import time
import torch.nn.functional as F
from imgaug import augmenters as iaa

from tqdm import trange
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.autograd import Variable

from dataset_utility import dataset, ToTensor
from dcnet import DCNet
from skimage.util import random_noise
import pdb

import numpy as np
import matplotlib.pyplot as plt

ts=np.load('test.npy')

for i in range(16):
    plt.imshow(ts[0][i])
    plt.savefig("./fig/test"+str(i)+".pdf")