import os
import time
import pickle

import torch
import torch.nn.functional as F

import PIL.Image
import numpy as np
import matplotlib.pyplot as plt
import sys

#from models.cusp.torch_utils import persistence
#import models.cusp.training
#import models.cusp.dnnlib
#save the literal filepath to both directories as strings

folder_root = '/mnt/c/Users/PernilleKopperud/Documents/InfoSec/MasterThesis/master_thesis/MSc_FR_Bias/src'
sys.path.append(folder_root)

import models.cusp.legacy
from models.cusp.torch_utils import persistence
import models.cusp.training
from models.cusp import dnnlib

from models.cusp.training.networks import VGG, module_no_grad
from models.cusp.torch_utils import misc
'''
tu_path = os.path.join('/mnt/c/Users/PernilleKopperud/Documents/InfoSec/MasterThesis/master_thesis/MSc_FR_Bias/src/models','cusp','torch_utils')
dnnlib_path = os.path.join('/mnt/c/Users/PernilleKopperud/Documents/InfoSec/MasterThesis/master_thesis/MSc_FR_Bias/src/models','cusp','dnnlib')

#add those strings to python path
sys.path.append(tu_path)
sys.path.append(dnnlib_path)
print(sys.path)

'''

def convert_pkl_to_pt_file(checkpoint_path):
    #print(f"Loading CUSP generator from path: {checkpoint_path}")
    with open(checkpoint_path, 'rb') as f:
        decoder = pickle.load(f)['G_ema'].cuda()
    print('Loading done!')

    state_dict = decoder.state_dict()
    torch.save(state_dict, "cusp_net.pt")
    print('Converting done!')


def load_pretrained_model(filename : str, device : torch.device):
    model = torch.load(filename)
    print(model)

    return model





def generate_synthetic_data():



    return

load_pretrained_model('/mnt/c/Users/PernilleKopperud/Documents/InfoSec/MasterThesis/master_thesis/MSc_FR_Bias/src/data/CUSP_network.pkl',torch.device('cpu'))
path = '/mnt/c/Users/PernilleKopperud/Documents/InfoSec/MasterThesis/master_thesis/MSc_FR_Bias/src/data/CUSP_network.pkl'
#convert_pkl_to_pt_file(path)