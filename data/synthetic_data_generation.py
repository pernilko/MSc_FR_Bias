import os
import pandas as pd
import torch
import pickle
import sys
from enum import Enum
from pathlib import Path
from typing import Optional


#sys.path.insert(0, '/mnt/c/Users/PernilleKopperud/Documents/InfoSec/MasterThesis/master_thesis/MSc_FR_Bias')
import models.cusp.torch_utils.persistence
import models.cusp.training
import models.cusp.dnnlib

def convert_pkl_to_pt_file():
    checkpoint_path = "data/CUSP_network.pkl"
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

load_pretrained_model('data/CUSP_network.pkl',torch.device('cpu'))
#convert_pkl_to_pt_file()