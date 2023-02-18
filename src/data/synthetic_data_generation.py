import os
import time
import pickle
import torch
import torch.nn.functional as F
import PIL.Image
import numpy as np
import matplotlib.pyplot as plt
import sys

folder_root = '/mnt/c/Users/PernilleKopperud/Documents/InfoSec/MasterThesis/master_thesis/MSc_FR_Bias/src'
sys.path.append(folder_root)

from models.cusp import legacy
from models.cusp.torch_utils import persistence
import models.cusp.training
from models.cusp import dnnlib
from models.cusp.training.networks import VGG, module_no_grad
from models.cusp.torch_utils import misc


'''
Method for reading the filenames of the input images to be subjected to age editing
Input: images_path -> path of where the input images are located
Output: batch_of_filenames -> filenames for the current path of images
'''
def read_image_filenames(images_path : str):
    batch_of_filenames = [
      os.path.join(images_path,f) 
      for  f in next(iter(os.walk(images_path)))[2] 
      if f[-4:] == '.png'
    ]
    return batch_of_filenames

'''
Method loads the pre-trained CUSP model
Input: device 
Output: g_ema
'''
def load_cusp(device : torch.device, weights_path : str):

    #weights_path = 'data/cusp_network.pkl'
    vgg_path = "data/dex_imdb_wiki.caffemodel.pt"

    
    with open(weights_path, 'rb') as f:
        data = legacy.load_network_pkl(f)
    
    g_ema = data['G_ema'] # exponential movign average model

    vgg = VGG()
    vgg_state_dict = torch.load(vgg_path)
    vgg_state_dict = {k.replace('-', '_'): v for k, v in vgg_state_dict.items()}
    vgg.load_state_dict(vgg_state_dict)
    module_no_grad(vgg) 

    g_ema.skip_grad_blur.model.classifier = vgg  

    g_ema = g_ema.to(device).eval().requires_grad_(False)

    return g_ema

'''
Method 
'''
def generate_synthetic_data(G, img, label, global_blur_val=None, mask_blur_val=None, return_msk=False):
    print("num classes: ", G.attr_map.fc0.init_args[0])
    print("label: " ,torch.tensor(label))
    ohe_label = torch.nn.functional.one_hot(torch.tensor(label), num_classes=71).to(img.device)

    _, c_out_skip = G.content_enc(img)

    s_out = G.style_enc(img)[0].mean((2,3))

    truncation_psi = 1
    truncation_cutoff = None
    s_out = G.style_map(s_out, None, truncation_psi, truncation_cutoff)

    a_out = G.attr_map(ohe_label.to(s_out.device), None, truncation_psi, truncation_cutoff)

    w = G.__interleave_attr_style__(a_out, s_out)

    for i, (f,_) in enumerate(zip(G.skip_transf, c_out_skip)):
        if f is not None:
            c_out_skip[i] = G._batch_blur(c_out_skip[i], blur_val = global_blur_val)
    
    cam = G.skip_grad_blur(img.float())
    msk = cam
    for i, (f, c) in enumerate(zip(G.skip_transf, c_out_skip)):
        if f is not None:
            im_size = c.size(-1)
            blur_c = G._batch_blur(c, blur_val= mask_blur_val)
            if msk.size(2) != im_size:
                msk = F.interpolate(msk,size=(im_size,im_size), mode='area')
            merged_c = c * msk + blur_c * (1 - msk)
            c_out_skip[i] = merged_c
    
    img_out = G.image_dec(c_out_skip, w)

    if return_msk:
        to_return = (img_out,msk,cam) if G.learn_mask is not None else (img_out,None,None)
    else:
        to_return = img_out
    

    return to_return

'''
Method prepares data for applying age editing
Parameters:
    side (int) : 
    batch_of_filenames () :
    data_labels () : 
    g_ema () :
Return:
    out_tensor () :
    images_as_tensor () :
    labels_exp () :
'''
def prep_data(side : int, batch_of_filenames, data_labels, g_ema):
    images = []

    for file in batch_of_filenames:
        img = np.array(PIL.Image.open(file).resize((side, side)), dtype=np.float32).transpose((2,0,1))
        images.append(img)

    images_as_tensor = (torch.tensor(np.array(images))/256*2-1).cuda()

    aging_steps = 8

    number_of_images = images_as_tensor.shape[0]
    images_as_tensor_exp = images_as_tensor[:, None].expand([number_of_images, aging_steps, *images_as_tensor.shape[1:]]).reshape([-1,*images_as_tensor.shape[1:]])

    labels_exp = torch.tensor(np.repeat(np.linspace(*data_labels,aging_steps,dtype=int)[:,None],number_of_images,1).T.reshape(-1))

    batch_size = 12

    out_tensor_exp = torch.cat([generate_synthetic_data(
    g_ema,
    mini_im,
    mini_label,
    global_blur_val=0.2, # CUSP global blur
    mask_blur_val=0.8)   # CUSP masked blur
    for mini_im, mini_label
    in zip(
        images_as_tensor_exp.split(batch_size),
        labels_exp.split(batch_size)
    )])

    out_tensor = out_tensor_exp.reshape([-1,aging_steps,*out_tensor_exp.shape[1:]])

    return out_tensor, images_as_tensor, labels_exp

'''
Method for converting tensor to uint8

Parameters:
    img_tensor () : tensor to be converted to uint8
Return:
    img_tensor (uint8) : converted tensor
'''
def to_uint8(img_tensor):
    img_tensor = (img_tensor.detach().cpu().numpy().transpose((1,2,0))+1)*(256/2)
    img_tensor = np.clip(img_tensor, 0, 255).astype(np.uint8)
    return img_tensor


'''
Method plots and saves plots of generated images

Parameters:
    batch_of_filenames () :
    img_in_tensor () :
    img_out_tensor () :
    labels_exp () :
    aging_steps () :

'''
def plot_output(batch_of_filenames, img_in_tensor, img_out_tensor, labels_exp, aging_steps):
    # For every input image
    counter = 1
    for fname, im_in, im_out, age_labels in zip(
            batch_of_filenames,img_in_tensor,img_out_tensor, 
            labels_exp.numpy().reshape(-1,aging_steps)
            ):
        # Create figure
        fig,axs = plt.subplots(1,aging_steps+1,figsize=(aging_steps*4,4),dpi=100)
        
        age_labels = ['Input'] + [f'Label "{i}"' for i in age_labels]
        # For every [input,step...]
        for ax,im,l in zip(axs,[im_in,*im_out],age_labels):
            ax.axis('off')
            ax.imshow(to_uint8(im))
            ax.set_title(l)


        plt.savefig(f"test_{counter}.png")
        counter = counter + 1

'''
Method generates synthetic face images of input images.
The method applies age editing to all input images, and generates new images with different ages of each identity.

Parameters:
    images_path (string) : path for the data which age editing should be applied to
Return:
    ---
'''
def run(images_path : str, aging_steps : int):

    FFHQ_LS_KEY = "lats"  # Model trained on LATS dataset
    FFHQ_RR_KEY = "hrfae" # Model trained on HRFAE dataset

    configs = {
    FFHQ_LS_KEY: dict(
        gdrive_id="1sWSH3tHgm9DkHrc19hoEMrR-KQgnaFuw",
        side=256, 
        classes=(0,70)),
    FFHQ_RR_KEY: dict(
        gdrive_id="17BOTEa6z3r6JFVs1KDutDxWEkTWbzaeD",
        side=224, 
        classes=(20,65))
    }

    batch_of_filenames = read_image_filenames(images_path)
    g_ema = load_cusp(torch.device('cuda', 0), 'data/cusp-network-ls.pkl')
    data_labels_range = configs[FFHQ_LS_KEY]['classes']
    side_config = configs[FFHQ_LS_KEY]['side']

    out_tensor, images_as_tensor, labels_exp = prep_data(side_config, batch_of_filenames, data_labels_range, g_ema)

    #plot_output(batch_of_filenames, images_as_tensor, out_tensor, labels_exp, aging_steps=4)
    generated_images_path = "datasets/cusp_generated_v2/"
    create_dataset(generated_images_path, batch_of_filenames, images_as_tensor, out_tensor, labels_exp, aging_steps)


'''
Method creates dataset of generated images, and saves the images to folder.

Parameters:
    synthetic_images_path (string) : 
    batch_of_filenames (string) : 
    img_in_tensor () : 
    img_out_tensor () : 
    labels_exp () : 
    aging_steps (int) : 

'''
def create_dataset(synthetic_images_path : str, batch_of_filenames, img_in_tensor, img_out_tensor, labels_exp, aging_steps):

    # create root folder for generated images
    os.makedirs(synthetic_images_path, exist_ok=True)

    # create one folder for each identity (00000, 00001, ...)

    for fname, im_in, im_out, age_labels in zip(
            batch_of_filenames,img_in_tensor,img_out_tensor, 
            labels_exp.numpy().reshape(-1,aging_steps)
            ):
        os.makedirs(synthetic_images_path + os.path.basename(fname)[:-4], exist_ok=True)
        fig,axs = plt.subplots(1,aging_steps+1,figsize=(aging_steps*4,4),dpi=100)
        
        age_labels = ['Input'] + [str(i) for i in age_labels]
        
        for ax,im,l in zip(axs,[im_in,*im_out],age_labels):
            if l != 'Input':
                img = PIL.Image.fromarray(to_uint8(im))
                path = synthetic_images_path + os.path.basename(fname)[:-4] + "/"+  os.path.basename(fname)[:-4] + "_"+ l + '.png'
                img.save(path)





run("models/cusp/synthetic_images/", 8)


