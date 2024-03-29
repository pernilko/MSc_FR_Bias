import os
import time
import pickle
import torch
import torch.nn.functional as F
import PIL.Image
import numpy as np
import matplotlib.pyplot as plt
import sys
from models.cusp import legacy
from models.cusp.torch_utils import persistence
import models.cusp.training
from models.cusp import dnnlib
from models.cusp.training.networks import VGG, module_no_grad
from models.cusp.torch_utils import misc
import random
from models.cusp.generate import generate_images
import argparse


'''
Method for reading the filenames of the input images to be subjected to age editing

Parameters:
    images_path (str) : path to images
Return:
    batch_of_filenames (list) : filenames
'''
def read_image_filenames(images_path : str):
    batch_of_filenames = [
      os.path.join(images_path,f) 
      for  f in next(iter(os.walk(images_path)))[2] 
      if f[-4:] == '.png'
    ]
    return batch_of_filenames

'''
Method for loading the pre-trained CUSP model
Parameters:
    device (torch.device) : device to be used load the model
    weights_path (str) : path to model weights
Return:
    g_ema
'''
def load_cusp(device : torch.device, weights_path : str):
    print("weights path: ", weights_path)

    #weights_path = 'data/cusp_network.pkl'
    vgg_path = "data/dex_imdb_wiki.caffemodel.pt"

    
    with open(weights_path, 'rb') as f:
        g_ema = legacy.load_network_pkl(f)['G_ema'].to(device)
    
    #g_ema = data['G_ema'] # exponential movign average model

    vgg = VGG()
    vgg_state_dict = torch.load(vgg_path)
    vgg_state_dict = {k.replace('-', '_'): v for k, v in vgg_state_dict.items()}
    vgg.load_state_dict(vgg_state_dict)
    module_no_grad(vgg) 

    g_ema.skip_grad_blur.model.classifier = vgg

    g_ema = g_ema.to(device).eval().requires_grad_(False)

    return g_ema

'''
Method for generating synthetic face images using CUSP

Parameters:
    G
    img
    label
    global_blur_val
    mask_blur_val
    return_msk (bool)
Return:
    to_return
'''
def generate_synthetic_data(G, img, label, global_blur_val=None, mask_blur_val=None, return_msk=False):
    ohe_label = torch.nn.functional.one_hot(torch.tensor(label), num_classes=G.attr_map.fc0.init_args[0]).to(img.device)
    c_out, c_out_skip = G.content_enc(img)
    
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
def prep_data(side : int, batch_of_filenames, data_labels, g_ema, aging_steps : int):
    images = []

    for file in batch_of_filenames:
        img = np.array(PIL.Image.open(file).resize((side, side)), dtype=np.float32).transpose((2,0,1))
        images.append(img)

    images_as_tensor = (torch.tensor(np.array(images))/256*2-1).cuda()

    #aging_steps = 8

    number_of_images = images_as_tensor.shape[0]
    images_as_tensor_exp = images_as_tensor[:, None].expand([number_of_images, aging_steps, *images_as_tensor.shape[1:]]).reshape([-1,*images_as_tensor.shape[1:]])
    #print("shape tensor exp: ", images_as_tensor_exp.shape)

    #print(np.array(data_labels))
    #labels_exp = torch.tensor(np.repeat(np.array(data_labels, dtype=int)[:,None],number_of_images,1).T.reshape(-1))
    labels_exp = torch.tensor(np.repeat(np.linspace(*data_labels,aging_steps,dtype=int)[:,None],number_of_images,1).T.reshape(-1))


    #print("labels exp: ",labels_exp)
    batch_size = 8

    out_tensor_exp = torch.cat([generate_synthetic_data(
    g_ema,
    mini_im,
    mini_label,
    global_blur_val=0.1, # CUSP global blur
    mask_blur_val=0.3)   # CUSP masked blur
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
    batch_of_filenames (list) : filenames
    img_in_tensor () :
    img_out_tensor () :
    labels_exp () :
    aging_steps (int) : number of aging steps

'''
def plot_output(batch_of_filenames, img_in_tensor, img_out_tensor, labels_exp, aging_steps):
    # For every input image
    os.makedirs("synthetic_img_plots/cusp/", exist_ok=True)
    counter = 1
    for fname, im_in, im_out, age_labels in zip(
            batch_of_filenames,img_in_tensor,img_out_tensor, 
            labels_exp.numpy().reshape(-1,aging_steps)
            ):
        if counter > 10:
            return
        # Create figure
        fig,axs = plt.subplots(1,aging_steps+1,figsize=(aging_steps*4,4),dpi=100)
        
        age_labels = ['Input'] + [f'Label "{str(age_cluster_get_random_age(int(i)))}"' for i in age_labels]
        # For every [input,step...]
        for ax,im,l in zip(axs,[im_in,*im_out],age_labels):
            ax.axis('off')
            ax.imshow(to_uint8(im))
            ax.set_title(l)

        plt.savefig(f"synthetic_img_plots/cusp/{os.path.basename(fname)[:-4]}.png")
        counter = counter + 1



'''
Method for getting a random age from a age bin

Parameters:
    age_bin (int) : the age bin to get a random age from
Return:
    random age (int) : random age from specified age bin
'''
def get_random_age(age_bin : int):
    if age_bin == 0:
        return random.randint(0, 5)
    elif age_bin == 1:
        return random.randint(5, 10)
    elif age_bin == 2:
        return random.randint(10, 15)
    elif age_bin == 3:
        return random.randint(15, 20)
    elif age_bin == 4:
        return random.randint(20, 30)
    elif age_bin == 5:
        return random.randint(30, 40)
    elif age_bin == 6:
        return random.randint(40, 50)
    else:
        return random.randint(50, 70)

'''
Method generates synthetic face images of input images.
The method applies age editing to all input images, and 
generates new images with different ages of each identity.

Parameters:
    images_path (str) : path to dataset to apply age editing to
    output_images_path (str) : path to where to save the generated images
    weights_path_ls (str) : path to model weights of the CUSP Lifespan model
    weights_path_rr (str) : path to model weights of the CUSP Restricted Range model
    device (torch.device) : device to use to load model
Return:
    None
'''
def run(images_path : str, output_images_path : str, weights_path_ls : str,  weights_path_rr : str, device : torch.device):

    FFHQ_LS_KEY = "lats"  # Model trained on LATS dataset
    FFHQ_RR_KEY = "hrfae" # Model trained on HRFAE dataset

    configs = {
    FFHQ_LS_KEY: dict(
        gdrive_id="1sWSH3tHgm9DkHrc19hoEMrR-KQgnaFuw",
        side=256, 
        classes=(1,8)),
    FFHQ_RR_KEY: dict(
        gdrive_id="17BOTEa6z3r6JFVs1KDutDxWEkTWbzaeD",
        side=224,
        classes=(20,65))
    }

    side_config_rr = configs[FFHQ_RR_KEY]['side']
    side_config_ls = configs[FFHQ_LS_KEY]['side']

    batch_of_filenames = read_image_filenames(images_path)

    age_range = (1,9)

    # LS
    g_ema_ls = load_cusp(device, weights_path_ls)
    aging_steps_ls =  8
    out_tensor_ls, images_as_tensor_ls, labels_exp_ls = prep_data(side_config_ls, batch_of_filenames, age_range, g_ema_ls, aging_steps_ls)
    create_dataset(output_images_path, batch_of_filenames, images_as_tensor_ls, out_tensor_ls, labels_exp_ls, aging_steps_ls)
    plot_output(batch_of_filenames,images_as_tensor_ls,out_tensor_ls,labels_exp_ls,aging_steps_ls)
    
    '''
    # RR
    g_ema_rr = load_cusp(device, weights_path_rr)
    aging_steps_rr = 4
    out_tensor_rr, images_as_tensor_rr, labels_exp_rr = prep_data(side_config_rr, batch_of_filenames, age_labels_rr, g_ema_rr, aging_steps_rr)
    create_dataset(output_images_path, batch_of_filenames, images_as_tensor_rr, out_tensor_rr, labels_exp_rr, aging_steps_rr)

    #plot_output(batch_of_filenames, images_as_tensor, out_tensor, labels_exp, aging_steps=4)
    #create_dataset(output_images_path, batch_of_filenames, images_as_tensor, out_tensor, labels_exp, aging_steps)
    '''

'''
Method for getting random age

Parameters:
    class_idx (int)
Return:
    age (int)
'''
def age_cluster_get_random_age(class_idx : int):
    if class_idx == 1 or class_idx == 2:
        age = get_random_age(0)
    elif class_idx == 2 or class_idx == 3:
        age = get_random_age(1)
    elif class_idx == 4:
        age = get_random_age(2)
    elif class_idx == 5:
        age = get_random_age(3)
    elif class_idx == 6:
        age = get_random_age(4)
    elif class_idx == 7:
        age = get_random_age(5)
    elif class_idx == 8:
        age = get_random_age(6)
    elif class_idx == 9:
        age = get_random_age(7)

    return age
    

'''
Method creates dataset of generated images, and saves the images to folder.

Parameters:
    synthetic_images_path (str) : 
    batch_of_filenames (list) : 
    img_in_tensor () : 
    img_out_tensor () : 
    labels_exp () : 
    aging_steps (int) : 
Return:
    None

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
        age_labels = ['Input'] + [str(i) for i in age_labels]
        
        for im,l in zip([im_in,*im_out],age_labels):
            if l != 'Input':
                img = PIL.Image.fromarray(to_uint8(im))
                path = synthetic_images_path + os.path.basename(fname)[:-4] + "/"+  os.path.basename(fname)[:-4] + "_"+ str(age_cluster_get_random_age(int(l))) + '.png'
                img.save(path)




'''
Generating synthetic face images with different ages using the CUSP model
'''

weights_path_rr = 'data/cusp-network.pkl' 
weights_path_ls = 'data/cusp-network-ls.pkl' #data/cusp-network-ls.pkl # LifeSpan 
vgg_path = "data/dex_imdb_wiki.caffemodel.pt"
input_images_path = "models/cusp/synthetic_images/"
output_images_path = "datasets/cusp_generated_v2/"
aging_steps = 4
device = torch.device('cuda', 0)

parser = argparse.ArgumentParser()
parser.add_argument('--input_img_path', type=str, default=input_images_path, help="path to input images")
parser.add_argument('--out_img_path', type=str, default=input_images_path, help="path to output images")
args = parser.parse_args()
print("inp images: ", args.input_img_path)
print("out path: ", args.out_img_path)

run(args.input_img_path, aging_steps, args.out_img_path, weights_path_ls, weights_path_rr, device)


