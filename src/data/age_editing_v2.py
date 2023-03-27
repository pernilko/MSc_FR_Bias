import torch
import numpy as np
import os
import PIL.Image
from models.eg3dAge.eg3d.camera_utils import LookAtPoseSampler, FOV_to_intrinsics
from models.eg3dAge.eg3d import dnnlib
from models.eg3dAge.eg3d import legacy
import matplotlib.pyplot as plt
import random

'''
Method for normalizing age between -1 and 1

Parameters:
    x () : 
    rmin () :
    rmax () :
    tmin () : 
    tmax () : 
Return:
    z () : normalized age
'''
def normalize(x, rmin = 0, rmax = 75, tmin = -1, tmax = 1):
    z = ((x - rmin) / (rmax - rmin)) * (tmax - tmin) + tmin
    return z

'''
Method for loading pre-trained model

Parameters:
    device (torch.device) : the device to be used. Cuda or CPU.
    network_pkl (str) : the path to the model
Return:
    G (Generator) : the generator of the GAN
'''
def load_model(device : torch.device, network_pkl : str):
    device = torch.device('cuda:0')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)

    return G

'''
Method for getting a random age

Parameters:
    age_bin (int) : the number of the age bin from where one wants a ranom number
Return:
    random number (int) : the random age in the age bin selected 
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
Method for getting 
'''
def get_random_age_list(number_of_ages : int):
    ages = []
    for i in range(number_of_ages):
        age = get_random_age(i)
        ages.append(age)
    
    return ages

'''
Method for plotting images

Parameters:
    identity_imgs () : 
    ages (list) : 
    identity_name () : 
    out_plot_dir (str) : the path where the images should be saved
Return:
    None.
'''
def plot_output(identity_imgs, ages, identity_name, out_plot_dir : str):
    # For every input image
    
    fig,axs = plt.subplots(1, len(identity_imgs), figsize=(len(identity_imgs)*4,4), dpi=100)
    age_labels = [f'Label "{str(i)}"' for i in ages]
    for ax, img, l in zip(axs, identity_imgs, age_labels):
        ax.axis('off')
        ax.imshow(img)
        ax.set_title(l)

    os.makedirs(out_plot_dir, exist_ok=True)
    plt.savefig(f"{out_plot_dir}/{identity_name}.png")

'''
Method for convering a tensor to uint8

Parameters:
    img_tensor (tensor) : the tensor to be converted
Return:
    img_tensor (uint8) : the converted tensor
'''
def to_uint8(img_tensor):
    img_tensor = (img_tensor.detach().cpu().numpy())
    img_tensor = np.clip(img_tensor, 0, 255).astype(np.uint8)
    return img_tensor

'''
Method for generating synthetic age editied face images using eg3d

Parameters:
    device (torch.device) : the device to be used. Cuda or CPU.
    network_pkl (str) : the path to the pre-trained model
    truncation_psi (float) : 
    truncation_cutoff (float) : 
    outdir (str) : the path where the generated images should be saved
    seeds (list) : the seeds to be used in generation
Return:
    None.


'''
def age_editing_eg3d(device : torch.device, network_pkl : str,  outdir : str, seeds : list, truncation_psi : float = 0.5, truncation_cutoff : float = 0):

    print("starting age editing with eg3d")
    # Load pre-trained model and input images
    G = load_model(device, network_pkl)
    os.makedirs(outdir, exist_ok=True)

    counter = 0
    for seed in seeds:

        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)

        fov_deg = 18.837
        intrinsics = FOV_to_intrinsics(fov_deg, device=device)
        cam_pivot = torch.tensor(G.rendering_kwargs.get('avg_camera_pivot', [0,0,0]), device=device)
        cam_radius = G.rendering_kwargs.get('avg_camera_radius', 2.7)
        cam2world_pose = LookAtPoseSampler.sample(np.pi/2 + 0, np.pi/2 + 0, cam_pivot, radius=cam_radius, device=device)
        conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi/2, np.pi/2, cam_pivot, radius=cam_radius, device=device)
        camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
        conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)    

        #ages = [0, 6, 11, 16, 21, 31, 41, 51, 61, 71]
        ages = get_random_age_list(8)
        imgs = []
        for age in ages:
            original_age = age
            age = [normalize(age, rmin=0, rmax=100)]
            c = torch.cat((conditioning_params, torch.tensor([age], device=device)), 1)
            c_params = torch.cat((camera_params, torch.tensor([age], device=device)), 1)
            ws = G.mapping(z, c.float(), truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
            img = G.synthesis(ws, c_params.float())['image']
            img = img.permute(0, 2, 3, 1) * 127.5 + 128
            img = img.clamp(0, 255).to(torch.uint8)
            pil_img = PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB')
            imgs.append(pil_img)

            os.makedirs(f"{outdir}/seed{seed:04d}/", exist_ok=True)
            pil_img.save(f"{outdir}/seed{seed:04d}/seed{seed:04d}_{original_age}.png")
        
        if counter < 5:
            plot_output(imgs, ages, f"seed{seed:04d}", "synthetic_img_plots/")
        counter = counter + 1
    

'''
Running Age-EG3D generation
'''

# Defining default parameters
device = torch.device('cuda:0')
network_pkl = "data/eg3d_age_network.pkl"
images_input_path = "models/cusp/synthetic_images/"
truncation_psi = 0.5
truncation_cutoff = 0
outdir = "datasets/eg3d_generated/"
seeds = list(range(1400))

# Running Age-EG3D generation
age_editing_eg3d(device, network_pkl, outdir, seeds, truncation_psi, truncation_cutoff)
