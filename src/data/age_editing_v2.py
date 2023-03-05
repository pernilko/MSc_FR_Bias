import torch
import numpy as np
import os
import PIL.Image
from models.eg3dAge.eg3d.camera_utils import LookAtPoseSampler, FOV_to_intrinsics
from models.eg3dAge.eg3d import dnnlib
from models.eg3dAge.eg3d import legacy
import matplotlib.pyplot as plt


def normalize(x, rmin = 0, rmax = 75, tmin = -1, tmax = 1):
    # normalize age between -1 and 1
    z = ((x - rmin) / (rmax - rmin)) * (tmax - tmin) + tmin
    return z

def load_model(device : torch.device, network_pkl):
    device = torch.device('cuda:0')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)

    return G

def create_dataset(outdir : str):
    os.makedirs(outdir, exist_ok=True)

def get_images(batch_of_filenames, side : int):
    images = []
    for file in batch_of_filenames:
        img = PIL.Image.open(file).convert('L')
        img2 = PIL.Image.open(file)
        #print("rgb img: ", img2.size)
        #print("grey scale: ", img.size)
        img = np.array(img.resize((side, side)), dtype=np.float32)
        img2 = np.array(img2.resize((side, side)), dtype=np.float32)
        #print("grey scale: ", img.shape)
        #print("rgb: ", img2.shape)
        #img = img.transpose((2,0,1))
        images.append(img2)
    return images

def read_image_filenames(images_path : str):
    batch_of_filenames = [
      os.path.join(images_path,f) 
      for  f in next(iter(os.walk(images_path)))[2] 
      if f[-4:] == '.png'
    ]
    return batch_of_filenames

def plot_output(identity_imgs, ages, identity_name, out_plot_dir):
    
    # For every input image
    for img in identity_imgs:
        fig,axs = plt.subplots(1, len(identity_imgs), figsize=(len(identity_imgs)*4,4), dpi=100)
        age_labels = [f'Label "{str(i)}"' for i in ages]
        for ax,im,l in zip(axs, img, age_labels):
            ax.axis('off')
            ax.imshow(im)
            ax.set_title(l)

        os.makedirs(out_plot_dir, exist_ok=True)
        plt.savefig(f"{out_plot_dir}/{identity_name}.png")


def age_editing_e(device : torch.device, network_pkl, input_images_path : str, truncation_psi : float, truncation_cutoff : float, outdir : str, seeds : list):

    print("starting age editing with e3gd")
    # Load pre-trained model and input images
    G = load_model(device, network_pkl)
    os.makedirs(outdir, exist_ok=True)

    #batch_of_filenames = read_image_filenames(input_images_path)
    #input_images = get_images(batch_of_filenames,256)
    #inp_images_tensor = (torch.tensor(np.array(input_images))/256*2-1)

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

        ages = [0, 6, 11, 16, 21, 31, 41, 51, 61, 71]
        imgs = []
        for age in ages:
            original_age = age
            age = [normalize(age, rmin=0, rmax=100)]
            c = torch.cat((conditioning_params, torch.tensor([age], device=device)), 1)
            c_params = torch.cat((camera_params, torch.tensor([age], device=device)), 1)
            ws = G.mapping(z, c.float(), truncation_psi=0.5, truncation_cutoff=0)
            img = G.synthesis(ws, c_params.float())['image']
            img = img.permute(0, 2, 3, 1) * 127.5 + 128
            img = img.clamp(0, 255).to(torch.uint8)
            imgs.append(img)
            pil_img = PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB')

            os.makedirs(f"{outdir}/seed{seed:04d}/", exist_ok=True)
            pil_img.save(f"{outdir}/seed{seed:04d}/seed{seed:04d}_{original_age}.png")
        
        if counter < 5:
            plot_output(imgs, ages, f"{seed:04d}", "synthetic_img_plots/")
        counter = counter + 1
    


device = torch.device('cuda:0')
network_pkl = "data/eg3d_age_network.pkl"
images_input_path = "models/cusp/synthetic_images/"
truncation_psi = 0.2
truncation_cutoff = 0.8
outdir = "datasets/eg3d_generated/"
seeds = list(range(301))

age_editing_e(device, network_pkl, images_input_path, truncation_psi, truncation_cutoff, outdir, seeds)



