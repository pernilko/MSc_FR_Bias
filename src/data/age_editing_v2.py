import torch
import numpy as np
import os
import PIL.Image
from models.eg3dAge.eg3d.camera_utils import LookAtPoseSampler, FOV_to_intrinsics
from models.eg3dAge.eg3d import dnnlib
from models.eg3dAge.eg3d import legacy


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
        img = np.array(PIL.Image.open(file).resize((side, side)), dtype=np.float32).transpose((2,0,1))
        images.append(img)
    return images

def read_image_filenames(images_path : str):
    batch_of_filenames = [
      os.path.join(images_path,f) 
      for  f in next(iter(os.walk(images_path)))[2] 
      if f[-4:] == '.png'
    ]
    return batch_of_filenames


def age_editing_e(device : torch.device, network_pkl, input_images_path : str, truncation_psi : float, truncation_cutoff : float, outdir : str):

    print("starting age editing with e3gd")
    # Load pre-trained model and input images
    G = load_model(device, network_pkl)

    batch_of_filenames = read_image_filenames(input_images_path)
    input_images = get_images(batch_of_filenames,256)
    inp_images_tensor = (torch.tensor(np.array(input_images))/256*2-1)

    for img_tensor in inp_images_tensor:

        z = torch.from_numpy(np.random.RandomState(1).randn(1, G.z_dim)).to(device)

        fov_deg = 18.837
        intrinsics = FOV_to_intrinsics(fov_deg, device=device)
        cam_pivot = torch.tensor(G.rendering_kwargs.get('avg_camera_pivot', [0,0,0]), device=device)
        cam_radius = G.rendering_kwargs.get('avg_camera_radius', 2.7)
        cam2world_pose = LookAtPoseSampler.sample(np.pi/2 + 0, np.pi/2 + 0, cam_pivot, radius=cam_radius, device=device)
        conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi/2, np.pi/2, cam_pivot, radius=cam_radius, device=device)
        camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
        conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)    

        age = 2
        age = [normalize(age, rmin=0, rmax=100)]
        c = torch.cat((conditioning_params, torch.tensor([age], device=device)), 1)
        c_params = torch.cat((camera_params, torch.tensor([age], device=device)), 1)
        ws = G.mapping(z, c.float(), truncation_psi=1, truncation_cutoff=0)
        img = G.synthesis(ws, c_params.float())['image']
        img.permute((0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        pil_img = PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB')

        pil_img.save("test_img_e3gd.png")
        
        return
    


device = torch.device('cuda:0')
network_pkl = "data/eg3d_age_network.pkl"
images_input_path = "models/cusp/synthetic_images/"
truncation_psi = 0.2
truncation_cutoff = 0.8
outdir = "datasets/eg3d_generated/"

age_editing_e(device, network_pkl, images_input_path, truncation_psi, truncation_cutoff, outdir)



