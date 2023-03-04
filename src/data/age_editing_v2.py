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
        img = np.array(PIL.Image.open(file).resize((side, side)).convert('L'), dtype=np.float32)
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
    print(G)

    batch_of_filenames = read_image_filenames(input_images_path)
    input_images = get_images(batch_of_filenames,128)
    inp_images_tensor = (torch.tensor(np.array(input_images))/256*2-1)

    for img_tensor in inp_images_tensor:

        #img_tensor = img_tensor.resize((128, 128))
        # normalize image to have values between -1 and 1
        img_tensor = (np.array(img_tensor) / 127.5) - 1.0

        print("img tensor: ", img_tensor)
        z = img_tensor.to(device)
        print(z.ndim)
        print(G.z_dim)
        fov_deg = 18.837
        intrinsics = FOV_to_intrinsics(fov_deg, device=device)

        imgs = []
        angle_p = -0.2
        for angle_y, angle_p in [(.4, angle_p), (0, angle_p), (-.4, angle_p)]:
            cam_pivot = torch.tensor(G.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0]), device=device)
            cam_radius = G.rendering_kwargs.get('avg_camera_radius', 2.7)
            cam2world_pose = LookAtPoseSampler.sample(np.pi/2 + angle_y, np.pi/2 + angle_p, cam_pivot, radius=cam_radius, device=device)
            conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi/2, np.pi/2, cam_pivot, radius=cam_radius, device=device)
            camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
            conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

            ws = G.mapping(z, conditioning_params, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
            img = G.synthesis(ws, camera_params)['image']

            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            imgs.append(img)

        age = 2
        age = [normalize(age, rmin=0, rmax=100)]
        print("age: ", age)
        c = torch.cat((conditioning_params, torch.tensor([age], device=device)), 1)
        c_params = torch.cat((camera_params, torch.tensor([age], device=device)), 1)
        ws = G.mapping(z, c.float(), truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
        img = G.synthesis(ws, c_params.float())['image']
        img.permute((0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)

        os.makedirs(outdir, exist_ok=True)
        pil_img = PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/seed_test.png')
        return
    


device = torch.device('cuda:0')
network_pkl = "data/eg3d_age_network.pkl"
images_input_path = "models/cusp/synthetic_images/"
truncation_psi = 0.2
truncation_cutoff = 0.8
outdir = "datasets/eg3d_generated/"

age_editing_e(device, network_pkl, images_input_path, truncation_psi, truncation_cutoff, outdir)



