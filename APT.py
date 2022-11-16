
import json
import os
from time import perf_counter

import click
import imageio
import numpy as np
import torch
import torchvision.models as md

from utils import *
from inversion import *


@click.command()
@click.option('--data-location', help='path to train imagenet dataset', type=str, default='', show_default=True)
@click.option('--perceptor-path', help='path to classifier weights', type=str, default='ResNet50_ImageNet_PRIME_noJSD.ckpt', show_default=True)
@click.option('--seed', help='Random seed', type=int, default=42, show_default=True)
@click.option('--startidx', help='start img', type=int, default=0, required=True, show_default=True)
@click.option('--endidx', help='end img', type=int, default=1000, required=True, show_default=True)
@click.option('--save-video', help='Save an mp4 video of optimization progress', type=bool, default=False, show_default=False)
@click.option('--inv-steps', help='Number of inversion steps', type=int, default=1000, show_default=True)
@click.option('--run-pti', help='run pivotal tuning', default=True)
@click.option('--pti-steps', help='Number of pti steps', type=int, default=350, show_default=True)
def run_projection(
    data_location: str,
    perceptor_path: str,
    save_video: bool,
    seed: int,
    startidx: int,
    endidx: int,
    inv_steps: int,
    run_pti: bool,
    pti_steps: int,
):
    device = torch.device('cuda')
    perceptor = md.resnet50(pretrained=True)
    perceptor.eval().requires_grad_(False)
    base_model=torch.load(perceptor_path)
    perceptor.load_state_dict(base_model)
    disable_gradient_flow_for_model(perceptor,device)

    network_pkl = "https://s3.eu-central-1.amazonaws.com/avg-projects/stylegan_xl/models/imagenet256.pkl"
    generator, discriminator = prepare_styleganxl_generator_discriminator(network_pkl, device)

    #from timm.models import load_checkpoint
    #from fan import fan_base_18_p16_224
    #perceptor = fan_base_18_p16_224(pretrained=True)
    #load_checkpoint(perceptor, "/.../fan_vit_base.pth.tar")
    #disable_gradient_flow_for_model(perceptor,device)

    disable_gradient_flow_for_model(generator, device)
    disable_gradient_flow_for_model(discriminator, device)

    discriminator = remove_tf_diff_discriminator(discriminator)
    discriminator = discriminators_on_device(discriminator,device)

    pti_trainer = PTIOptimization(device,generator,discriminator,perceptor)
    pth, folders_considered = yield_individual_class_folder_with_imagenet_images(data_location,startidx,endidx)


    f= open("imagenet_class_index.json")
    idxs = json.load(f)
    inv_map = {v[0]: int(k) for k, v in idxs.items()}


    print(folders_considered)
    for folder in folders_considered:
        consider_this_folder=pth+"/"+folder+"/"
        ids = get_unique_imagenet_image_name_and_skip_generated_content(consider_this_folder)

        for file in ids:
            print(file)
            target_fname = consider_this_folder + folder + "_"+ file
            np.random.seed(seed)
            torch.manual_seed(seed)

            # Load target image.
            target_pil, target_uint8 = load_target_image(target_fname, generator.img_resolution)

            # Latent optimization
            start_time = perf_counter()
            all_images = []
            # path to precomputed latent variable
            w_init = False #target_fname + "_projected_w.npz"

            target_class = inv_map[folder]
            c_samples = torch.tensor(np.zeros([1, pti_trainer.CPP.c_dim], dtype=np.float32)).to(device)
            c_samples[:,target_class]=1
        
            if not w_init:
                print('Running Initial Latent Optimization...')
                all_images, projected_w = pti_trainer.LO.project(
                target_class=target_class,
                target=torch.tensor(target_uint8.transpose([2, 0, 1]), device=device), # pylint: disable=not-callable
                num_steps=inv_steps,
                verbose=True,
                noise_mode='const',
                )
                print(f'Elapsed time: {(perf_counter()-start_time):.1f} s')
            else:
                projected_w = torch.from_numpy(np.load(w_init)['w'])[0].to(device)
                
            np.savez(f'{target_fname}_projected_w.npz', w=projected_w.unsqueeze(0).cpu().numpy())
            synthesize_and_save_image(generator, projected_w, f'{target_fname}_proj.png')
            
            start_time = perf_counter()
            # Run PTI
            if run_pti:
                print('Running Adversarial Pivotal Tuning...')
                gen_images, G = pti_trainer.pivotal_tuning(
                projected_w,
                target_class=c_samples,
                target=torch.tensor(target_uint8.transpose([2, 0, 1]), device=device),
                num_steps=pti_steps,
                learning_rate = 3e-4,
                noise_mode="const",
                verbose=True,
                )
                all_images += gen_images
                print(f'Elapsed time: {(perf_counter()-start_time):.1f} s')
                

            # Render debug output: optional video and projected image and W vector.
            os.makedirs(consider_this_folder, exist_ok=True)
            if save_video:
                video = imageio.get_writer(f'{consider_this_folder}/proj_pti.mp4', mode='I', fps=60, codec='libx264', bitrate='16M')
                print (f'Saving optimization progress video "{consider_this_folder}/proj.mp4"')
                for synth_image in all_images:
                    video.append_data(np.concatenate([target_uint8, synth_image], axis=1))
                video.close()

            # Save final projected frame and W vector.
            target_pil.save(f'{target_fname}_target_pti.png')
            if run_pti:
                synthesize_and_save_image(G, projected_w, f'{target_fname}_proj_pti.png')

            # save latents
            np.savez(f'{target_fname}_projected_w_pti.npz', w=projected_w.unsqueeze(0).cpu().numpy())

            # Save Generator weights
            # snapshot_data = {'G': G, 'G_ema': G}
            # with open(f"{consider_this_folder}/G.pkl", 'wb') as f:
            #    dill.dump(snapshot_data, f)

            #----------------------------------------------------------------------------


if __name__ == "__main__":

    run_projection()

