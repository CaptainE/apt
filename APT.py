
import json
import os
from time import perf_counter

import click
import imageio
import numpy as np
import torch

from utils import *
from inversion import *


@click.command()
@click.option('--data-location', help='path to train imagenet dataset', type=str, default='', show_default=True)
@click.option('--perceptor', help='classification model to be used, default is prime resnet50', type=object, default=None, show_default=True)
@click.option('--seed', help='Random seed', type=int, default=42, show_default=True)
@click.option('--startidx', help='start img', type=int, default=0, required=True, show_default=True)
@click.option('--endidx', help='end img', type=int, default=1000, required=True, show_default=True)
@click.option('--save-video', help='Save an mp4 video of optimization progress', type=bool, default=False, show_default=False)
@click.option('--inv-steps', help='Number of inversion steps', type=int, default=1000, show_default=True)
@click.option('--run-pti', help='run pivotal tuning', default=True)
@click.option('--pti-steps', help='Number of pti steps', type=int, default=350, show_default=True)
@click.option('--device', help='Random seed', type=str, default='cuda', show_default=True)
def run_projection(
    data_location: str,
    perceptor: object,
    save_video: bool,
    seed: int,
    startidx: int,
    endidx: int,
    inv_steps: int,
    run_pti: bool,
    pti_steps: int,
    device: str
):
    device = torch.device(device)

    perceptor = prepare_classifier(perceptor,device)
    generator, discriminator = prepare_discriminator_generator(device)

    pti_trainer = PTIOptimization(device,generator,discriminator,perceptor)
    pth, folders_considered = yield_individual_class_folder_with_imagenet_images(data_location,startidx,endidx)

    inv_map = get_imagenet_classname_to_class_mapping()


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
            # path to precomputed latent variable if exist
            w_init, target_class, c_samples = prepare_init_latent_optimization_input(target_fname, inv_map, folder, pti_trainer.CPP.c_dim, device)

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


if __name__ == "__main__":

    run_projection()

