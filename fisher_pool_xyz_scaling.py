#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
from fisher_renderer import fisher_render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel

from torchvision.utils import save_image
import copy
import math

def pool_fisher_python(view_idx, view, gaussians, pipeline, background,
                fishers, resolution):

    image = render(view, gaussians, pipeline, background)['render']

    o_grads = torch.autograd.grad(
        image.sum(), gaussians._opacity, retain_graph=True)[0]
    _filter = o_grads.sum(dim=1) != 0

    image_gaussians = copy.deepcopy(gaussians)
    image_gaussians._xyz.data = gaussians._xyz.data[_filter]
    image_gaussians._rotation.data = gaussians._rotation.data[_filter]
    image_gaussians._scaling.data = gaussians._scaling.data[_filter]
    image_gaussians._features_dc.data = gaussians._features_dc.data[_filter]
    image_gaussians._features_rest.data = gaussians._features_rest.data[_filter]
    image_gaussians._opacity.data = gaussians._opacity.data[_filter]

    image = render(view, image_gaussians, pipeline, background)['render']

    cs, ys, xs = image.shape
    y_idxs, x_idxs = ys//resolution, xs//resolution

    progress_bar = tqdm(range(y_idxs*x_idxs*3), desc="patches")
    progress_bar.set_postfix({"Image": f"{view_idx}"})
    for c in range(3):
        for y_idx in range(y_idxs):
            for x_idx in range(x_idxs):
                x1, y1 = x_idx*resolution, y_idx*resolution
                x2, y2 = x1 + resolution, y1 + resolution

                grads = torch.autograd.grad(
                    image[c, y1:y2, x1:x2].sum(),
                    (image_gaussians._xyz, image_gaussians._scaling),
                    retain_graph=True)

                grads = torch.cat(grads, dim=1)
                patch_fisher = grads[:, None, :] * grads[:, :, None]
                fishers[_filter] += patch_fisher.detach()

                progress_bar.update(1)

    return fishers


def pool_fisher_cuda(view_idx, view, gaussians, pipeline, background,
                fishers, resolution):

    sym_fishers = torch.zeros(
        (fishers.shape[0], 21), dtype=fishers.dtype, device=fishers.device)
    sym_fishers.requires_grad = True

    # set patch resolution as view downsample
    view_copy = copy.deepcopy(view)
    view_copy.image_height = math.ceil(view.image_height / resolution)
    view_copy.image_width = math.ceil(view.image_width / resolution)

    image = fisher_render(view_copy, gaussians, pipeline, background,
                          sym_fishers)['render']

    # Backward computes and stores symetric fisher values
    # in sym_fisher's grad
    image_sum = image.sum()
    image_sum.backward()

    fishers[:, 0, 0] += sym_fishers.grad[:, 0]
    fishers[:, 0, 1] += sym_fishers.grad[:, 1]
    fishers[:, 0, 2] += sym_fishers.grad[:, 2]
    fishers[:, 0, 3] += sym_fishers.grad[:, 3]
    fishers[:, 0, 4] += sym_fishers.grad[:, 4]
    fishers[:, 0, 5] += sym_fishers.grad[:, 5]

    fishers[:, 1, 0] += sym_fishers.grad[:, 1]
    fishers[:, 1, 1] += sym_fishers.grad[:, 6]
    fishers[:, 1, 2] += sym_fishers.grad[:, 7]
    fishers[:, 1, 3] += sym_fishers.grad[:, 8]
    fishers[:, 1, 4] += sym_fishers.grad[:, 9]
    fishers[:, 1, 5] += sym_fishers.grad[:, 10]

    fishers[:, 2, 0] += sym_fishers.grad[:, 2]
    fishers[:, 2, 1] += sym_fishers.grad[:, 7]
    fishers[:, 2, 2] += sym_fishers.grad[:, 11]
    fishers[:, 2, 3] += sym_fishers.grad[:, 12]
    fishers[:, 2, 4] += sym_fishers.grad[:, 13]
    fishers[:, 2, 5] += sym_fishers.grad[:, 14]

    fishers[:, 3, 0] += sym_fishers.grad[:, 3]
    fishers[:, 3, 1] += sym_fishers.grad[:, 8]
    fishers[:, 3, 2] += sym_fishers.grad[:, 12]
    fishers[:, 3, 3] += sym_fishers.grad[:, 15]
    fishers[:, 3, 4] += sym_fishers.grad[:, 16]
    fishers[:, 3, 5] += sym_fishers.grad[:, 17]

    fishers[:, 4, 0] += sym_fishers.grad[:, 4]
    fishers[:, 4, 1] += sym_fishers.grad[:, 9]
    fishers[:, 4, 2] += sym_fishers.grad[:, 13]
    fishers[:, 4, 3] += sym_fishers.grad[:, 16]
    fishers[:, 4, 4] += sym_fishers.grad[:, 18]
    fishers[:, 4, 5] += sym_fishers.grad[:, 19]

    fishers[:, 5, 0] += sym_fishers.grad[:, 5]
    fishers[:, 5, 1] += sym_fishers.grad[:, 10]
    fishers[:, 5, 2] += sym_fishers.grad[:, 14]
    fishers[:, 5, 3] += sym_fishers.grad[:, 17]
    fishers[:, 5, 4] += sym_fishers.grad[:, 19]
    fishers[:, 5, 5] += sym_fishers.grad[:, 20]

    del sym_fishers

    return fishers


def run(dataset : ModelParams, iteration : int, pipeline : PipelineParams,
        resolution : int, fisher_via_cuda : bool):

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    output_path = os.path.join(dataset.model_path, f'fisher-pool-{iteration}')
    N = gaussians.get_xyz.shape[0]
    device = gaussians.get_xyz.device
    fishers = torch.zeros(N,6,6,device=device).float()

    fisher_path = os.path.join(output_path, "fishers_xyz_scaling")
    makedirs(fisher_path, exist_ok=True)
    pool_fisher_func = pool_fisher_cuda if fisher_via_cuda else pool_fisher_python

    for view_idx, view in tqdm(enumerate(scene.getTrainCameras()), desc='Computing Fishers'):

        pool_fisher_func(
            view_idx, view, gaussians, pipeline, background,
            fishers, resolution)

    save_file = os.path.join(fisher_path, f'fisher_resolution_{resolution}.pt')
    torch.save(fishers, save_file)
    print(f'Fishers saved at: {save_file}')


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Fishers script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--pool-resolution", default=1, type=int)
    parser.add_argument("--fisher-via-cuda", action="store_true")

    args = get_combined_args(parser)
    print("Computing Fishers " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    run(model.extract(args), args.iteration, pipeline.extract(args),
        args.pool_resolution, args.fisher_via_cuda)
