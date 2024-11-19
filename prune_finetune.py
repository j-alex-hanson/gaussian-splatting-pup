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

import os
import torch
from utils.loss_utils import l1_loss, ssim
from lpipsPyTorch import lpips
from gaussian_renderer import render, network_gui
import sys
from random import randint
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import numpy as np
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
import gc
from os import makedirs
from torch.optim.lr_scheduler import ExponentialLR
from utils.logger_utils import training_report, prepare_output_and_logger

from fisher_pool_xyz_scaling import pool_fisher_cuda
from prune import prune_list, calculate_v_imp_score

def training(
    dataset,
    opt,
    pipe,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    debug_from,
    args,
):
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    if checkpoint:
        gaussians.training_setup(opt)
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
    elif args.start_pointcloud:
        gaussians.load_ply(args.start_pointcloud)
        first_iter = int(args.start_pointcloud.split('/')[-2].split('_')[-1]) 
        gaussians.training_setup(opt)
        gaussians.max_radii2D = torch.zeros((gaussians.get_xyz.shape[0]), device="cuda")
    else:
        raise ValueError("A checkpoint file or a pointcloud is required to proceed.")

    print(f"\nInitial Model: Number of Gaussians is {len(gaussians.get_xyz)}")
    prune_idx = 0

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    lr_iter = first_iter
    gaussians.scheduler = ExponentialLR(gaussians.optimizer, gamma=0.95)    

    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                (
                    custom_cam,
                    do_training,
                    pipe.convert_SHs_python,
                    pipe.compute_cov3D_python,
                    keep_alive,
                    scaling_modifer,
                ) = network_gui.receive()
                if custom_cam != None:
                    net_image = render(
                        custom_cam, gaussians, pipe, background, scaling_modifer
                    )["render"]
                    net_image_bytes = memoryview(
                        (torch.clamp(net_image, min=0, max=1.0) * 255)
                        .byte()
                        .permute(1, 2, 0)
                        .contiguous()
                        .cpu()
                        .numpy()
                    )
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and (
                    (iteration < int(opt.iterations)) or not keep_alive
                ):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(lr_iter)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if lr_iter % 1000 == 0:
            gaussians.oneupSHdegree()
        if lr_iter % 400 == 0:
            gaussians.scheduler.step()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
        )

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (
            1.0 - ssim(image, gt_image)
        )

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 1000 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(1000)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            if iteration in checkpoint_iterations:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                if not os.path.exists(scene.model_path):
                    os.makedirs(scene.model_path)
                torch.save(
                    (gaussians.capture(), iteration),
                    scene.model_path + "/chkpnt" + str(iteration) + ".pth",
                )
            init_report = iteration==args.prune_iterations[0]
            training_report(
                tb_writer,
                prune_idx,
                iteration,
                Ll1,
                loss,
                l1_loss,
                iter_start.elapsed_time(iter_end),
                testing_iterations,
                scene,
                render,
                (pipe, background),
                after_prune=False,
                init_report=init_report
            )
                
            if iteration in args.prune_iterations:
                prune_percent = args.prune_percent[prune_idx]
                prune_idx += 1

                if args.prune_type == 'fisher':
                    # Compute and save CUDA Fisher
                    N = gaussians.get_xyz.shape[0]
                    device = gaussians.get_xyz.device
                    with torch.enable_grad():
                        fishers = torch.zeros(N,6,6, device=device).float()
                        for view_idx, view in tqdm(
                            enumerate(scene.getTrainCameras()), total=len(scene.getTrainCameras()),
                                                   desc="Computing Fisher..."):
                            pool_fisher_cuda(
                                view_idx, view, gaussians, pipe, background,
                                fishers, args.fisher_resolution
                            )
                    torch.save(fishers, scene.model_path + f'/fisher_iter{iteration}.pt')
                    # Prune using log determinant
                    fishers_sv = torch.linalg.svdvals(fishers)
                    fishers_log_dets = torch.log(fishers_sv).sum(dim=1)
                    gaussians.prune_gaussians(
                        prune_percent,
                        fishers_log_dets
                   )
                # Borrowed from https://github.com/VITA-Group/LightGaussian/blob/main/prune_finetune.py#L222
                elif args.prune_type == 'v_important_score':
                    gaussian_list, imp_list = prune_list(gaussians, scene, pipe, background)
                    v_list = calculate_v_imp_score(gaussians, imp_list, args.v_pow)
                    gaussians.prune_gaussians(
                        prune_percent, 
                        v_list
                    )
                else:
                    raise Exception("Unsupportive pruning method")

                print(f"Prune Round {prune_idx}: Number of Gaussians is {len(gaussians.get_xyz)}")
                training_report(
                    tb_writer,
                    prune_idx,
                    iteration,
                    Ll1,
                    loss,
                    l1_loss,
                    iter_start.elapsed_time(iter_end),
                    testing_iterations,
                    scene,
                    render,
                    (pipe, background),
                    after_prune=True,
                    init_report=False,
                )
                
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
            else:
                return

            # Refreshing environment before pruning boosts performance
            if lr_iter == opt.position_lr_max_steps:  
                print("\n[ITER {}] Refreshing environment".format(iteration))
                point_cloud_path = os.path.join(scene.model_path, "point_cloud/iteration_{}".format(iteration))
                if os.path.exists(point_cloud_path):
                    print("Loading {}".format(point_cloud_path))
                    del gaussians.scheduler
                    del gaussians.optimizer
                    del gaussians.xyz_gradient_accum
                    del gaussians.denom
                    del gaussians
                    del scene.gaussians
                    del scene
                    gc.collect()
                    gaussians = GaussianModel(dataset.sh_degree)
                    scene = Scene(dataset, gaussians)
                    gaussians.load_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
                gaussians.training_setup(opt)
                gaussians.max_radii2D = torch.zeros((gaussians.get_xyz.shape[0]), device="cuda")
                gaussians.scheduler = ExponentialLR(gaussians.optimizer, gamma=0.95)
                viewpoint_stack = None
                lr_iter = first_iter
            else:
                lr_iter += 1


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Pruning script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument(
        "--test_iterations", nargs="+", type=int, default=[30_000, 30_001, 35_000, 35_001, 40_000]
    )
    parser.add_argument(
        "--save_iterations", nargs="+", type=int, default=[35_000, 40_000]
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--checkpoint_iterations", nargs="+", type=int, default=[35_000, 40_000]
    )
    parser.add_argument("--prune_iterations", nargs="+", type=int, default=[30_001, 35_001])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--start_pointcloud", type=str, default=None)
    parser.add_argument("--prune_type", type=str, default="fisher")
    parser.add_argument("--prune_percent", nargs="+", type=float, default=[0.8, 0.5])
    parser.add_argument("--fisher_resolution", type=int, default=1)
    parser.add_argument("--v_pow", type=float, default=0.1)
    args = parser.parse_args(sys.argv[1:])

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.debug_from,
        args,
    )

    # All done
    print("\nPruning complete.")
