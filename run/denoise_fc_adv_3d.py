import os
import numpy as np
import h5py
# from ipdb import set_trace
import functools
import pprint
import sys
import traceback
# import argparse
from pathlib import Path
import time

# torch related
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP


from lib.utils.generic import create_logger
from lib.dataset.h36m import H36MDataset3D, denormalize_data
from lib.dataset.EvaSampler import DistributedEvalSampler
from lib.algorithms.advanced.model import ScoreModelFC_Adv
from lib.algorithms.advanced import losses, sde_lib, sampling
from lib.algorithms.ema import ExponentialMovingAverage
from lib.utils.transforms import align_to_gt

from absl import app
from absl import flags
from absl.flags import argparse_flags
from ml_collections.config_flags import config_flags

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
  "config", None, "Training configuration.", lock_config=False)
flags.mark_flags_as_required(["config"])

N_JOINTS = 17
JOINT_DIM = 3
HIDDEN_DIM = 1024
EMBED_DIM = 512
CONDITION_DIM = 3


def parse_args(argv):
    parser = argparse_flags.ArgumentParser(description='valid score model')
    parser.add_argument('--ckpt-dir', type=str)
    # parser.add_argument('--prior-t0', type=float, default=1e-1)
    # parser.add_argument('--eps', type=float, default=1e-5)
    parser.add_argument('--batch', type=int, default=886)
    # parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--best', action='store_true')
    parser.add_argument('--sample', type=int, default=640, help='sample testset to reduce data')
    parser.add_argument('--gt', action='store_true', default=False, help='use gt2d as condition')
    parser.add_argument('--save', action='store_true', help='save data for visualization')

    subparsers = parser.add_subparsers(help='sub-command help', dest='task')
    parser_den = subparsers.add_parser('den', help='pose denoising')
    parser_den.add_argument('--noise-type', type=str, choices=['gaussian', 'uniform'], required=True)
    parser_den.add_argument('--std', type=int, required=True)
    parser_den.add_argument('--t', type=float, required=True, default=1.0)

    args = parser.parse_args(argv[1:])

    return args


def get_dataloader(subset='train', sample_interval=None, gt2d=False, flip=False):
    dataset = H36MDataset3D(Path('data', 'h36m'),
        subset, gt2d=gt2d,
        read_confidence=False, sample_interval=sample_interval,
        flip=False,
        cond_3d_prob=0)

    if subset == 'train':
        # train_labels = torch.FloatTensor(train_labels).reshape((-1, 17, 3)) # [N, 17, 3]
        dataloader = DataLoader(dataset,
            batch_size=FLAGS.config.training.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True)
    else:
        # test_labels = torch.FloatTensor(test_labels).reshape((-1, 17, 3)) # [N, 17, 3]
        dataloader = DataLoader(dataset,
            batch_size=FLAGS.config.eval.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True)

    return dataloader, dataset


def main(args):
    config = FLAGS.config


    ## Load the pre-trained checkpoint from disk.
    device = torch.device("cuda")

    ''' setup score networks '''
    model = ScoreModelFC_Adv(
        config,
        n_joints=N_JOINTS,
        joint_dim=JOINT_DIM,
        hidden_dim=HIDDEN_DIM,
        embed_dim=EMBED_DIM,
        cond_dim=CONDITION_DIM,
        # n_blocks=1,
    )
    model.to(device)

    ema = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_rate)
    # optimizer = losses.get_optimizer(config, model.parameters())
    state = dict(optimizer=None, model=model, ema=ema, step=0)

    # restore checkpoint
    if args.best:
        ckpt_path = os.path.join(args.ckpt_dir, 'best_model.pth')
    else:
        ckpt_path = os.path.join(args.ckpt_dir, 'checkpoint.pth')
    print(f'loading model from {ckpt_path}')
    map_location = {'cuda:0': 'cuda:0'}
    checkpoint = torch.load(ckpt_path, map_location=map_location)

    model.load_state_dict(checkpoint['model_state_dict'])
    ema.load_state_dict(checkpoint['ema'])
    state['step'] = checkpoint['step']
    print(f"=> loaded checkpoint '{ckpt_path}' (step {state['step']})")

    model.eval()
    # TODO: choose whether to use ema here
    ema.copy_to(model.parameters())

    # Identity func
    scaler = lambda x: x
    inverse_scaler = lambda x: x

    # Setup SDEs
    if config.training.sde.lower() == 'vpsde':
        sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max,
            N=config.model.num_scales, T=args.t)
        sampling_eps = 1e-3
    elif config.training.sde.lower() == 'subvpsde':
        sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max,
            N=config.model.num_scales, T=args.t)
        sampling_eps = 1e-3
    elif config.training.sde.lower() == 'vesde':
        sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max,
            N=config.model.num_scales, T=args.t)
        sampling_eps = 1e-5
    else:
        raise NotImplementedError(f"SDE {config.training.sde} unknown.")

    # Setup sampling functions
    sampling_shape = (args.batch, N_JOINTS, JOINT_DIM)
    config.sampling.probability_flow = True
    sampling_fn = sampling.get_sampling_fn(config, sde,
        sampling_shape, inverse_scaler, sampling_eps, device=device)

    with torch.no_grad():

        test_dataset = H36MDataset3D(
            Path('data', 'h36m'),
            'test',
            gt2d=args.gt,
            read_confidence=False,
            sample_interval=args.sample,
            flip=False)
        gt_3d = test_dataset.db_3d  # [n, j, 3]

        assert args.batch == len(gt_3d), f'batch: {args.batch}, dataset len: {len(gt_3d)}'

        org_gt_3d = denormalize_data(gt_3d)
        noisy_gt_3d = test_dataset.add_noise(org_gt_3d, std=args.std, noise_type=args.noise_type)
        noisy_gt_3d[..., :2] = noisy_gt_3d[..., :2] / 1000 * 2 - [1, 1]
        noisy_gt_3d[..., 2:] = noisy_gt_3d[..., 2:] / 1000 * 2

        condition = torch.tensor(noisy_gt_3d[:, :, :3], device=device) * config.training.data_scale
        denoise_x = torch.tensor(noisy_gt_3d[:], device=device) * config.training.data_scale

        # condition = torch.tensor(test_dataset.db_2d, device=device) * config.training.data_scale  # [B, j, 2]
        # condition = condition / 5  # match prior

        trajs, results = sampling_fn(
                model,
                condition=condition * 0,
                denoise_x=denoise_x,
                args=args
            )  # [b ,j ,3]
        results = results / config.training.data_scale
        trajs = trajs / config.training.data_scale

    noisy_gt_3d = denormalize_data(noisy_gt_3d)  # [b, j, 3]
    results = denormalize_data(results)  # [b, j, 3]

    # remeber to comment out reprojection of pred in eval()
    print('eval noisy data...')
    test_dataset.eval(noisy_gt_3d, protocol2=True, print_verbose=True)

    print('eval...')
    test_dataset.eval(results, protocol2=True, print_verbose=True)

    # Save data
    if args.save:
        gt_save = gt_3d.reshape((1, -1, 1, N_JOINTS, JOINT_DIM))  # [1, b, 1, j, 3]
        trajs = trajs[:, :, None, ...]  # [t, b, 1, j, 3]

        # align to gt
        print('align to gt...')
        for time_idx in range(trajs.shape[0]):
            for batch_idx in range(trajs.shape[1]):
                pred = align_to_gt(pose=trajs[time_idx, batch_idx, 0], pose_gt=gt_save[0, batch_idx, 0])
                trajs[time_idx, batch_idx, 0] = pred

        save_path = os.path.join(args.ckpt_dir, 'denoise.npz')
        print(f'save eval samples to {save_path}')
        np.savez(save_path, **{'pred3d': trajs, 'gt3d': gt_save})

if __name__ == '__main__':

    app.run(main, flags_parser=parse_args)