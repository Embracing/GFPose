import os
import numpy as np
import functools
import pprint
import sys
import traceback
from pathlib import Path
import time

from absl import app
from absl import flags
from absl.flags import argparse_flags
from ml_collections.config_flags import config_flags

# torch related
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from lib.utils.generic import create_logger
from lib.dataset.h36m import H36MDataset3D, denormalize_data, normalize_data
from lib.dataset.EvaSampler import DistributedEvalSampler

from lib.algorithms.advanced.model import ScoreModelFC_Adv
from lib.algorithms.advanced import losses, sde_lib, sampling
from lib.algorithms.ema import ExponentialMovingAverage


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
    parser = argparse_flags.ArgumentParser(description='valid score model', prog='GFPose')

    parser.add_argument('--ckpt-dir', type=str)
    # parser.add_argument('--prior-t0', type=float, default=1e-1)
    # parser.add_argument('--eps', type=float, default=1e-5)
    parser.add_argument('--hypo', type=int, default=1, help='number of hypotheses to sample')
    # parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--best', action='store_true')
    parser.add_argument('--sample', type=int, help='sample testset to reduce data')
    parser.add_argument('--gt', action='store_true', default=False, help='use gt2d as condition')
    parser.add_argument('--proto2', action='store_true', default=False, help='eval protocol2')
    parser.add_argument('--gpus', type=int, help='num gpus to inference parallel')
    parser.add_argument('--save', type=str, choices=['trajs', 'results'], help='save data for visualization')
    parser.add_argument('--pflow', action='store_true', default=False, help='probability flow for deterministic sampling')

    subparsers = parser.add_subparsers(help='sub-command help', dest='task')
    parser_est = subparsers.add_parser('est', help='3d pose estimation')
    parser_gen = subparsers.add_parser('gen', help='pose genearation')

    parser_comp2d = subparsers.add_parser('comp2d', help='pose completion from 2d observation')
    group_comp2d = parser_comp2d.add_mutually_exclusive_group(required=True)
    group_comp2d.add_argument('--randj', type=int, help='number of randomly masked joints')
    group_comp2d.add_argument('--jlist', type=str,
        choices=['1,2,3', '4,5,6', '11,12,13', '14,15,16', '0,7,8,9,10', '1,2,3,4,5,6'])  # ...and combinations

    parser_comp3d = subparsers.add_parser('comp3d', help='pose completion from 3d observation')
    group_comp3d = parser_comp3d.add_mutually_exclusive_group(required=True)
    group_comp3d.add_argument('--randj', type=int, help='number of randomly masked joints')
    group_comp3d.add_argument('--jlist', type=str,
        choices=['1,2,3', '4,5,6', '11,12,13', '14,15,16', '0,7,8,9,10', '1,2,3,4,5,6'])  # ...and combinations

    args = parser.parse_args(argv[1:])

    if args.task is None:
        raise RuntimeError("arg [task] should be specified")

    return args


def get_dataloader(subset='test',
     sample_interval=None, gt2d=False,
     num_replicas=1, rank=0, rep=1, batch_size=10000):
    if gt2d:
        print('use gt2d as condition')
    else:
        print('use dt2d as condition')

    dataset = H36MDataset3D(
        Path('data', 'h36m'), 
        subset, gt2d=gt2d,
        read_confidence=False,
        sample_interval=sample_interval,
        rep=rep,
        cond_3d_prob=0)  # [data] * rep for multi-sample-eval
    
    sampler = DistributedEvalSampler(dataset,
        num_replicas=num_replicas,
        rank=rank,
        shuffle=False)

    # test_labels = torch.FloatTensor(test_labels).reshape((-1, 17, 3)) # [N, 17, 3]
    dataloader = DataLoader(dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        sampler=sampler,
        persistent_workers=False,
        pin_memory=True,)

    return dataloader, dataset

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '15389'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def inference(rank, args, config):
    print(f"Running DDP on rank {rank}.")
    setup(rank, args.gpus)

    ## Load the pre-trained checkpoint from disk.
    device = torch.device("cuda", rank)

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
    map_location = {'cuda:0': 'cuda:%d' % rank}
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
        sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
        sampling_eps = 1e-3
    elif config.training.sde.lower() == 'subvpsde':
        sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
        sampling_eps = 1e-3
    elif config.training.sde.lower() == 'vesde':
        sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
        sampling_eps = 1e-5
    else:
        raise NotImplementedError(f"SDE {config.training.sde} unknown.")

    # Setup sampling functions
    sampling_shape = (config.eval.batch_size, N_JOINTS, JOINT_DIM)
    if args.pflow:
        config.sampling.probability_flow = args.pflow
    sampling_fn = sampling.get_sampling_fn(config, sde, 
        sampling_shape, inverse_scaler, sampling_eps, device=device)

    test_loader, test_dataset = get_dataloader('test', args.sample, args.gt,
        num_replicas=args.gpus, rank=rank, rep=args.hypo,
        batch_size=config.eval.batch_size)
    if rank == 0:
        print(f'real samples: {len(test_dataset.db_3d)}')
        print(f'total samples with repeat: {len(test_dataset)}')

    with torch.no_grad():

        all_results = []
        all_trajs = []

        start = time.time()
        for idx, (data_2d, labels_3d) in enumerate(test_loader):
            data_2d = data_2d.to(device, non_blocking=True) * config.training.data_scale
            if args.task == 'comp3d':
                # for 3d completion
                data_2d = labels_3d.clone().to(device) * config.training.data_scale
                denoise_x = labels_3d.to(device) * config.training.data_scale
            elif args.task == 'comp2d':
                denoise_x = data_2d
            elif args.task == 'est':
                # for 3d pose estimation
                denoise_x = None
            elif args.task == 'gen':
                data_2d = data_2d * 0
                denoise_x = None
            else:
                raise NotImplementedError

            trajs, results = sampling_fn(
                model,
                condition=data_2d,
                denoise_x=denoise_x,
                args=args
            )  # [b ,j ,3]

            results = results / config.training.data_scale
            all_results.append(results)
            if args.save == 'trajs':
                trajs = trajs / config.training.data_scale
                all_trajs.append(trajs)
        if rank == 0:
            print(f'total sample time: {time.time() - start}')

    all_results = np.concatenate(all_results, axis=0)  # [split_set, j, 3]
    all_results = denormalize_data(all_results)  # [split_set, j, 3]

    # collect data from other process
    print(f'rank[{rank}] subset len: {len(all_results)}')

    results_collection = [None for _ in range(args.gpus)]
    dist.gather_object(
        all_results,
        results_collection if rank == 0 else None,
        dst=0
    )

    if args.save == 'trajs':
        all_trajs = np.concatenate(all_trajs, axis=1)  # [t, split_set, j, 3]
        trajs_collection = [None for _ in range(args.gpus)]
        dist.gather_object(
            all_trajs,
            trajs_collection if rank == 0 else None,
            dst=0
        )

    if rank == 0:
        collected_results = np.concatenate(results_collection, axis=0)  # [m*N, j, 3]
        # [N, m, j, 3]
        collected_results = collected_results.reshape((args.hypo, -1, N_JOINTS, 3)).transpose((1, 0, 2, 3))

        # eval data
        start = time.time()
        test_dataset.eval_multi(collected_results, protocol2=args.proto2, print_verbose=True)
        print(f'eval time: {time.time() - start}')

        # save
        if args.save:
            if args.save == 'trajs':
                # save the whole trajectory
                collected_trajs = np.concatenate(trajs_collection, axis=1)  # [t, h*n, j, 3]
                # [t, n, h, j, 3]
                save_data = collected_trajs.reshape((collected_trajs.shape[0], args.hypo, -1, N_JOINTS, 3)).transpose((0, 2, 1, 3, 4))
            else:
                # save final results
                save_data = normalize_data(collected_results)[None, ...]  # [1, n, h, j, 3]
            gt_3d = test_dataset.db_3d.reshape((1, -1, 1, N_JOINTS, JOINT_DIM))  # [1, n, 1, j, 3]
            save_path = os.path.join(args.ckpt_dir, 'results.npz')
            print(f'save eval samples to {save_path}')
            np.savez(save_path, **{'pred3d': save_data, 'gt3d': gt_3d})

    cleanup()


def main(args):
    # mp.freeze_support()
    mp.set_start_method('spawn')

    config = FLAGS.config

    processes = []
    for rank in range(args.gpus):
        p = mp.Process(target=inference, args=(rank, args, config))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()


if __name__ == '__main__':

    app.run(main, flags_parser=parse_args)