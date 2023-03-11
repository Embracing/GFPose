import os
import numpy as np
import functools
import pprint
import sys
import traceback
import argparse
from pathlib import Path

from absl import app
from absl import flags
from absl.flags import argparse_flags
from ml_collections.config_flags import config_flags

# torch related
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

try:
    from tensorboardX import SummaryWriter
except ImportError as e:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as e:
        print('Tensorboard is not Installed')

from lib.utils.generic import create_logger

from lib.algorithms.advanced.model import ScoreModelFC_Adv
from lib.algorithms.advanced import losses, sde_lib, sampling
from lib.algorithms.ema import ExponentialMovingAverage

from lib.dataset.h36m import H36MDataset3D, denormalize_data


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
  "config", None, "Training configuration.", lock_config=False)
flags.mark_flags_as_required(["config"])

# global device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_JOINTS = 17
JOINT_DIM = 3
HIDDEN_DIM = 1024
EMBED_DIM = 512
CONDITION_DIM = 3
# BATCH_SIZE = 100000
# TEST_BATCH_SIZE = 10000
N_EPOCHES = 20000
EVAL_FREQ = 5  # 20


def parse_args(argv):
    parser = argparse_flags.ArgumentParser(description='train score model')

    # parser.add_argument('--prior-t0', type=float, default=0.5)
    # parser.add_argument('--test-num', type=int, default=20)
    # parser.add_argument('--sample-steps', type=int, default=2000)
    parser.add_argument('--restore-dir', type=str)
    parser.add_argument('--gt', action='store_true', 
        default=False, help='use gt2d as condition')
    parser.add_argument('--sample', type=int, help='sample trainset to reduce data')
    parser.add_argument('--flip', default=False, action='store_true', help='random flip pose during training')

    # optional
    parser.add_argument('--name', type=str, default='', help='name of checkpoint folder')

    args = parser.parse_args(argv[1:])

    return args


def get_dataloader(subset='train', sample_interval=None, gt2d=False, flip=False, cond_3d_prob=0):
    dataset = H36MDataset3D(Path('data', 'h36m'), 
        subset, gt2d=gt2d,
        read_confidence=False, sample_interval=sample_interval,
        flip=flip,
        cond_3d_prob=cond_3d_prob)
    print(f'H36M 3D {subset} dataset with 3D conditional prob: {cond_3d_prob}')
    
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
    # args = parse_args()
    config = FLAGS.config

    logger, final_output_dir, tb_log_dir = create_logger(
            config, 'train', folder_name=args.name)
    logger.info(pprint.pformat(config))
    logger.info(pprint.pformat(args))
    writer = SummaryWriter(tb_log_dir)

    ''' setup datasets, dataloaders'''
    if args.gt:
        logger.info('use gt data as condition')
    else:
        logger.info('use dt data as condition')
    if args.sample:
        logger.info(f'sample trainset every {args.sample} frame')

    train_loader, train_dataset = get_dataloader('train', args.sample, args.gt, flip=args.flip,
        cond_3d_prob=config.training.cond_3d_prob)
    test_loader, test_dataset = get_dataloader('test', 640, args.gt, flip=False,
        cond_3d_prob=0)  # always sample testset to save time
    logger.info(f'total train samples: {len(train_dataset.db_3d)}')
    logger.info(f'total test samples: {len(test_dataset.db_3d)}')

    ''' setup score networks '''
    # sigma = 25.0  # @param {'type':'number'}
    # marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
    # diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)
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
    # optimizer = optim.Adam(model.parameters(), lr=LR, betas=(BETA1, 0.999))
    ema = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_rate)
    optimizer = losses.get_optimizer(config, model.parameters())
    # patience is the number of eval times
    # lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.2)
    # lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80,120], gamma=0.1)

    state = dict(optimizer=optimizer, model=model, ema=ema, step=0)  # based on iteration instead of epochs

    # auto resume
    start_epoch = 0
    if args.restore_dir and os.path.exists(args.restore_dir):
        ckpt_path = os.path.join(args.restore_dir, 'checkpoint.pth')
        logger.info(f'=> loading checkpoint: {ckpt_path}')

        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        ema.load_state_dict(checkpoint['ema'])
        state['step'] = checkpoint['step']

        logger.info(f"=> loaded checkpoint '{ckpt_path}' (epoch {start_epoch})")

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

    # Build one-step training and evaluation functions
    optimize_fn = losses.optimization_manager(config)
    continuous = config.training.continuous
    reduce_mean = config.training.reduce_mean
    likelihood_weighting = config.training.likelihood_weighting
    train_step_fn = losses.get_step_fn(sde, train=True, optimize_fn=optimize_fn,
                                       reduce_mean=False, continuous=continuous,
                                       likelihood_weighting=likelihood_weighting)

    sampling_shape = (config.eval.batch_size, N_JOINTS, JOINT_DIM)
    config.sampling.probability_flow = True
    sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)

    # num_train_steps = config.training.n_iters

    best_error = 1e5
    try:
        ''' training loop '''
        # WARNING!!! This code assumes all poses are normed into [-1, 1]
        for epoch in range(start_epoch, N_EPOCHES):
            model.train()
            for idx, (data_2d, labels_3d) in enumerate(train_loader):
                labels_3d = labels_3d.to(device, non_blocking=True) * config.training.data_scale
                data_2d = data_2d.to(device, non_blocking=True) * config.training.data_scale

                cur_loss = train_step_fn(state, batch=labels_3d, condition=data_2d)

                writer.add_scalar('train_loss', cur_loss.item(), idx + epoch * len(train_loader))

            logger.info(
                f'EPOCH: [{epoch}/{N_EPOCHES}, {epoch/N_EPOCHES*100:.2f}%][{idx}/{len(train_loader)}],\t'
                f'Loss: {cur_loss.item()}'
            )

            ''' eval '''
            if epoch % EVAL_FREQ == 0:
                # sampling process
                model.eval()
                with torch.no_grad():
                    all_results = []

                    for idx, (data_2d, labels_3d) in enumerate(test_loader):
                        labels_3d = labels_3d.to(device, non_blocking=True)
                        data_2d = data_2d.to(device, non_blocking=True) * config.training.data_scale

                        # Generate and save samples
                        ema.store(model.parameters())
                        ema.copy_to(model.parameters())
                        trajs, results = sampling_fn(
                            model,
                            condition=data_2d
                        )  # [b ,j ,3]
                        ema.restore(model.parameters())

                        # # trajs: [t, b, j, 3], i.e., the pose-trajs
                        # # results: [b, j, 3], i.e., the end pose of each traj
                        results = results / config.training.data_scale
                        all_results.append(results)

                all_results = np.concatenate(all_results, axis=0)  # [N, j, 3]
                all_results = denormalize_data(all_results)  # [N, j, 3]
                mpjpe = test_dataset.eval(all_results, print_verbose=False)  # scala

                logger.info(f'TEST: [{epoch}/{N_EPOCHES}]')
                logger.info(f'{config.training.sde} {config.sampling.method} sampler: {config.sampling.predictor} {config.sampling.corrector}')
                logger.info(f'TEST:  MPJPE: {mpjpe:.2f}')
                writer.add_scalar('test_mpjpe', mpjpe, epoch)

                # save normalized pose, not org pose
                save_path = os.path.join(final_output_dir, 'last_samples.npz')
                logger.info(f'save eval samples to {save_path}')
                np.savez(save_path,
                    **{
                        'pred3d': trajs[:, :20, None, :, :],  # [t, b, 1, j, 3]
                        'gt3d': labels_3d[:20, :, :].cpu().numpy()[None, :, None, ...]  # [1, b, 1, j, 3]
                    }
                )

                # log and save ckpt
                logger.info(f'Save checkpoint to {final_output_dir}')
                save_dict = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'ema': state['ema'].state_dict(),
                    'step': state['step'],
                }
                torch.save(save_dict, os.path.join(final_output_dir, 'checkpoint.pth'))

                if mpjpe < best_error:
                    # best checkpoint under my metric
                    best_error = mpjpe
                    logger.info('saving best checkpoint')
                    torch.save(
                        {
                            'model_state_dict': model.state_dict(),
                            'epoch': epoch + 1,
                            'ema': state['ema'].state_dict(),
                            'step': state['step'],
                        },
                        os.path.join(final_output_dir, 'best_model.pth')
                    )
            # lr_scheduler.step()
    except Exception as e:
        traceback.print_exc()
    finally:
        writer.close()
        logger.info(f'End. Final output dir: {final_output_dir}')


if __name__ == '__main__':
    app.run(main, flags_parser=parse_args)

