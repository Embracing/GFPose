# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
# pytype: skip-file
"""Various sampling methods."""
import functools

import torch
import numpy as np
import abc

from .utils import from_flattened_numpy, to_flattened_numpy, get_score_fn
from scipy import integrate
from lib.algorithms.advanced import sde_lib
from . import utils as mutils

_CORRECTORS = {}
_PREDICTORS = {}


def register_predictor(cls=None, *, name=None):
  """A decorator for registering predictor classes."""

  def _register(cls):
    if name is None:
      local_name = cls.__name__
    else:
      local_name = name
    if local_name in _PREDICTORS:
      raise ValueError(f'Already registered model with name: {local_name}')
    _PREDICTORS[local_name] = cls
    return cls

  if cls is None:
    return _register
  else:
    return _register(cls)


def register_corrector(cls=None, *, name=None):
  """A decorator for registering corrector classes."""

  def _register(cls):
    if name is None:
      local_name = cls.__name__
    else:
      local_name = name
    if local_name in _CORRECTORS:
      raise ValueError(f'Already registered model with name: {local_name}')
    _CORRECTORS[local_name] = cls
    return cls

  if cls is None:
    return _register
  else:
    return _register(cls)


def get_predictor(name):
  return _PREDICTORS[name]


def get_corrector(name):
  return _CORRECTORS[name]


def get_sampling_fn(config, sde, shape, inverse_scaler, eps, device=None):
  """Create a sampling function.

  Args:
    config: A `ml_collections.ConfigDict` object that contains all configuration information.
    sde: A `sde_lib.SDE` object that represents the forward SDE.
    shape: A sequence of integers representing the expected shape of a single sample.
    inverse_scaler: The inverse data normalizer function.
    eps: A `float` number. The reverse-time SDE is only integrated to `eps` for numerical stability.

  Returns:
    A function that takes random states and a replicated training state and outputs samples with the
      trailing dimensions matching `shape`.
  """

  if device is None:
    device = config.device
  sampler_name = config.sampling.method
  # Probability flow ODE sampling with black-box ODE solvers
  if sampler_name.lower() == 'ode':
    sampling_fn = get_ode_sampler(sde=sde,
                                  shape=shape,
                                  inverse_scaler=inverse_scaler,
                                  denoise=config.sampling.noise_removal,
                                  eps=eps,
                                  device=device)
  # Predictor-Corrector sampling. Predictor-only and Corrector-only samplers are special cases.
  elif sampler_name.lower() == 'pc':
    predictor_name = config.sampling.predictor.lower()
    corrector_name = config.sampling.corrector.lower()
    predictor = get_predictor(predictor_name)
    corrector = get_corrector(corrector_name)
    sampling_fn = get_pc_sampler(sde=sde,
                                 shape=shape,
                                 predictor=predictor,
                                 corrector=corrector,
                                 inverse_scaler=inverse_scaler,
                                 snr=config.sampling.snr,
                                 n_steps=config.sampling.n_steps_each,
                                 probability_flow=config.sampling.probability_flow,
                                 continuous=config.training.continuous,
                                 denoise=config.sampling.noise_removal,
                                 eps=eps,
                                 device=device)
  else:
    raise ValueError(f"Sampler name {sampler_name} unknown.")

  return sampling_fn


class Predictor(abc.ABC):
  """The abstract class for a predictor algorithm."""

  def __init__(self, sde, score_fn, probability_flow=False):
    super().__init__()
    self.sde = sde
    # Compute the reverse SDE/ODE
    self.rsde = sde.reverse(score_fn, probability_flow)
    self.score_fn = score_fn

  @abc.abstractmethod
  def update_fn(self, x, t, condition, mask):
    """One update of the predictor.

    Args:
      x: A PyTorch tensor representing the current state
      t: A Pytorch tensor representing the current time step.

    Returns:
      x: A PyTorch tensor of the next state.
      x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
    """
    pass


class Corrector(abc.ABC):
  """The abstract class for a corrector algorithm."""

  def __init__(self, sde, score_fn, snr, n_steps):
    super().__init__()
    self.sde = sde
    self.score_fn = score_fn
    self.snr = snr
    self.n_steps = n_steps

  @abc.abstractmethod
  def update_fn(self, x, t, condition, mask):
    """One update of the corrector.

    Args:
      x: A PyTorch tensor representing the current state
      t: A PyTorch tensor representing the current time step.

    Returns:
      x: A PyTorch tensor of the next state.
      x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
    """
    pass


@register_predictor(name='euler_maruyama')
class EulerMaruyamaPredictor(Predictor):
  def __init__(self, sde, score_fn, probability_flow=False):
    super().__init__(sde, score_fn, probability_flow)

  def update_fn(self, x, t, condition, mask):
    dt = -1. / self.rsde.N
    z = torch.randn_like(x)
    drift, diffusion = self.rsde.sde(x, t, condition, mask)
    x_mean = x + drift * dt
    x = x_mean + diffusion[:, None, None] * np.sqrt(-dt) * z
    return x, x_mean



@register_predictor(name='reverse_diffusion')
class ReverseDiffusionPredictor(Predictor):
  def __init__(self, sde, score_fn, probability_flow=False):
    super().__init__(sde, score_fn, probability_flow)

  def update_fn(self, x, t, condition, mask):
    f, G = self.rsde.discretize(x, t, condition, mask)
    z = torch.randn_like(x)
    x_mean = x - f
    x = x_mean + G[:, None, None] * z
    return x, x_mean


@register_predictor(name='ancestral_sampling')
class AncestralSamplingPredictor(Predictor):
  """The ancestral sampling predictor. Currently only supports VE/VP SDEs."""

  def __init__(self, sde, score_fn, probability_flow=False):
    super().__init__(sde, score_fn, probability_flow)
    if not isinstance(sde, sde_lib.VPSDE) and not isinstance(sde, sde_lib.VESDE):
      raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")
    assert not probability_flow, "Probability flow not supported by ancestral sampling"

  def vesde_update_fn(self, x, t, condition, mask):
    sde = self.sde
    timestep = (t * (sde.N - 1) / sde.T).long()
    sigma = sde.discrete_sigmas[timestep]
    adjacent_sigma = torch.where(timestep == 0, torch.zeros_like(t), sde.discrete_sigmas.to(t.device)[timestep - 1])
    score = self.score_fn(x, t, condition, mask)
    x_mean = x + score * (sigma ** 2 - adjacent_sigma ** 2)[:, None, None]
    std = torch.sqrt((adjacent_sigma ** 2 * (sigma ** 2 - adjacent_sigma ** 2)) / (sigma ** 2))
    noise = torch.randn_like(x)
    x = x_mean + std[:, None, None] * noise
    return x, x_mean

  def vpsde_update_fn(self, x, t, condition, mask):
    sde = self.sde
    timestep = (t * (sde.N - 1) / sde.T).long()
    beta = sde.discrete_betas.to(t.device)[timestep]
    score = self.score_fn(x, t, condition, mask)
    x_mean = (x + beta[:, None, None] * score) / torch.sqrt(1. - beta)[:, None, None]
    noise = torch.randn_like(x)
    x = x_mean + torch.sqrt(beta)[:, None, None] * noise
    return x, x_mean

  def update_fn(self, x, t, condition, mask):
    if isinstance(self.sde, sde_lib.VESDE):
      return self.vesde_update_fn(x, t, condition, mask)
    elif isinstance(self.sde, sde_lib.VPSDE):
      return self.vpsde_update_fn(x, t, condition, mask)


@register_predictor(name='none')
class NonePredictor(Predictor):
  """An empty predictor that does nothing."""

  def __init__(self, sde, score_fn, probability_flow=False):
    pass

  def update_fn(self, x, t, condition, mask):
    return x, x


@register_corrector(name='langevin')
class LangevinCorrector(Corrector):
  def __init__(self, sde, score_fn, snr, n_steps):
    super().__init__(sde, score_fn, snr, n_steps)
    if not isinstance(sde, sde_lib.VPSDE) \
        and not isinstance(sde, sde_lib.VESDE) \
        and not isinstance(sde, sde_lib.subVPSDE):
      raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

  def update_fn(self, x, t, condition, mask):
    sde = self.sde
    score_fn = self.score_fn
    n_steps = self.n_steps
    target_snr = self.snr
    if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
      timestep = (t * (sde.N - 1) / sde.T).long()
      alpha = sde.alphas.to(t.device)[timestep]
    else:
      alpha = torch.ones_like(t)

    for i in range(n_steps):
      grad = score_fn(x, t, condition, mask)
      noise = torch.randn_like(x)
      grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
      noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
      step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
      x_mean = x + step_size[:, None, None] * grad
      x = x_mean + torch.sqrt(step_size * 2)[:, None, None] * noise

    return x, x_mean


@register_corrector(name='ald')
class AnnealedLangevinDynamics(Corrector):
  """The original annealed Langevin dynamics predictor in NCSN/NCSNv2.

  We include this corrector only for completeness. It was not directly used in our paper.
  """

  def __init__(self, sde, score_fn, snr, n_steps):
    super().__init__(sde, score_fn, snr, n_steps)
    if not isinstance(sde, sde_lib.VPSDE) \
        and not isinstance(sde, sde_lib.VESDE) \
        and not isinstance(sde, sde_lib.subVPSDE):
      raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

  def update_fn(self, x, t, condition, mask):
    sde = self.sde
    score_fn = self.score_fn
    n_steps = self.n_steps
    target_snr = self.snr
    if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
      timestep = (t * (sde.N - 1) / sde.T).long()
      alpha = sde.alphas.to(t.device)[timestep]
    else:
      alpha = torch.ones_like(t)

    std = self.sde.marginal_prob(x, t)[1]

    for i in range(n_steps):
      grad = score_fn(x, t, condition, mask)
      noise = torch.randn_like(x)
      step_size = (target_snr * std) ** 2 * 2 * alpha
      x_mean = x + step_size[:, None, None] * grad
      x = x_mean + noise * torch.sqrt(step_size * 2)[:, None, None]

    return x, x_mean


@register_corrector(name='none')
class NoneCorrector(Corrector):
  """An empty corrector that does nothing."""

  def __init__(self, sde, score_fn, snr, n_steps):
    pass

  def update_fn(self, x, t, condition, mask):
    return x, x


def shared_predictor_update_fn(x, t, condition, mask, sde, model, predictor, probability_flow, continuous):
  """A wrapper that configures and returns the update function of predictors."""
  score_fn = mutils.get_score_fn(sde, model, train=False, continuous=continuous)
  if predictor is None:
    # Corrector-only sampler
    predictor_obj = NonePredictor(sde, score_fn, probability_flow)
  else:
    predictor_obj = predictor(sde, score_fn, probability_flow)
  return predictor_obj.update_fn(x, t, condition, mask)


def shared_corrector_update_fn(x, t, condition, mask, sde, model, corrector, continuous, snr, n_steps):
  """A wrapper tha configures and returns the update function of correctors."""
  score_fn = mutils.get_score_fn(sde, model, train=False, continuous=continuous)
  if corrector is None:
    # Predictor-only sampler
    corrector_obj = NoneCorrector(sde, score_fn, snr, n_steps)
  else:
    corrector_obj = corrector(sde, score_fn, snr, n_steps)
  return corrector_obj.update_fn(x, t, condition, mask)

def get_match_grad_fn(weight=1.0):
  def match_grad_fn(x, t, condition):
    # Add extra gradient guid
    with torch.enable_grad():
      x_extra = x.detach().clone()
      x_extra.requires_grad = True
      condition_extra = condition.detach().clone()
      condition_extra.requires_grad = True
      # 2d match loss
      loss = torch.sum(torch.linalg.norm(x_extra[..., :2] - condition_extra, dim=-1))
      match_grad = torch.autograd.grad(loss, x_extra)[0]  # [b, j ,3], zeros for unused elements
    return match_grad * weight
  return match_grad_fn

def get_sym_grad_fn(weight=1.0):
  def sym_grad_fn(x, t, condition):
    """
    data: [B, j, 3]
    """
    with torch.enable_grad():
      norm = functools.partial(torch.linalg.norm, dim=-1)
      mean = torch.mean

      left_parent_joints = [12, 11, 8, 0, 4, 5]
      left_children_joints = [13, 12, 11, 4, 5, 6]

      right_parent_joints = [15, 14, 8, 0, 1, 2]
      right_children_joints = [16, 15, 14, 1, 2, 3]

      left_limbs = norm(
          x[:, left_parent_joints, :] - x[:, left_children_joints, :],
      )  # [b, 6]
      right_limbs = norm(
          x[:, right_parent_joints, :] - x[:, right_children_joints, :],
      )  # [b, 6]

      res = mean((left_limbs - right_limbs) ** 2) * weight
    return res
  return sym_grad_fn


def get_pc_sampler(sde, shape, predictor, corrector, inverse_scaler, snr,
                   n_steps=1, probability_flow=False, continuous=False,
                   denoise=True, eps=1e-3, device='cuda'):
  """Create a Predictor-Corrector (PC) sampler.

  Args:
    sde: An `sde_lib.SDE` object representing the forward SDE.
    shape: A sequence of integers. The expected shape of a single sample.
    predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
    corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
    inverse_scaler: The inverse data normalizer.
    snr: A `float` number. The signal-to-noise ratio for configuring correctors.
    n_steps: An integer. The number of corrector steps per predictor update.
    probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
    continuous: `True` indicates that the score model was continuously trained.
    denoise: If `True`, add one-step denoising to the final samples.
    eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
    device: PyTorch device.

  Returns:
    A sampling function that returns samples and the number of function evaluations during sampling.
  """
  # Create predictor & corrector update functions
  predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                          sde=sde,
                                          predictor=predictor,
                                          probability_flow=probability_flow,
                                          continuous=continuous)
  corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                          sde=sde,
                                          corrector=corrector,
                                          continuous=continuous,
                                          snr=snr,
                                          n_steps=n_steps)

  def get_imputation_update_fn(update_fn):
    """Modify the update function of predictor & corrector to incorporate data information."""

    def imputation_update_fn(x, vec_t, condition, mask, model, args):
      x, x_mean = update_fn(x, vec_t, condition, mask, model=model)

      if args is not None and args.task in ['comp3d', 'comp2d']:
        masked_data_mean, std = sde.marginal_prob(condition, vec_t)
        masked_data = masked_data_mean + torch.randn_like(x) * std[:, None, None]

        x = x * (1 - mask) + masked_data * mask
        x_mean = x_mean * (1 - mask) + masked_data_mean * mask
      # masked_data_mean, std = sde.marginal_prob(data, vec_t)
      # masked_data = masked_data_mean + torch.randn_like(x) * std[:, None, None, None]
      # x = x * (1. - mask) + masked_data * mask
      # x_mean = x * (1. - mask) + masked_data_mean * mask
      return x, x_mean

    return imputation_update_fn

  projector_imputation_update_fn = get_imputation_update_fn(predictor_update_fn)
  corrector_imputation_update_fn = get_imputation_update_fn(corrector_update_fn)


  def pc_sampler(model, condition, denoise_x=None, args=None):
    """ The PC sampler funciton.

    Args:
      model: A score model.
    Returns:
      Samples, number of function evaluations.
    """
    with torch.no_grad():
      # Initial sample
      batch_size = condition.shape[0]
      defined_batch_size = shape[0]
      if batch_size < defined_batch_size:
        # e.g. the last batch
        real_shape = (batch_size, *shape[1:])
      else:
        real_shape = shape
      # x = sde.prior_sampling(real_shape).to(device)
      # x = torch.zeros(real_shape, device=device)
      timesteps = torch.linspace(sde.T, eps, sde.N, device=device)
      # TODO: Add extra t here
      # part_list = [14, 15, 16]
      # rest_list = [i for i in range(17) if i not in part_list]
      # x = denoise_x
      # x = sde.prior_sampling(real_shape).to(device)
      # x[:, :, :2] = condition
      # condition = condition * 0
      # timesteps = torch.linspace(sde.T, eps, sde.N, device=device)

      x = sde.prior_sampling(real_shape).to(device)

      # part_list = [14, 15, 16]
      # rest_list = [i for i in range(17) if i not in part_list]

      # make mask
      mask = torch.ones_like(x)  # [B, j, 3], non-zero joints
      if args is None or args.task == 'est':
        # 2D pose estimation
        mask[..., -1] = 0  # mask depth
      elif args.task == 'comp2d':
        if args.jlist:
            part_list = list(map(int, args.jlist.split(',')))
            rest_list = [i for i in range(17) if i not in part_list]
            mask[:, part_list, :] = 0
        elif args.randj:
            # for 2d completion, random joints from limbs
            joint_idx = np.array([np.random.choice([12, 13, 15, 16, 5, 6, 2, 3], args.randj, replace=False) for _ in range(len(x))])
            joint_idx = torch.tensor(joint_idx, device=device).view(-1).long()  # [b*2]
            batch_idx = torch.repeat_interleave(torch.arange(len(x)), args.randj).view(-1).long()  # [b*2]
            mask[batch_idx, joint_idx] = 0  # [B, j, 3]
        mask[..., -1] = 0  # mask depth
      elif args.task == 'comp3d':
        if args.jlist:
          part_list = list(map(int, args.jlist.split(',')))
          rest_list = [i for i in range(17) if i not in part_list]
          mask[:, part_list, :] = 0
        elif args.randj:
          # for 3d completion, random joints from limbs
          joint_idx = np.array([np.random.choice([12, 13, 15, 16, 5, 6, 2, 3], args.randj, replace=False) for _ in range(len(x))])
          joint_idx = torch.tensor(joint_idx, device=device).view(-1).long()  # [b*2]
          batch_idx = torch.repeat_interleave(torch.arange(len(x)), args.randj).view(-1).long()  # [b*2]
          mask[batch_idx, joint_idx] = 0  # [B, j, 3]
      elif args.task == 'den':
        mask = mask * 0
      elif args.task == 'gen':
        mask = mask * 0

      # make init_x
      if args is None or args.task == 'est':
        # 3D pose estimation
        # eval or eval epoch during training
        # x[:, :, :2] = denoise_x[:, :, :2]
        pass
      elif args.task == 'comp3d':
        # denoise_x == gt3d
        # condition == gt3d
        x = x * (1 - mask) + condition * mask
      elif args.task == 'comp2d':
        # denoise_x == dt2d
        # condition == dt2d
        x = x * (1 - mask) + condition * mask
      elif args.task == 'den':
        # denoise_x == noisy3d
        # condition == 0
        x = denoise_x
      else:
        # generation
        pass

      trajs = []
      for i in range(sde.N):
        t = timesteps[i]
        vec_t = torch.ones(batch_size, device=t.device) * t
        x, x_mean = corrector_imputation_update_fn(x, vec_t, condition, mask, model=model, args=args)
        x, x_mean = projector_imputation_update_fn(x, vec_t, condition, mask, model=model, args=args)
        trajs.append(x.cpu().numpy())

        # if args is not None and args.task in ['comp3d', 'comp2d']:
        #   masked_data_mean, std = sde.marginal_prob(condition, vec_t)
        #   masked_data = masked_data_mean + torch.randn_like(x) * std[:, None, None]

        #   x = x * (1 - mask) + masked_data * mask
        #   x_mean = x_mean * (1 - mask) + masked_data_mean * mask

      trajs = np.stack(trajs, axis=0)  # [t, b, j, 3]
      x_mean = x_mean.cpu().numpy()
      trajs[-1] = x_mean  # last step == x_mean
      return trajs, x_mean if denoise else x

  return pc_sampler


def get_ode_sampler(sde, shape, inverse_scaler,
                    denoise=False, rtol=1e-5, atol=1e-5,
                    method='RK45', eps=1e-3, device='cuda'):
  """Probability flow ODE sampler with the black-box ODE solver.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    shape: A sequence of integers. The expected shape of a single sample.
    inverse_scaler: The inverse data normalizer.
    denoise: If `True`, add one-step denoising to final samples.
    rtol: A `float` number. The relative tolerance level of the ODE solver.
    atol: A `float` number. The absolute tolerance level of the ODE solver.
    method: A `str`. The algorithm used for the black-box ODE solver.
      See the documentation of `scipy.integrate.solve_ivp`.
    eps: A `float` number. The reverse-time SDE/ODE will be integrated to `eps` for numerical stability.
    device: PyTorch device.

  Returns:
    A sampling function that returns samples and the number of function evaluations during sampling.
  """

  def denoise_update_fn(model, x, condition):
    score_fn = get_score_fn(sde, model, train=False, continuous=True)
    # Reverse diffusion predictor for denoising
    predictor_obj = ReverseDiffusionPredictor(sde, score_fn, probability_flow=False)
    vec_eps = torch.ones(x.shape[0], device=x.device) * eps
    _, x = predictor_obj.update_fn(x, vec_eps, condition)
    return x

  def drift_fn(model, x, t, condition):
    """Get the drift function of the reverse-time SDE."""
    score_fn = get_score_fn(sde, model, train=False, continuous=True)
    rsde = sde.reverse(score_fn, probability_flow=True)
    return rsde.sde(x, t, condition)[0]

  def ode_sampler(model, z=None):
    """The probability flow ODE sampler with black-box ODE solver.

    Args:
      model: A score model.
      z: If present, generate samples from latent code `z`.
    Returns:
      samples, number of function evaluations.
    """
    with torch.no_grad():
      # Initial sample
      if z is None:
        # If not represent, sample the latent code from the prior distibution of the SDE.
        x = sde.prior_sampling(shape).to(device)
      else:
        x = z

      def ode_func(t, x):
        x = from_flattened_numpy(x, shape).to(device).type(torch.float32)
        vec_t = torch.ones(shape[0], device=x.device) * t
        drift = drift_fn(model, x, vec_t)
        return to_flattened_numpy(drift)

      # Black-box ODE solver for the probability flow ODE
      solution = integrate.solve_ivp(ode_func, (sde.T, eps), to_flattened_numpy(x),
                                     rtol=rtol, atol=atol, method=method)
      nfe = solution.nfev
      x = torch.tensor(solution.y[:, -1]).reshape(shape).to(device).type(torch.float32)

      # Denoising is equivalent to running one predictor step without adding noise
      if denoise:
        x = denoise_update_fn(model, x)

      x = inverse_scaler(x)
      return x, nfe

  return ode_sampler
