import numpy as np
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def prune(w, perc, prune_type):
  if prune_type == 'weight':
    w_shape = list(w.size())
    w = w.view(w_shape[0], -1).transpose(0, 1)
    norm = torch.abs(w)
    idx = int(perc * w.shape[0])
    threshold = (torch.sort(norm, dim=0)[0])[idx]
    mask = (norm < threshold).to(device)
    mask = (1. - mask.float()).transpose(0, 1).view(w_shape)
  elif prune_type == 'unit':
    w_shape = list(w.size())
    w = w.view(w_shape[0], -1).transpose(0, 1)
    norm = torch.norm(w, dim=0)
    idx = int(perc * int(w.shape[1]))
    sorted_norms = torch.sort(norm)
    threshold = (sorted_norms[0])[idx]
    mask = (norm < threshold)[None, :]
    mask = mask.repeat(w.shape[0], 1).to(device)
    mask = (1. - mask.float()).transpose(0, 1).view(w_shape)
  else:
    raise NotImplementedError('Pruning type not implemented.')

  return mask


class Pruner:
  def __init__(
      self,
      model,
      optimizer,
      inital_sparsity=0,
      final_sparsity=0.5,
      start_step=0,
      end_step=0.5,
      total_steps=None,
      prune_freq=100,
      lr_scheduler=None,
      prune_type='weight',
      prune_level='layer',
      ramping=False,
      ramp_type='linear'
  ):
    self.model = model
    self.optimizer = optimizer
    self.lr_scheduler = lr_scheduler
    self.initial_weights = []
    self.initial_masks = []
    self.start_step = start_step
    self.end_step = end_step
    self.total_steps = total_steps
    self.prune_freq = prune_freq
    self.prune_type = prune_type
    self.prune_level = prune_level
    self.ramping = ramping
    self.ramp_type = ramp_type
    self.initial_sparsity = inital_sparsity
    self.final_sparsity = final_sparsity
    assert (total_steps, not None)
    if self.end_step is float:
      self.end_step = self.end_step * total_steps
      self.start_step = self.start_step * total_steps

      # Save initial data
    for layer in self.model.modules():
      if hasattr(layer, 'mask'):
        self.initial_weights.append(layer.weight.data)
        self.initial_masks.append(layer.mask.data)
    self.global_step = 0

  def GlobalPrune(self, model, prune_compute):
    weights = []
    for module in model.modules():
      if hasattr(module, 'mask'):
        weights.append(module.weight * module.mask)
    scores = torch.cat([torch.flatten(w) for w in weights])
    idx = int(prune_compute * scores.shape[0])
    norm = torch.abs(scores)
    threshold = (torch.sort(norm, dim=0)[0])[idx]
    # threshold = values[-1]
    masks = [(torch.abs(w) > threshold).float() for w in weights]
    count = 0
    for layer in model.modules():
      if hasattr(layer, 'mask'):
        layer.mask.data = masks[count]
        count += 1
    return model

  def LayerPrune(self, model, prune_compute):
    for module in model.modules():
      if hasattr(module, 'mask'):
        mask_sparsity = round(1. - np.sum(module.mask.detach().cpu().numpy())
                              / module.mask.detach().cpu().numpy().size, 2)
        if mask_sparsity < prune_compute:
          module_mask = prune(module.weight * module.mask,
                              prune_compute,
                              self.prune_type)
          module.mask.data = module_mask
    return model

  def ramp_sparsify(self, model):
    if self.start_step <= self.global_step <= self.end_step:
      if self.global_step % self.prune_freq == 0:
        if self.ramp_type == 'linear':
          rate_of_increase = (self.final_sparsity - self.initial_sparsity) / (
              self.end_step - self.start_step)
          prune_compute = self.initial_sparsity + rate_of_increase * (
              self.global_step - self.start_step)
        else:
          raise NotImplementedError('Ramping type not implemented')
        if self.prune_level == 'global':
          model = self.GlobalPrune(model, prune_compute)
        elif self.prune_level == 'layer':
          model = self.LayerPrune(model, prune_compute)
    elif self.global_step == self.end_step:
      if self.prune_level == 'global':
        model = self.GlobalPrune(model, self.final_sparsity)
      elif self.prune_level == 'layer':
        model = self.LayerPrune(model, self.final_sparsity)

    return model

  def sparsify(self, model):
    if self.global_step == self.start_step:
      if self.prune_level == 'global':
        model = self.GlobalPrune(model, self.final_sparsity)
      elif self.prune_level == 'layer':
        model = self.LayerPrune(model, self.final_sparsity)
    return model

  def step(self):
    if self.ramping:
      self.model = self.ramp_sparsify(self.model)
    else:
      self.model = self.sparsify(self.model)

    self.optimizer.step()
    if self.lr_scheduler:
      self.lr_scheduler.step()
    self.global_step += 1

  def mask_sparsity(self):
    sparsities = []
    for module in self.model.modules():
      if hasattr(module, 'mask'):
        mask = module.mask.detach().cpu().numpy()
        sparsities.append(round(1. - np.sum(mask) / mask.size, 2))
    print(sparsities)

  def reset_masks(self):
    for module in self.model.modules():
      if hasattr(module, 'mask'):
        mask = prune(module.weight, 0, self.prune_type)
        module.mask.data = mask

  def reset_weights(self):
    count = 0
    for module in self.model.modules():
      if hasattr(module, 'mask'):
        module.weight.data = self.initial_weights[count]
        count += 1

  def mask_check(self):
    all_ones = False
    zeros_and_ones = False
    mixed_mask = False
    for module in self.model.modules():
      if hasattr(module, 'mask'):
        mask = module.mask.detach().cpu().numpy()
        # mask = mask.tolist()
        if set(mask.flatten()) == {1}:
          all_ones = True
        elif set(mask.flatten()) == {1, 0}:
          zeros_and_ones = True
        else:
          mixed_mask = True
    if all_ones:
      print('Mask is all 1s')
    elif zeros_and_ones:
      print('Mask is 0s and 1s')
    elif mixed_mask:
      print('Mask is not binary')
