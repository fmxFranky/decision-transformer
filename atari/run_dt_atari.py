import logging
import os

import hydra
import numpy as np
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset

from create_dataset import create_dataset
from mingpt.model_atari import GPT, GPTConfig
from mingpt.trainer_atari import Trainer, TrainerConfig
from mingpt.utils import set_seed


class StateActionReturnDataset(Dataset):

  def __init__(self, data, block_size, actions, done_idxs, rtgs, timesteps):
    self.block_size = block_size
    self.vocab_size = max(actions) + 1
    self.data = data
    self.actions = actions
    self.done_idxs = done_idxs
    self.rtgs = rtgs
    self.timesteps = timesteps

  def __len__(self):
    return len(self.data) - self.block_size

  def __getitem__(self, idx):
    block_size = self.block_size // 3
    done_idx = idx + block_size
    for i in self.done_idxs:
      if i > idx:  # first done_idx greater than idx
        done_idx = min(int(i), done_idx)
        break
    idx = done_idx - block_size
    states = torch.tensor(np.array(self.data[idx:done_idx]),
                          dtype=torch.float32).reshape(
                              block_size, -1)  # (block_size, 4*84*84)
    states = states / 255.
    actions = torch.tensor(self.actions[idx:done_idx],
                           dtype=torch.long).unsqueeze(1)  # (block_size, 1)
    rtgs = torch.tensor(self.rtgs[idx:done_idx],
                        dtype=torch.float32).unsqueeze(1)
    timesteps = torch.tensor(self.timesteps[idx:idx + 1],
                             dtype=torch.int64).unsqueeze(1)

    return states, actions, rtgs, timesteps


@hydra.main(config_path="./", config_name="config")
def run(cfg: DictConfig) -> None:
  set_seed(cfg.seed)

  if cfg.save_dir is None:
    root_dir = os.getcwd(),
  elif os.path.isabs(cfg.save_dir):
    root_dir = cfg.save_dir
  else:
    root_dir = os.path.join(hydra.utils.get_original_cwd(),
                            os.path.relpath(cfg.save_dir))

  if cfg.use_wandb:
    wandb.init(project=cfg.wandb_project_name,
               name=cfg.run_name,
               config=cfg._content)

  if not os.path.isabs(cfg.data_dir_prefix):
    cfg.data_dir_prefix = os.path.join(hydra.utils.get_original_cwd(),
                                       os.path.relpath(cfg.data_dir_prefix))

  obss, actions, returns, done_idxs, rtgs, timesteps, min_rtg, max_rtg = create_dataset(
      cfg.num_buffers, cfg.num_steps, cfg.game, cfg.data_dir_prefix,
      cfg.trajectories_per_buffer)

  # set up logging
  logging.basicConfig(
      format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
      datefmt="%m/%d/%Y %H:%M:%S",
      level=logging.INFO,
  )

  train_dataset = StateActionReturnDataset(obss, cfg.context_length * 3,
                                           actions, done_idxs, rtgs, timesteps)

  mconf = GPTConfig(train_dataset.vocab_size,
                    train_dataset.block_size,
                    n_layer=cfg.n_layer,
                    n_head=cfg.n_head,
                    n_embd=cfg.n_embd,
                    model_type=cfg.model_type,
                    embd_pdrop=cfg.embd_pdrop,
                    resid_pdrop=cfg.resid_pdrop,
                    attn_pdrop=cfg.attn_pdrop,
                    max_timestep=max(timesteps),
                    n_regmask=cfg.n_regmask,
                    mask_mode=cfg.mask_mode,
                    traj_pmask=cfg.traj_pmask,
                    reg_coef=cfg.reg_coef)

  model = GPT(mconf)

  # initialize a trainer instance and kick off training
  epochs = cfg.epochs
  final_tokens = 2 * len(train_dataset) * cfg.context_length * 3
  tconf = TrainerConfig(max_epochs=epochs,
                        batch_size=cfg.batch_size,
                        learning_rate=6e-4,
                        lr_decay=True,
                        warmup_tokens=512 * 20,
                        final_tokens=final_tokens,
                        num_workers=os.cpu_count(),
                        seed=cfg.seed,
                        model_type=cfg.model_type,
                        game=cfg.game,
                        max_timestep=max(timesteps),
                        use_wandb=cfg.use_wandb,
                        save_results=cfg.save_results,
                        save_ckpt=cfg.save_ckpt,
                        root_dir=root_dir,
                        save_prefix=cfg.run_name,
                        prog_bar=cfg.prog_bar,
                        log_interval=cfg.log_interval)
  trainer = Trainer(model, train_dataset, None, tconf)

  trainer.train()


if __name__ == "__main__":
  run()
