import time
import os

import torch

from typing import Optional, List
from tqdm import tqdm
from offlinerlkit.buffer import ReplayBuffer
from offlinerlkit.utils.logger import Logger
from offlinerlkit.policy import BasePolicy


# model-free policy trainer
class MFPolicyTrainer:
    def __init__(
        self,
        policy: BasePolicy,
        buffer: ReplayBuffer,
        logger: Logger,
        epoch: int = 1000,
        step_per_epoch: int = 1000,
        batch_size: int = 256,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        checkpoint_epochs: Optional[List[int]] = None,
        show_progress: bool = True,
    ) -> None:
        self.policy = policy
        self.buffer = buffer
        self.logger = logger

        self._epoch = epoch
        self._step_per_epoch = step_per_epoch
        self._batch_size = batch_size
        self.lr_scheduler = lr_scheduler
        self._checkpoint_epochs = set(checkpoint_epochs or [])
        self._show_progress = show_progress

    def train(self) -> None:
        start_time = time.time()

        num_timesteps = 0
        epochs = tqdm(
            range(1, self._epoch + 1),
            desc="Training epochs",
            disable=not self._show_progress,
        )
        for e in epochs:

            self.policy.train()

            for _ in range(self._step_per_epoch):
                batch = self.buffer.sample(self._batch_size)
                loss = self.policy.learn(batch)

                for k, v in loss.items():
                    self.logger.logkv_mean(k, v)
                
                num_timesteps += 1

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            self.logger.set_timestep(num_timesteps)
            self.logger.dumpkvs()
        
            # save checkpoint
            torch.save(self.policy.state_dict(), os.path.join(self.logger.checkpoint_dir, "policy.pth"))
            if e in self._checkpoint_epochs:
                checkpoint_dir = os.path.join(self.logger.checkpoint_dir, f"step_{num_timesteps}")
                os.makedirs(checkpoint_dir, exist_ok=True)
                torch.save(self.policy.state_dict(), os.path.join(checkpoint_dir, "policy.pth"))

        self.logger.log("total time: {:.2f}s".format(time.time() - start_time))
        torch.save(self.policy.state_dict(), os.path.join(self.logger.model_dir, "policy.pth"))
        self.logger.close()
