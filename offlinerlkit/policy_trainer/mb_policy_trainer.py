import time
import os

import torch

from typing import Optional, List, Tuple
from tqdm import tqdm
from offlinerlkit.buffer import ReplayBuffer
from offlinerlkit.utils.logger import Logger
from offlinerlkit.policy import BasePolicy


# model-based policy trainer
class MBPolicyTrainer:
    def __init__(
        self,
        policy: BasePolicy,
        real_buffer: ReplayBuffer,
        fake_buffer: ReplayBuffer,
        logger: Logger,
        rollout_setting: Tuple[int, int, int],
        epoch: int = 1000,
        step_per_epoch: int = 1000,
        batch_size: int = 256,
        real_ratio: float = 0.05,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        dynamics_update_freq: int = 0,
        checkpoint_epochs: Optional[List[int]] = None,
        show_progress: bool = True,
    ) -> None:
        self.policy = policy
        self.real_buffer = real_buffer
        self.fake_buffer = fake_buffer
        self.logger = logger

        self._rollout_freq, self._rollout_batch_size, \
            self._rollout_length = rollout_setting
        self._dynamics_update_freq = dynamics_update_freq

        self._epoch = epoch
        self._step_per_epoch = step_per_epoch
        self._batch_size = batch_size
        self._real_ratio = real_ratio
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
                if num_timesteps % self._rollout_freq == 0:
                    init_obss = self.real_buffer.sample(self._rollout_batch_size)["observations"].cpu().numpy()
                    rollout_transitions, rollout_info = self.policy.rollout(init_obss, self._rollout_length)
                    self.fake_buffer.add_batch(**rollout_transitions)
                    self.logger.log(
                        "num rollout transitions: {}, reward mean: {:.4f}".\
                            format(rollout_info["num_transitions"], rollout_info["reward_mean"])
                    )
                    for _key, _value in rollout_info.items():
                        self.logger.logkv_mean("rollout_info/"+_key, _value)

                real_sample_size = int(self._batch_size * self._real_ratio)
                fake_sample_size = self._batch_size - real_sample_size
                real_batch = self.real_buffer.sample(batch_size=real_sample_size)
                fake_batch = self.fake_buffer.sample(batch_size=fake_sample_size)
                batch = {"real": real_batch, "fake": fake_batch}
                loss = self.policy.learn(batch)

                for k, v in loss.items():
                    self.logger.logkv_mean(k, v)
                
                # update the dynamics if necessary
                if 0 < self._dynamics_update_freq and (num_timesteps+1)%self._dynamics_update_freq == 0:
                    dynamics_update_info = self.policy.update_dynamics(self.real_buffer)
                    for k, v in dynamics_update_info.items():
                        self.logger.logkv_mean(k, v)
                
                num_timesteps += 1

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            self.logger.set_timestep(num_timesteps)
            self.logger.dumpkvs(exclude=["dynamics_training_progress"])
        
            # save checkpoint
            torch.save(self.policy.state_dict(), os.path.join(self.logger.checkpoint_dir, "policy.pth"))
            if e in self._checkpoint_epochs:
                checkpoint_dir = os.path.join(self.logger.checkpoint_dir, f"step_{num_timesteps}")
                os.makedirs(checkpoint_dir, exist_ok=True)
                torch.save(self.policy.state_dict(), os.path.join(checkpoint_dir, "policy.pth"))
                if self._dynamics_update_freq > 0:
                    self.policy.dynamics.save(checkpoint_dir)

        self.logger.log("total time: {:.2f}s".format(time.time() - start_time))
        torch.save(self.policy.state_dict(), os.path.join(self.logger.model_dir, "policy.pth"))
        self.policy.dynamics.save(self.logger.model_dir)
        self.logger.close()
