import copy
import glob
import os
from abc import abstractmethod, ABC
from typing import Callable

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from utils.export import create_video


class Agent(ABC):

    TORCH_MODULE_NAME_TEMPLATE = 'env_{}_model_{}_ep_{}_seed_{}.tar'

    def __init__(self, name: str, make_env_fn: Callable, make_env_kwargs: dict, gamma: float,
                 seed: int, params_out_path: str, video_out_path: str):
        # Name
        self.name = name
        self.env_name = make_env_kwargs['env_name']

        # Environment Factory
        self.make_env_fn = make_env_fn
        self.make_env_kwargs = make_env_kwargs
        self.gamma = gamma

        self.seed = seed
        self.params_out_path = params_out_path
        self.video_out_path = video_out_path
        self.checkpoint_paths = {}
        self.replay_model = None
        self.stats = None

    @abstractmethod
    def evaluate(self, model: nn.Module, env: gym.Env, n_episodes: int):
        pass

    @abstractmethod
    def render(self, model: nn.Module, env: gym.Env) -> list:
        pass

    def get_cleaned_checkpoints(self, n_checkpoints: int = 5):
        """
        Make dictionary[episode number(int) : module path(str)] which match name, env_name, seed of the current agent.
        Only uniformly spaced n_checkpoints numbers of module path are stored in dict, and other modules are removed.
        :param n_checkpoints: number of checkpoints to save.
        """
        self.checkpoint_paths.clear()

        paths: list[str] = glob.glob(os.path.join(self.params_out_path, '*.tar'))
        paths_dic: dict[int, str] = {}
        for path in paths:
            path_token: list[str] = path.split('_')
            assert len(path_token) == len(self.TORCH_MODULE_NAME_TEMPLATE.split('_'))
            if int(path_token[-1].split('.')[0]) == self.seed and \
                    path_token[1] == self.env_name and \
                    path_token[3] == self.name:
                paths_dic[int(path_token[-3])] = path
        assert len(paths_dic) >= n_checkpoints
        last_ep: int = max(paths_dic.keys())
        checkpoint_idxs = np.linspace(0, last_ep, n_checkpoints, endpoint=True, dtype=np.int32)

        for idx, path in paths_dic.items():
            if idx in checkpoint_idxs:
                self.checkpoint_paths[idx] = path
            else:
                os.unlink(path)

        self.checkpoint_paths = copy.deepcopy(self.checkpoint_paths)

    def demo_last(self, n_episodes: int = 3):
        assert self.checkpoint_paths  # checkpoint_paths must not empty
        assert self.replay_model is not None

        env = self.make_env_fn(**self.make_env_kwargs)
        last_ep = max(self.checkpoint_paths.keys())
        self.replay_model.load_state_dict(torch.load(self.checkpoint_paths[last_ep], weights_only=True))

        for i in range(n_episodes):
            frames = self.render(self.replay_model, env)
            create_video(
                frames,
                env.metadata['render_fps'],
                os.path.join(
                    self.video_out_path,
                    'env_{}_model_{}_seed_{}_trial_{}_last'.format(self.env_name, self.name, self.seed, i)
                )
            )

        env.close()
        del env

    def demo_progression(self):
        assert self.checkpoint_paths  # checkpoint_paths must not empty
        assert self.replay_model is not None

        env = self.make_env_fn(**self.make_env_kwargs)

        for i in sorted(self.checkpoint_paths.keys()):
            self.replay_model.load_state_dict(torch.load(self.checkpoint_paths[i], weights_only=True))
            frames = self.render(self.replay_model, env)
            create_video(
                frames,
                env.metadata['render_fps'],
                os.path.join(
                    self.video_out_path,
                    'env_{}_model_{}_ep_{}_seed_{}'.format(self.env_name, self.name, i, self.seed)
                )
            )

        env.close()
        del env

    def save_checkpoint(self, episode_idx: int, model: nn.Module):
        torch.save(
            model.state_dict(),
            os.path.join(
                self.params_out_path,
                self.TORCH_MODULE_NAME_TEMPLATE.format(self.env_name, self.name, episode_idx, self.seed)
            ),
        )
