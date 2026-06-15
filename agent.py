import os
import numpy as np
import torch

from rl_model import RLModel


class Agent:
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space

        path = "model.pt"
        self.model = RLModel(obs_dim=observation_space.shape[0],
                             num_actions=action_space.n)
        if os.path.exists(path):
            self.model.load(path)
        self.model.eval()

    @torch.inference_mode()
    def choose_action(self, observation, reward=0.0, terminated=False, truncated=False,
                      info=None, action_mask=None):
        obs = torch.from_numpy(np.asarray(observation, dtype=np.float32))[None, :]
        logits, values = self.model(obs)

        ## TODO:
        # Compute the action index using the model forward function:
        # action_index = ...
        raise NotImplementedError()

        return action_index
