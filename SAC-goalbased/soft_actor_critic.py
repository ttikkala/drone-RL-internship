#########

# From co-adaptation code, using rlkit version 0.2.1

#########


from rlkit.torch.sac.policies import TanhGaussianPolicy
# from rlkit.torch.sac.sac import SoftActorCritic
from rlkit.torch.networks import FlattenMlp
import numpy as np
from rl_algorithm import RL_algorithm
from rlkit.torch.sac.sac import SACTrainer
import rlkit.torch.pytorch_util as ptu
import torch
import utils
import matplotlib.pyplot as plt
import matplotlib

class SoftActorCritic(RL_algorithm):

    def __init__(self, 
                 replay, 
                 networks):
        """ Basically a wrapper class for SAC from rlkit.

        Args:
            replay: Replay buffer
            networks: dict containing the networks.

        """
        super().__init__(replay, 
                         networks
                        )

        self._qf1 = networks['qf1']
        self._qf2 = networks['qf2']
        self._qf1_target = networks['qf1_target']
        self._qf2_target = networks['qf2_target']
        self._policy = networks['policy']

        self._batch_size = 64
        self._nmbr_updates = 1000

        self._algorithm = SACTrainer(
            policy=self._policy,
            qf1=self._qf1,
            qf2=self._qf2,
            target_qf1=self._qf1_target,
            target_qf2=self._qf2_target,
            use_automatic_entropy_tuning = False,
            alpha=0.01,
        )


    def episode_init(self):
        """ Initializations to be done before the first episode.

        In this case basically creates a fresh instance of SAC for the
        individual networks and copies the values of the target network.
        """
        self._algorithm = SACTrainer(
            policy=self._policy,
            qf1=self._qf1,
            qf2=self._qf2,
            target_qf1=self._qf1_target,
            target_qf2=self._qf2_target,
            use_automatic_entropy_tuning = False,
            # alt_alpha = self._alt_alpha,
        )

    def single_train_step(self):
        """ 
        A single training step.
        """

        for i in range(self._nmbr_updates):
            batch = self._replay.random_batch(self._batch_size)
            self._algorithm.train(batch)
            # if i == self._nmbr_updates - 1:
            #     print('Double update on one batch')
            #     print(self._algorithm.get_diagnostics())
            #     self._algorithm.train(batch)



    @staticmethod
    def _create_networks(obs_dim, action_dim):
        """ Creates all networks necessary for SAC.

        These networks have to be created before instantiating this class and
        used in the constructor.

        TODO: Maybe this should be reworked one day...

        Args:
            obs_dim: Dimension of the observation space.
            action_dim: Dimension of the action space.

        Returns:
            A dictonary which contains the networks.
        """
        obs_dim = obs_dim
        action_dim = action_dim
        net_size = 256
        hidden_sizes = [256] * 3
        qf1 = FlattenMlp(
            hidden_sizes=hidden_sizes,
            input_size=obs_dim + action_dim,
            output_size=1,
        ).to(device=ptu.device)
        qf2 = FlattenMlp(
            hidden_sizes=hidden_sizes,
            input_size=obs_dim + action_dim,
            output_size=1,
        ).to(device=ptu.device)
        qf1_target = FlattenMlp(
            hidden_sizes=hidden_sizes,
            input_size=obs_dim + action_dim,
            output_size=1,
        ).to(device=ptu.device)
        qf2_target = FlattenMlp(
            hidden_sizes=hidden_sizes,
            input_size=obs_dim + action_dim,
            output_size=1,
        ).to(device=ptu.device)
        policy = TanhGaussianPolicy(
            hidden_sizes=hidden_sizes,
            obs_dim=obs_dim,
            action_dim=action_dim,
        ).to(device=ptu.device)

        # print('Device: ', ptu.device)

        clip_value = 1.0
        for p in qf1.parameters():
            p.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))
        for p in qf2.parameters():
            p.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))
        for p in policy.parameters():
            p.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))

        return {'qf1' : qf1, 'qf2' : qf2, 'qf1_target' : qf1_target, 'qf2_target' : qf2_target, 'policy' : policy}

    @staticmethod
    def get_q_network(networks):
        """ Returns the q network from a dict of networks.

        This method extracts the q-network from the dictonary of networks
        created by the function create_networks.

        Args:
            networks: Dict containing the networks.

        Returns:
            The q-network as torch object.
        """
        return networks['qf1']

    @staticmethod
    def get_policy_network(networks):
        """ Returns the policy network from a dict of networks.

        This method extracts the policy network from the dictonary of networks
        created by the function create_networks.

        Args:
            networks: Dict containing the networks.

        Returns:
            The policy network as torch object.
        """
        return networks['policy']
    

