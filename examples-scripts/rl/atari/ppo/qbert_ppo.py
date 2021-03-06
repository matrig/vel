import torch
import torch.optim as optim

from vel.rl.metrics import EpisodeRewardMetric
from vel.storage.streaming.stdout import StdoutStreaming
from vel.util.random import set_seed
from vel.api.info import TrainingInfo, EpochInfo

from vel.modules.input.image_to_tensor import ImageToTensorFactory
from vel.rl.env.classic_atari import ClassicAtariEnv
from vel.rl.vecenv.subproc import SubprocVecEnvWrapper
from vel.rl.models.stochastic_policy_model import StochasticPolicyModelFactory
from vel.rl.models.backbone.nature_cnn import NatureCnnFactory

from vel.rl.reinforcers.on_policy_iteration_reinforcer import (
    OnPolicyIterationReinforcer, OnPolicyIterationReinforcerSettings
)

from vel.rl.algo.policy_gradient.ppo import PpoPolicyGradient
from vel.rl.env_roller.step_env_roller import StepEnvRoller
from vel.rl.commands.rl_train_command import FrameTracker

from vel.schedules.linear import LinearSchedule


def qbert_ppo():
    device = torch.device('cuda:0')
    seed = 1001

    # Set random seed in python std lib, numpy and pytorch
    set_seed(seed)

    # Create 16 environments evaluated in parallel in sub processess with all usual DeepMind wrappers
    # These are just helper functions for that
    vec_env = SubprocVecEnvWrapper(
        ClassicAtariEnv('QbertNoFrameskip-v4'), frame_history=4
    ).instantiate(parallel_envs=8, seed=seed)

    # Again, use a helper to create a model
    # But because model is owned by the reinforcer, model should not be accessed using this variable
    # but from reinforcer.model property
    model = StochasticPolicyModelFactory(
        input_block=ImageToTensorFactory(),
        backbone=NatureCnnFactory(input_width=84, input_height=84, input_channels=4)
    ).instantiate(action_space=vec_env.action_space)

    # Set schedule for gradient clipping.
    cliprange = LinearSchedule(
        initial_value=0.1,
        final_value=0.0
    )

    # Reinforcer - an object managing the learning process
    reinforcer = OnPolicyIterationReinforcer(
        device=device,
        settings=OnPolicyIterationReinforcerSettings(
            batch_size=256,
            experience_replay=4,
            number_of_steps=128
        ),
        model=model,
        algo=PpoPolicyGradient(
            entropy_coefficient=0.01,
            value_coefficient=0.5,
            max_grad_norm=0.5,
            discount_factor=0.99,
            gae_lambda=0.95,
            cliprange=cliprange
        ),
        env_roller=StepEnvRoller(
            environment=vec_env,
            device=device,
        )
    )

    # Model optimizer
    optimizer = optim.Adam(reinforcer.model.parameters(), lr=2.5e-4, eps=1.0e-5)

    # Overall information store for training information
    training_info = TrainingInfo(
        metrics=[
            EpisodeRewardMetric('episode_rewards'),  # Calculate average reward from episode
        ],
        callbacks=[
            StdoutStreaming(),   # Print live metrics every epoch to standard output
            FrameTracker(1.1e7)      # We need frame tracker to track the progress of learning
        ]
    )

    # A bit of training initialization bookkeeping...
    training_info.initialize()
    reinforcer.initialize_training(training_info)
    training_info.on_train_begin()

    # Let's make 10 batches per epoch to average metrics nicely
    # Rollout size is 8 environments times 128 steps
    num_epochs = int(1.1e7 / (128 * 8) / 10)

    # Normal handrolled training loop
    for i in range(1, num_epochs+1):
        epoch_info = EpochInfo(
            training_info=training_info,
            global_epoch_idx=i,
            batches_per_epoch=10,
            optimizer=optimizer
        )

        reinforcer.train_epoch(epoch_info)

    training_info.on_train_end()


if __name__ == '__main__':
    qbert_ppo()
