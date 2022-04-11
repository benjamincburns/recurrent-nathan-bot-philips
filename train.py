import numpy as np
from rlgym.envs import Match
from rlgym.utils.action_parsers import DiscreteAction
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize, VecCheckNan

from rlgym.utils.obs_builders import AdvancedObs
from rlgym.utils.state_setters import DefaultState
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition, GoalScoredCondition
from rlgym_tools.sb3_utils import SB3MultipleInstanceEnv

from rlgym_tools.extra_rewards.kickoff_reward import KickoffReward
from rlgym.utils.reward_functions.common_rewards.misc_rewards import EventReward
from rlgym.utils.reward_functions.common_rewards import VelocityPlayerToBallReward
from rlgym.utils.reward_functions.common_rewards.ball_goal_rewards import VelocityBallToGoalReward
from rlgym.utils.reward_functions import CombinedReward

if __name__ == '__main__':  # Required for multiprocessing
    frame_skip = 8          # Number of ticks to repeat an action
    half_life_seconds = 5   # Easier to conceptualize, after this many seconds the reward discount is 0.5

    fps = 120 / frame_skip
    gamma = np.exp(np.log(0.5) / (fps * half_life_seconds))  # Quick mafs
    agents_per_match = 2
    num_instances = 10
    target_steps = 100_000
    steps = target_steps // (num_instances * agents_per_match)
    batch_size = steps

    print(f"fps={fps}, gamma={gamma})")

    reward_function = CombinedReward(
        (
            VelocityPlayerToBallReward(),
            KickoffReward(),
            VelocityBallToGoalReward(),
            EventReward(
                team_goal=100.0,
                concede=-100.0,
                shot=10.0,
                save=50.0,
                demo=10.0
            ),
        ),
        (0.1, 1.0, 1.0, 1.0)
    )

    def get_match():  # Need to use a function so that each instance can call it and produce their own objects
        return Match(
            team_size=1,
            tick_skip=frame_skip,
            reward_function=reward_function,  # Simple reward since example code
            self_play=True,
            terminal_conditions=[TimeoutCondition(round(fps * 30)), GoalScoredCondition()],  # Some basic terminals
            obs_builder=AdvancedObs(),  # Not that advanced, good default
            state_setter=DefaultState(),  # Resets to kickoff position
            action_parser=DiscreteAction(),  # Discrete > Continuous don't @ me
        )

    env = SB3MultipleInstanceEnv(get_match, num_instances, force_paging=True)  # Start multiple instances, waiting 60 seconds between each
    env = VecCheckNan(env)                                  # Optional
    env = VecMonitor(env)                                   # Recommended, logs mean reward and ep_len to Tensorboard
    env = VecNormalize(env, norm_obs=False, gamma=gamma)    # Highly recommended, normalizes rewards

    try:
        model = RecurrentPPO.load(
            "models/exit_save.zip",
            env,
            custom_objects=dict(n_envs=env.num_envs, _last_obs=None),  # Need this to change number of agents
            device="cuda",
            #device="cpu",
            # force_reset=True  # Make SB3 reset the env so it doesn't think we're continuing from last state
        )

        reset_num_timestamps=False

    except:

        from torch.nn import Tanh
        policy_kwargs = dict(
            activation_fn=Tanh,
            net_arch=[512, 512, dict(pi=[256, 256, 256], vf=[256, 256, 256])],
        )

        model = RecurrentPPO(
            "MlpLstmPolicy",
            env,
            n_epochs=1,                  # PPO calls for multiple epochs
            policy_kwargs=policy_kwargs, # network architecture
            learning_rate=5e-5,          # Around this is fairly common for PPO
            ent_coef=0.01,               # From PPO Atari
            vf_coef=1.,                  # From PPO Atari
            gamma=gamma,                 # Gamma as calculated using half-life
            verbose=3,                   # Print out all the info as we're going
            batch_size=batch_size,       # Batch size as high as possible within reason
            n_steps=steps,               # Number of steps to perform before optimizing network
            tensorboard_log="logs",      # `tensorboard --logdir logs` in terminal to see graphs
            # lstm_hidden_size=256,
            # n_lstm_layers=1,
            device="cuda"
            #device="cpu"
        )

        reset_num_timestamps=True

    # Save model every so often
    # Divide by num_envs (number of agents) because callback only increments every time all agents have taken a step
    # This saves to specified folder with a specified name
    callback = CheckpointCallback(round(5_000_000 / env.num_envs), save_path="models", name_prefix="rl_model")

    while True:
        # Use reset_num_timesteps=False to keep going with same logger/checkpoints
        model.learn(25_000_000, callback=callback, reset_num_timesteps=reset_num_timestamps)
        model.save("models/exit_save")
        model.save(f"mmr_models/snapshot_{model.num_timesteps}")

