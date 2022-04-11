import numpy as np
from sb3_contrib import RecurrentPPO
import pathlib
from parsers.discrete_act import DiscreteAction


class Agent:
    def __init__(self):
        _path = pathlib.Path(__file__).parent.resolve()
        custom_objects = {
            "lr_schedule": 0.000001,
            "clip_range": .02,
            "device": "cpu",
            "n_envs": 1,
        }
        
        self.actor = RecurrentPPO.load(str(_path) + '/exit_save_20220409', device='cpu', custom_objects=custom_objects)
        self.parser = DiscreteAction()
        self.lstm_states = None


    def act(self, obs):
        action, lstm_states = self.actor.predict(obs, state=self.lstm_states, deterministic=True)
        self.lstm_states = lstm_states
        x = self.parser.parse_actions(action[0], state)

        return x[0]

if __name__ == "__main__":
    print("You're doing it wrong.")
