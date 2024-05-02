import mjx
from typing import Dict, List
import argparse
import os

import numpy as np
import tensorflow as tf
from tensorflow import keras

TILES_COUNT = 34

parser = argparse.ArgumentParser()
    parser.add_argument('--discard_model_path', action='store', type=str,
                        required=True)
args = parser.parse_args()
discard_model_path = args.discard_model_path

class MLAgent(mjx.Agent):
    def __init__(self):
        super().__init__()
        self.discount_factor = 0.99
        self.learning_rate = 1e-3
        self.discard_model = keras.models.load_model(discard_model_path, compile=False)
        self.observations: List[mjx.Observation]
        self.actions: List[mjx.Action]

    def act(self, obs) -> Action:
        # Using the output of the policy network, pick action stochastically
        policy = self.discard_model.predict(obs).flatten()
        action = np.random.choice(TILES_COUNT, p=policy)

        # Save the state and action
        self.observations.append(obs)
        self.actions.append(action)

        # Output the chosen action
        return action

    def discount_rewards(self, final_reward):
        discounted_returns = np.zeros_like(self.actions)
        discounted_returns[-1] = final_reward
        for t in reversed(range(len(discounted_returns) - 1)):
            discounted_returns[t] = self.discount_factor \
                                    * discounted_returns[t + 1]
        return discounted_returns

    def train_discard_model(self, all_discounted_returns):
        episode_length = len(self.observations)
        update_inputs = tf.stack(self.observations)
        expected_returns = np.zeros((episode_length, TILES_COUNT))
        for i in range(episode_length):
            expected_returns[i][self.actions[i]] = all_discounted_returns[i]
        self.discard_model.fit(update_inputs, expected_returns, epochs=1,
                               verbose=0)

    def update_discard_model(self, new_discard_model):
        self.discard_model = new_discard_model

agent = MLAgent()
env = mjx.MjxEnv()
obs_dict = env.reset()
all_discounted_returns = []
while not env.done():
    actions = {player_id: agent.act(obs) for player_id, obs in obs_dict.items()}
    all_discounted_returns.extend(env.rewards())
    all_discounted_returns -= np.mean(all_discounted_returns)
    all_discounted_returns /= np.std(all_discounted_returns)
    agent.train_discard_model(all_discounted_returns)
    obs_dict = env.step(actions)

agent.discard_model.save(os.path.join('models', 'discard_rl.h5'))

