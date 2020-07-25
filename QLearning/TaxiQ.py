#!/usr/bin/env python3
import gym
import numpy as np
import collections
import itertools

import io
import cv2

import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter


ENV_NAME = "Taxi-v3"
GAMMA = 0.9
ALPHA = 0.2
TEST_EPISODES = 20

def get_img_from_fig(fig, dpi=180):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img

def plot_q_vals(env, qDict):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
    cm (array, shape = [n, n]): a confusion matrix of integer classes
    class_names (array, shape = [n]): String names of the integer classes
    """

    nObs = env.observation_space.n
    nAcc = env.action_space.n

    mat = np.zeros((nObs, nAcc))

    for obs in range(nObs):
        for acc in range(nAcc):
            val = qDict.get((obs,acc), np.nan)
            mat[obs, acc] = val

    figure = plt.figure(figsize=(8, 8))
    plt.imshow(mat, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marksx = np.arange(nAcc)
    tick_marksy = np.arange(nObs)
    plt.xticks(tick_marksx, tick_marksx, rotation=45)
    plt.yticks(tick_marksy, tick_marksy)

    # Use white text if squares are dark; otherwise black.
    threshold = mat.max() / 2.
    for i, j in itertools.product(range(mat.shape[0]), range(mat.shape[1])):
        color = "white" if mat[i, j] > threshold else "black"
    plt.text(j, i, mat[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    return figure

def log_confusion_matrix(env, qDict, iter):

    # Log the confusion matrix as an image summary.
    figure = plot_q_vals(env, qDict)
    cm_image = plot_to_image(figure)

    # Log the confusion matrix as an image summary.
    with file_writer_cm.as_default():
        tf.summary.image("Confusion Matrix", cm_image, step=iter)  

class Agent:
    def __init__(self):
        self.env = gym.make(ENV_NAME)
        self.state = self.env.reset()
        self.values = collections.defaultdict(float)

    def sample_env(self):
        action = self.env.action_space.sample()
        old_state = self.state
        new_state, reward, is_done, _ = self.env.step(action)
        self.state = self.env.reset() if is_done else new_state
        return (old_state, action, reward, new_state)

    def best_value_and_action(self, state):
        best_value, best_action = None, None
        for action in range(self.env.action_space.n):
            action_value = self.values[(state, action)]
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action
        return best_value, best_action

    def value_update(self, s, a, r, next_s):
        best_v, _ = self.best_value_and_action(next_s)
        new_val = r + GAMMA * best_v
        old_val = self.values[(s, a)]
        self.values[(s, a)] = old_val * (1-ALPHA) + new_val * ALPHA

    def play_episode(self, env):
        total_reward = 0.0
        state = env.reset()
        while True:
            _, action = self.best_value_and_action(state)
            new_state, reward, is_done, _ = env.step(action)
            total_reward += reward
            if is_done:
                break
            state = new_state
        return total_reward


if __name__ == "__main__":
    test_env = gym.make(ENV_NAME)
    agent = Agent()
    writer = SummaryWriter(comment="-q-learning")

    iter_no = 0
    best_reward = 0.0
    while True:
        iter_no += 1
        s, a, r, next_s = agent.sample_env()
        agent.value_update(s, a, r, next_s)
        reward = 0.0
        for _ in range(TEST_EPISODES):
            reward += agent.play_episode(test_env)
        reward /= TEST_EPISODES
        if iter_no%200 == 0:
            print("Iteration number: %i Reward: %.3f" % (iter_no, reward))
            fig = plot_q_vals(agent.env, agent.values)
            img = get_img_from_fig(fig)
            writer.add_image(str(iter_no), img, global_step=None, walltime=None, dataformats='HWC')
        writer.add_scalar("reward", reward, iter_no)
        if reward > best_reward:
            print("Best reward updated %.3f -> %.3f" % (best_reward, reward))
            best_reward = reward
        if reward > 0.80:
            print("Solved in %d iterations!" % iter_no)
            break
    writer.close()
