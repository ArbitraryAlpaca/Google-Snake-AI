import pyautogui
import time
import numpy as np
import torch
import torch.nn as nn
import random
from collections import deque
from collections import namedtuple
from Environment import Env

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class DQNAgent:
    def __init__(
            self, env, discount_factor=0.999, epsilon_greedy=1.0,
            epsilon_min=0.01, epsilon_decay=0.9995, learning_rate=1e-3,
            max_memory_size=100000
    ):
        self.env = env

        self.state_size = env.state_size
        self.action_size = env.num_actions
        self.memory = deque(maxlen=max_memory_size)
        self.gamma = discount_factor
        self.epsilon = epsilon_greedy #1.0
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.lr = learning_rate
        self._build_nn_model()

    def _build_nn_model(self):
        self.model = nn.Sequential(nn.Linear(self.state_size, 128),
                                   nn.ReLU(),
                                   nn.Linear(128, 128),
                                   nn.ReLU(),
                                   nn.Linear(128, 64),
                                   nn.ReLU(),
                                   nn.Linear(64, self.action_size)
                                   )
        self.loss_fn = nn.SmoothL1Loss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), self.lr
        )

    def remember(self, transition):
        self.memory.append(transition)

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        with torch.no_grad():
            q_values = self.model(state)[0]
            return torch.argmax(q_values).item()

    def _learn(self, batch_samples):
        batch_states, batch_targets = [], []

        for transition in batch_samples:
            s, a, r, next_s, done = transition
            s_flat = s.flatten()
            next_s_flat = next_s.flatten()

            with torch.no_grad():
                if done:
                    target = r
                else:
                    pred = self.model(next_s_flat.unsqueeze(0))[0]
                    target = r + self.gamma * pred.max().item()

            target_all = self.model(s_flat.unsqueeze(0))[0].clone()
            target_all[a] = target

            batch_states.append(s_flat)
            batch_targets.append(target_all)

        #self._adjust_epsilon()

        states_tensor = torch.stack(batch_states)  # Shape: (batch_size, 255)
        targets_tensor = torch.stack(batch_targets)  # Shape: (batch_size, action_size)

        self.optimizer.zero_grad()
        pred = self.model(states_tensor)
        loss = self.loss_fn(pred, targets_tensor)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def _adjust_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon*self.epsilon_decay)
    def replay(self, batch_size):
        samples = random.sample(self.memory, batch_size)
        return self._learn(samples)

    def save_weights(self, filepath="snake_dqn.pt"):
        torch.save(self.model.state_dict(), filepath)

    def load_weights(self, filepath):
        state_dict = torch.load(filepath)
        self.model.load_state_dict(state_dict)


EPISODES = 5000
batch_size = 16
init_replay_memory_size = 500


if __name__ == "__main__":
    #time.sleep(2)
    file = open("output.txt", 'a')
    env = Env()
    agent = DQNAgent(env)
    state, apple_loc, head_loc = env.reset()
    state = state.float()
    #state = np.reshape(state, [1, agent.state_size])
    best = -1
    #agent.load_weights("snake.pt")
    for i in range(init_replay_memory_size):
        action = agent.choose_action(state)
        next_state, reward, done, apple_loc, head_loc = env.step(action, apple_loc, head_loc)
        next_state = next_state.float()
        #next_state = np.reshape(next_state, [1, agent.state_size])
        agent.remember(Transition(state, action, reward, next_state, done))
        if done:
            state, apple_loc, head_loc = env.reset()
            state = state.float()
            #state = np.reshape(state, [1, agent.state_size])
        else:
            state = next_state

    total_rewards, losses = [], []
    total_apples,total_lengths = [],[]
    for e in range(EPISODES):
        state, apple_loc, head_loc = env.reset()
        state = state.float()
        #state = np.reshape(state, [1, agent.state_size])
        episode_reward = 0
        episode_apples = 0
        episode_losses = []
        for i in range(500):
            action = agent.choose_action(state)
            next_state, reward, done, apple_loc, head_loc = env.step(action, apple_loc, head_loc)
            next_state = next_state.float()
            #next_state = np.reshape(next_state, [1, agent.state_size])
            agent.remember(Transition(state, action, reward, next_state, done))
            state = next_state
            episode_reward += reward
            if reward == 200:
                episode_apples+=1

            if len(agent.memory) >= batch_size:
                loss = agent.replay(batch_size)
                episode_losses.append(loss)

            if done:
                break
        total_rewards.append(episode_reward)
        total_lengths.append(i+1)
        total_apples.append(episode_apples)
        avg_loss = np.mean(episode_losses) if episode_losses else 0
        #loss = agent.replay(batch_size)
        #losses.append(loss)
        losses.append(avg_loss)
        agent._adjust_epsilon()
        file.write( f"Episode: {e + 1}/{EPISODES}, "
                    f"Apples: {episode_apples}, "
                    f"Steps: {i + 1}, "
                    f"Reward: {episode_reward}, "
                    f"Epsilon: {agent.epsilon:.3f}\n")
        print(
            f"Episode: {e + 1}/{EPISODES}, "
            f"Apples: {episode_apples}, "
            f"Steps: {i + 1}, "
            f"Reward: {episode_reward}, "
            f"Epsilon: {agent.epsilon:.3f}"
        )

        if episode_reward > best:
            best = episode_reward
            agent.save_weights("snake2.pt")
    file.close()






# class Agent:
#     def run(self, is_training=True):
#         time.sleep(5)
#         i = 0
#         prev_state = np.array([-1])
#         action = 0
#         apple_loc = (-1, -1)
#         while True:
#             reward, apple_loc, prev_state = Env.step(i,action,apple_loc)
#
#             with open(fr".\Boards\board{i}.txt","w") as file:
#                 file.write(f"{str(prev_state)}\n {str(apple_loc)}\n {str(reward)}")
#
#             if prev_state.size == 0:
#                 print("Ending")
#                 return
#             #current_board = torch.tensor(prev_state, dtype=torch.float32).flatten().unsqueeze(0)
#             i += 1
#
#np.save('myarray.npy', image)
#image = np.load('myarray.npy',mmap_mode=None)
