import pyautogui
import time
import numpy as np
import torch
import torch.nn as nn
import random
from collections import deque
from collections import namedtuple
from Environment import Env
import snake_engine

# 1. Properly set up the GPU device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device.type)

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class DQNAgent:
    def __init__(
            self,
            env,
            discount_factor=0.999,
            epsilon_greedy=1.0,
            epsilon_min=0.001,
            epsilon_decay=0.998,
            learning_rate=1e-3,
            max_memory_size=100000
    ):
        self.env = env
        self.state_size = env.state_size
        self.action_size = env.num_actions
        self.memory = deque(maxlen=max_memory_size)
        self.gamma = discount_factor
        self.epsilon = epsilon_greedy
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.lr = learning_rate
        self._build_nn_model()

    def _build_nn_model(self):
        # 2. Push the model to the GPU
        self.model = nn.Sequential(
            nn.Linear(self.state_size, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_size)
        ).to(device)

        self.loss_fn = nn.SmoothL1Loss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr)

    def remember(self, transition):
        self.memory.append(transition)

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)

        with torch.no_grad():
            state_tensor = state.flatten().unsqueeze(0).to(device)
            q_values = self.model(state_tensor)[0]
            return torch.argmax(q_values).item()

    def _learn(self, batch_samples):
        states = torch.stack([t.state.flatten() for t in batch_samples]).to(device)
        actions = torch.tensor([t.action for t in batch_samples], dtype=torch.long, device=device).unsqueeze(1)
        rewards = torch.tensor([t.reward for t in batch_samples], dtype=torch.float32, device=device)
        next_states = torch.stack([t.next_state.flatten() for t in batch_samples]).to(device)
        dones = torch.tensor([t.done for t in batch_samples], dtype=torch.float32, device=device)

        current_q = self.model(states).gather(1, actions).squeeze(1)

        with torch.no_grad():
            next_q = self.model(next_states).max(1)[0]
            target_q = rewards + (1.0 - dones) * self.gamma * next_q

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def _adjust_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def replay(self, batch_size):
        samples = random.sample(self.memory, batch_size)
        return self._learn(samples)

    def save_weights(self, filepath="snake_dqn.pt"):
        torch.save(self.model.state_dict(), filepath)

    def load_weights(self, filepath):
        state_dict = torch.load(filepath, map_location=device)
        self.model.load_state_dict(state_dict)


EPISODES = 50_000
batch_size = 256
init_replay_memory_size = 500


if __name__ == "__main__":
    engi = snake_engine.SnakeEngine()
    env = Env(engi)
    agent = DQNAgent(env)
    state, apple_loc, head_loc = env.reset()
    state = state.float()

    #state = np.reshape(state, [1, agent.state_size])
    best = 25
    highscore = 0
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
    total_apples,total_lengths = [], []
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
            if reward == 10:
                episode_apples+=1

            if len(agent.memory) >= batch_size:
                loss = agent.replay(batch_size)
                episode_losses.append(loss)

            if done:
                # total_rewards.append(i)
                # with open("output.txt", "a") as file:
                #     file.write(f'Episode: {e}/{EPISODES}, Total steps: {i}, Total reward: {episode_reward}, {best}\n')
                # print(f'Episode: {e}/{EPISODES}, Total steps: {i}, Total reward: {episode_reward}, {best}')
                # if episode_reward > best:
                #     best = episode_reward
                #     agent.save_weights("snake2.pt")
                break
        total_rewards.append(episode_reward)
        total_lengths.append(i+1)
        total_apples.append(episode_apples)
        avg_loss = np.mean(episode_losses) if episode_losses else 0
        #loss = agent.replay(batch_size)
        #losses.append(loss)
        losses.append(avg_loss)
        agent._adjust_epsilon()
        with open("output.txt", "a") as file:
            file.write( f"Episode: {e + 1}/{EPISODES}, "
            f"Apples: {episode_apples}, "
            f"Steps: {i + 1}, "
            f"Reward: {episode_reward}, "
            f"Epsilon: {agent.epsilon:.3f}\n"
        )
        print(
            f"Episode: {e + 1}/{EPISODES}, "
            f"Apples: {episode_apples}, "
            f"Steps: {i + 1}, "
            f"Reward: {episode_reward}, "
            f"Epsilon: {agent.epsilon:.3f}"
            f"Best: {highscore}"
        )

        if episode_reward > best:
            best = episode_reward
            agent.save_weights("snake1.pt")
