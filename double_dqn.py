import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from Environment import Env
from animate import Animate
import time

class TimeCapsule:
    def __init__(self, max_memories=100000, input_state_size=259):
        self.max_memories = max_memories
        self.input_state_size = input_state_size
        self.current_memories_counter = 0

        self.state_memory = torch.zeros((self.max_memories, input_state_size), dtype=torch.float32) #state
        self.next_state_memory = torch.zeros((self.max_memories, input_state_size), dtype=torch.float32) #next state
        self.action_memory = torch.zeros((self.max_memories,), dtype=torch.long) #action during this state
        self.reward_memory = torch.zeros((self.max_memories,), dtype=torch.float32) #reward provided at state
        self.terminal_memory = torch.zeros((self.max_memories,), dtype=torch.bool) #did the game end at this state

    def add_memory(self, state, next_state, action, reward, terminal):
        index = self.current_memories_counter % self.max_memories

        self.state_memory[index] = state.detach().clone().to(dtype=self.state_memory.dtype)
        self.next_state_memory[index] = next_state.detach().clone().to(dtype=self.state_memory.dtype)
        self.action_memory[index] = torch.tensor(action, dtype=self.action_memory.dtype)
        self.reward_memory[index] = torch.tensor(reward, dtype=self.reward_memory.dtype)
        self.terminal_memory[index] = torch.tensor(terminal, dtype=self.terminal_memory.dtype)

        self.current_memories_counter += 1

    def access_memories(self, batch_size, device="cpu"):
        total_memories = min(self.current_memories_counter, self.max_memories)
        if total_memories < batch_size:
            return None

        rand_sample_index = torch.randint(0,total_memories,(batch_size,)) #np.random.choice(np.arange(total_memories), size=batch_size, replace=False)
        #rand_sample_index = torch.tensor(rand_sample_index, dtype=torch.long)

        batch = {"states": self.state_memory[rand_sample_index].to(device),
                 "next_states": self.next_state_memory[rand_sample_index].to(device),
                 "actions": self.action_memory[rand_sample_index].to(device),
                 "rewards": self.reward_memory[rand_sample_index].to(device),
                 "terminal": self.terminal_memory[rand_sample_index].to(device)}
        return batch

class DQN(nn.Module):
    def __init__(self, input_state_size=259, num_actions=5, hidden_features=128, fc1_dim=512, fc2_dim=256, fc3_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(input_state_size, fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.fc3 = nn.Linear(fc2_dim, fc3_dim)
        self.fc4 = nn.Linear(fc3_dim, num_actions)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class Agent:
    def __init__(self,
                 max_memories=100000,
                 discount_factor=0.99,
                 learning_rate=0.001,
                 input_state_size=259,
                 num_actions=5,
                 hidden_features=128,
                 epsilon=1.0,
                 epsilon_decay=0.9995,
                 min_epsilon=0.05,
                 device="cpu"
        ):
        self.max_memories = max_memories
        self.discount_factor = discount_factor
        self.lr = learning_rate
        self.state_size = input_state_size
        self.num_actions = num_actions
        self.action_space = np.arange(self.num_actions)
        self.hidden_features = hidden_features
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.device = device

        self.DQN = DQN(self.state_size, self.num_actions, self.hidden_features).to(self.device)

        self.DQN_NEXT = DQN(self.state_size, self.num_actions, self.hidden_features).to(self.device)
        self.DQN_NEXT.load_state_dict(self.DQN.state_dict())
        self.DQN_NEXT.eval()

        self.optimizer = optim.Adam(self.DQN.parameters(), lr=self.lr)
        self.loss_fn =  nn.SmoothL1Loss()

        self.timecapsule = TimeCapsule(self.max_memories,self.state_size)

    def select_action(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, device=self.device)

        if state.dim() == 1:
            state = state.unsqueeze(0)

        assert state.shape[-1] == self.state_size

        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            self.DQN.eval()
            with torch.no_grad():
                Q = self.DQN(state)
            action = torch.argmax(Q).item()
            self.DQN.train()

        return action

    def update_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def inference(self, state, device="cpu"):
        self.DQN = self.DQN.to(device)
        self.DQN.eval()

        with torch.no_grad():
            Q = self.DQN(state)

        return torch.argmax(Q).item()

    def update_target_network(self):
        self.DQN_NEXT.load_state_dict(self.DQN.state_dict())

    #Double Deep Q Network
    def train_step(self, batch_size):
        batch = self.timecapsule.access_memories(batch_size, self.device)

        if batch is None:
            return None

        self.DQN.train()
        q_estimate = self.DQN(batch["states"])
        q_estimate = torch.gather(q_estimate, index=batch["actions"].unsqueeze(-1),dim=-1).squeeze(-1)

        with torch.no_grad():
            next_q_values = self.DQN(batch["next_states"])
            next_actions = torch.argmax(next_q_values,dim=-1, keepdim=True)
            next_q_values_target = self.DQN_NEXT(batch["next_states"])
            max_q_next_estimate = next_q_values_target.gather(dim=-1,index=next_actions).squeeze(-1)

        td_target = batch["rewards"] + self.discount_factor * max_q_next_estimate * (~batch["terminal"])
        loss = self.loss_fn(td_target, q_estimate)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.DQN.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.update_epsilon()

    # Deep Q Network
    # def train_step(self, batch_size):
    #     batch = self.timecapsule.access_memories(batch_size, self.device)
    #
    #     if batch is None:
    #         return None
    #
    #     Q_estimate = self.DQN(batch["states"])
    #     Q_estimate = torch.gather(Q_estimate, index=batch["actions"].unsqueeze(-1), dim=-1).squeeze(-1)
    #
    #     with torch.no_grad():
    #         Q_next_estimate = self.DQN_NEXT(batch["next_states"])
    #
    #     max_Q_next_estimate = torch.max(Q_next_estimate, dim=-1).values
    #
    #     td_target = batch["rewards"] + self.discount_factor * max_Q_next_estimate * (~batch["terminal"])
    #
    #     loss = self.loss_fn(td_target, Q_estimate)
    #     loss.backward()
    #     self.optimizer.step()
    #     self.optimizer.zero_grad()
    #     self.update_epsilon()

    def save_weights(self, filepath="snake_dqn.pt"):
        torch.save(self.DQN.state_dict(), filepath)

    def load_weights(self, filepath):
        state_dict = torch.load(filepath, map_location=self.device)
        self.DQN.load_state_dict(state_dict)
        self.DQN.eval()

def trainer(env,
            num_games=25000,
            update_target_freq=25,
            min_reward=200000, #reward for game to be counted as being successful
            game_tolerance=10, #how many successful games in a row before ending
            max_memories=100000,
            discount_factor=0.95,
            learning_rate=0.001,
            batch_size=64,
            state_size=259,
            num_actions=5,
            hidden_features=128,
            epsilon=1.0,
            epsilon_decay=0.99999,
            min_epsilon=0.01,
            log_freq=1, #how often to log values
            running_avg_steps=25,
            device="cpu",
            load_weights=False
            ):
    agent = Agent(max_memories=max_memories, discount_factor=discount_factor, learning_rate=learning_rate,
                  input_state_size=state_size, num_actions=num_actions, hidden_features=hidden_features,
                  epsilon=epsilon, epsilon_decay=epsilon_decay, min_epsilon=min_epsilon, device=device)


    ending_tol = 0
    log = {"scores": [], "running_avg_scores": []}
    best_score = -200
    apple_reward = 200
    game = []
    cur_game = []
    if load_weights:
        agent.load_weights("snake_dqn.pt")

    file = open("output.txt", 'a')
    try:
        for i in range(num_games):
            score = 0
            steps = 0

            state, apple, head = env.reset()

            done = False
            apples_consumed = 0
            cur_game.clear()

            while not done:
                action = agent.select_action(state)
                next_state, reward, done, apple, head = env.step(action, apple, head)
                score+=reward

                cur_game.append(next_state.squeeze(0).reshape(1,state_size))

                if reward == apple_reward:
                    apples_consumed += 1

                agent.timecapsule.add_memory(state,next_state,action,reward,done)

                agent.train_step(batch_size)
                if steps % update_target_freq == 0:
                    agent.update_target_network()
                steps += 1

                state = next_state

            cur_game.append(state.squeeze(0).reshape(1,state_size))
            cur_game.append(i)

            log["scores"].append(score)
            running_avg_scores = np.mean(log["scores"][-running_avg_steps:])
            log["running_avg_scores"].append(running_avg_scores)

            if score > best_score:
                best_score = score
                game = cur_game.copy()
                agent.save_weights()

            if i % log_freq == 0:
                file.write(f"Game: {i} "
                      f"| Score: {score} "
                      f"| Moving Avg Scores: {running_avg_scores: .2f} "
                      f"| Epsilon : {agent.epsilon: .5f} "
                      f"| Steps: {steps} "
                      f"| Apples : {apples_consumed}\n")

                print(f"Game: {i} "
                      f"| Score: {score} "
                      f"| Moving Avg Scores: {running_avg_scores: .2f} "
                      f"| Epsilon : {agent.epsilon: .5f} "
                      f"| Steps: {steps} "
                      f"| Apples : {apples_consumed}")

            if score > min_reward:
                ending_tol += 1
                if ending_tol == game_tolerance:
                    break
            else:
                ending_tol = 0
    except KeyboardInterrupt:
        file.close()
        return agent, log, game, game

    print("done")
    file.close()
    return agent, log, game, cur_game


if __name__ == "__main__":
    env = Env()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    agent, log, highest_scoring_game, last_game = trainer(env, device=device, load_weights=False)
    best_game = []
    recent_game = []

    for state in highest_scoring_game:
        best_game.append(state)

    for state in last_game:
        recent_game.append(state)

    anim1 = Animate(interval=200, game=recent_game[-1])
    anim1.add_frames(recent_game[:-1])
    anim1.show(save_path="recent_snake.gif")

    anim2 = Animate(interval=200,game=best_game[-1])
    anim2.add_frames(best_game[:-1])
    anim2.show(save_path="snake.gif",display=False)

    anim1.clear()
    anim2.clear()







