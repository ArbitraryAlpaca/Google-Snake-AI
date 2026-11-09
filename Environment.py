import numpy as np
import torch
import time

from snake_engine import SnakeEngine


class Env:
    engine = None
    num_actions = 5
    state_size = 259

    def __init__(self, engine: SnakeEngine):
        self.engine = engine

    def reset(self):
        self.engine.reset()
        board, apple, head = self.engine.get_state()
        direction = np.array([0, 0, 0, 1], dtype=np.float32)
        state = np.concatenate((board.flatten(), direction))
        return torch.tensor(state, dtype=torch.float32).unsqueeze(0), apple, head

    def step(self, action, prev_apple, prev_head=(7, 3), i=''):
        done = False

        directions = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]
        self.engine.change_direction(directions[action])
        self.engine.step()

        cur_state, cur_apple, snake_head = self.engine.get_state()
        if self.engine.game_over:
            reward = -100
            done = True
            cur_state = np.zeros((15, 17), dtype='float32')
        elif prev_apple != (-1, -1) and cur_apple != prev_apple:
            reward = 500
        else:
            reward = 10
            old_dist = abs(prev_head[0] - prev_apple[0]) + abs(prev_head[1] - prev_apple[1])
            new_dist = abs(snake_head[0] - cur_apple[0]) + abs(snake_head[1] - cur_apple[1])
            if new_dist < old_dist:
                reward += 1
            elif new_dist > old_dist:
                reward -= 20
        direction_map = {
            (-1, 0): [1, 0, 0, 0],  # up
            (1, 0): [0, 1, 0, 0],  # down
            (0, -1): [0, 0, 1, 0],  # left
            (0, 1): [0, 0, 0, 1]  # right
        }
        movement = (
            snake_head[0] - prev_head[0],
            snake_head[1] - prev_head[1]
        )
        direction = direction_map.get(movement, [0, 0, 0, 1])
        direction = np.array(direction, dtype='float32')
        cur_state = np.concatenate((cur_state.flatten(), direction.flatten()))
        tens = torch.tensor(cur_state, dtype=torch.float32).flatten().unsqueeze(0)
        
        return tens, reward, done, cur_apple, snake_head

if __name__ == "__main__":
    time.sleep(2)
    h = Env.step(0, (-1, -1))[-1]
    print(h)
    h = Env.step(3, (-1, -1), 1)[-1]
    print(h)
    h = Env.step(2, (-1, -1), 2)[-1]
    print(h)
    h = Env.step(3, (-1, -1), 3)[-1]
    print(h)
    h = Env.step(2, (-1, -1), 4)[-1]
    print(h)
    h = Env.step(3, (-1, -1), 5)[-1]
    print(h)
