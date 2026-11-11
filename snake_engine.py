import random
import sys
import time

import numpy as np
from sympy.printing.pytorch import torch


class SnakeEngine:
    def __init__(self, width=15, height=17):
        self.width = width
        self.height = height
        self.reset()

    def reset(self):
        # Snake starts in the middle moving right
        start_x = self.width // 2
        start_y = self.height // 2
        self.snake = [
            (start_x, start_y),
            (start_x - 1, start_y),
            (start_x - 2, start_y)
        ]
        self.direction = (1, 0)  # (dx, dy)
        self.food = self._spawn_food()
        self.score = 0
        self.game_over = False

    def _spawn_food(self):
        while True:
            pos = (random.randint(0, self.width - 1), random.randint(0, self.height - 1))
            if pos not in self.snake:
                return pos

    def change_direction(self, new_dir):
        # Prevent reversing directly into itself
        dx, dy = new_dir
        if dx == 0 and dy == 0:
            return
        curr_dx, curr_dy = self.direction
        if (dx * -1 != curr_dx) or (dy * -1 != curr_dy):
            self.direction = new_dir

    def step(self):
        if self.game_over:
            return False

        head_x, head_y = self.snake[0]
        dx, dy = self.direction
        new_head = (head_x + dx, head_y + dy)

        # Check wall collision
        if (
                new_head[0] < 0 or new_head[0] >= self.width or
                new_head[1] < 0 or new_head[1] >= self.height
        ):
            self.game_over = True
            return False

        # Check self collision
        if new_head in self.snake:
            self.game_over = True
            return False

        self.snake.insert(0, new_head)

        # Check food consumption
        if new_head == self.food:
            self.score += 1
            self.food = self._spawn_food()
        else:
            self.snake.pop()

        return True

    def render_ascii(self):
        board = [["." for _ in range(self.width)] for _ in range(self.height)]
        fx, fy = self.food
        board[fy][fx] = "F"
        for i, (sx, sy) in enumerate(self.snake):
            board[sy][sx] = "O" if i == 0 else "x"

        output = "\n".join(["".join(row) for row in board])
        print(f"\0.6.26{output}\nScore: {self.score} | Game Over: {self.game_over}\n")

    def get_state(self):
        x, y = self.snake[0]
        dx, dy = self.direction

        danger_straight = 0 if 0 <= x + dx <= self.width and 0 <= y + dy <= self.height else 1

        lmap = {
            (1, 0): (0,1),
            (-1, 0): (0, -1),
            (0, -1): (1,0),
            (0, 1): (-1,0),
        }

        ldx, ldy = lmap[self.direction]

        danger_left = 0 if 0 <= x + ldx <= self.width and 0 <= y + ldy <= self.height else 1

        rmap = {
            (1, 0): (0,-1),
            (-1, 0): (0, 1),
            (0, -1): (-1,0),
            (0, 1): (1,0),
        }

        rdx, rdy = rmap[self.direction]

        danger_right = 0 if 0 <= x + rdx <= self.width and 0 <= y + rdy <= self.height else 1

        fx, fy = self.food
        fdx, fdy = (-1 if fx - x < 0 else 0 if fx == x else 1, -1 if fy - y < 0 else 0 if fy == y else 1)

        state = np.array([danger_straight, danger_left, danger_right, dx, dy, fdx, fdy], dtype='float32')
        state = torch.tensor(state, dtype=torch.float32).flatten().unsqueeze(0)

        """
        board = [[0 for _ in range(self.width)] for _ in range(self.height)]
        for i, (sx, sy) in enumerate(self.snake):
            board[sy][sx] = 0.5 if i == 0 else -1
        fx, fy = self.food
        board[fy][fx] = 1

        board = np.array(board, dtype='float32')
        board = torch.tensor(board, dtype=torch.float32).flatten().unsqueeze(0)

        return (board, (fy, fx), self.snake[0])
        """

        return state

if __name__ == "__main__":
    # Quick demo loop
    engine = SnakeEngine()
    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]

    while not engine.game_over:
        # Simple random turn logic for demonstration
        if random.random() < 0.3:
            engine.change_direction(random.choice(directions))

        engine.step()
        engine.render_ascii()
        time.sleep(0.1)