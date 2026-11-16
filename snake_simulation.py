import time

import numpy as np
from collections import deque

class Simulation:
    head_val = 0.5
    body_val = 0.3
    apple_val = 1
    def __init__(self):
        self.rows, self.cols = 15, 17
        self.board = np.zeros((self.rows,self.cols), dtype=np.float32)
        self.snake_head = (7, 3)
        self.snake_body = deque([(7, 2), (7, 1)])
        self.direction = (0, 1)
        self.actions = {'': self.direction, 'right': (0, 1), 'left': (0, -1), 'up':(-1, 0), 'down':(1, 0)}
        self.set_snake()
        self.apple = self.spawn_apple()


    def set_snake(self):
        for coords in self.snake_body:
            self.board[coords[0]][coords[1]] = self.body_val
        self.board[self.snake_head[0]][self.snake_head[1]] = self.head_val

    def spawn_apple(self):
        for _ in range(10):
            i,j = np.random.randint(0,self.rows), np.random.randint(0,self.cols)
            if self.board[i][j] == 0.0:
                self.board[i][j] = self.apple_val
                return i,j

        empty_positions = np.argwhere(self.board == 0.0)

        if len(empty_positions) == 0:
            return None
        i,j = empty_positions[np.random.randint(len(empty_positions))]
        self.board[i][j] = self.apple_val
        return i,j

    def crashed(self, coordinates):
        if coordinates[0] < 0 or coordinates[1] < 0 or coordinates[0] >= self.rows or coordinates[1] >= self.cols:
            return True
        if self.board[coordinates[0]][coordinates[1]] == self.body_val:
            return coordinates != self.snake_body[-1]
        return False

    def consumed_apple(self, coordinates):
        if self.board[coordinates[0]][coordinates[1]] == self.apple_val:
            self.apple = self.spawn_apple()
            return True
        return False

    def move(self, action_chosen):
        act = self.actions[action_chosen]

        if (-act[0], -act[1]) == self.direction: # prevent snake from moving backwards
            act = self.direction

        new_head = (self.snake_head[0] + act[0], self.snake_head[1] + act[1])

        self.direction = act
        self.actions[''] = self.direction # set default path of snake as direction snake is currently moving in

        if self.crashed(new_head):
            return np.array([]), (-1, -1), (-1, -1)

        if not self.consumed_apple(new_head):
            if self.apple is None:
                return self.board, self.apple, self.snake_head

            tail = self.snake_body.pop()
            self.board[tail[0]][tail[1]] = 0.0

        self.snake_body.appendleft(self.snake_head)
        self.snake_head = new_head
        self.set_snake()

        return self.board, self.apple, self.snake_head


if __name__ == '__main__':
    sim = Simulation()
    i = ""
    while True:
        print(sim.board)
        print(sim.apple, sim.snake_head)
        i = input()
        b, a, h = sim.move(i)
        if len(b) == 0:
            break
