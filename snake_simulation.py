import numpy as np
from collections import deque

class Simulation:
    def __init__(self):
        self.rows, self.cols = 15,17
        self.board = np.zeros((self.rows,self.cols), dtype='float32')
        self.apple = self.change_apple()
        self.snake_head = (7,3)
        self.snake_body = deque([(7,2), (7,1)])
        self.direction = (0,1)
        self.actions = {'': self.direction, 'right': (0,1), 'left': (0,-1), 'up':(-1,0), 'down':(1,0)}
        self.set_snake()

    def set_snake(self):
        self.board = np.zeros((self.rows,self.cols), dtype='float32')
        for coords in self.snake_body:
            self.board[coords[0]][coords[1]] = 0.3
        self.board[self.snake_head[0]][self.snake_head[1]] = 0.5
        self.board[self.apple[0]][self.apple[1]] = 1.0

    def change_apple(self):
        i,j = np.random.randint(0,15), np.random.randint(0,17)
        while self.board[i][j] != 0.0:
            i,j = np.random.randint(0,15), np.random.randint(0,17)
        self.board[i][j] = 1.0
        return i,j

    def crashed(self, coordinates):
        if coordinates[0] < 0 or coordinates[1] < 0 or coordinates[0] >= self.rows or coordinates[1] >= self.cols:
            return True
        if self.board[coordinates[0]][coordinates[1]] == 0.3:
            return True
        return False

    def consumed_apple(self, coordinates):
        if self.board[coordinates[0]][coordinates[1]] == 1.0:
            self.apple = self.change_apple()
            return True
        return False

    def move(self, action_chosen):
        act = self.actions[action_chosen]
        new_head = (self.snake_head[0] + act[0], self.snake_head[1] + act[1])
        if (-1*act[0], -1*act[1]) == self.direction: # prevent snake from moving backwards
            new_head = (self.snake_head[0] + self.direction[0], self.snake_head[1] + self.direction[1])

        self.direction = act
        self.actions[''] = self.direction # set default path of snake as direction snake currently moving in

        if self.crashed(new_head):
            return np.array([]), (-1, -1), (-1, -1)

        if not self.consumed_apple(new_head):
            self.snake_body.pop()

        self.snake_body.appendleft(self.snake_head)
        self.snake_head = new_head
        print(self.snake_body)
        self.set_snake()

        return self.board, self.snake_head, self.apple


if __name__ == '__main__':
    sim = Simulation()
    i = ""
    while(i != 'e'):
        print(sim.board)
        i = input()
        sim.move(i)