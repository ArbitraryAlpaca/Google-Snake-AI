import numpy as np
import torch
import pyautogui
import time
from snake_simulation import Simulation

class Env:
    sim = Simulation()
    direction = [0,0,0,1]
    num_actions = 5
    state_size = 259
    @staticmethod
    def __state_from_image(state, ROWS=15, COLS=17):
        height, width = state.shape[:2]
        cell_h = height / ROWS
        cell_w = width / COLS

        apple = (-1,-1)
        head = (-1,-1)
        board = np.zeros((ROWS, COLS), dtype='float32')

        for row in range(ROWS):
            for col in range(COLS):
                #determine center of each cell
                x = int((col + 0.5) * cell_w)
                y = int((row + 0.5) * cell_h)

                pixel = state[y][x]
                if pixel[0] == 253 and pixel[1] == 192 and pixel[2] == 1:
                    return np.array([]), (-1, -1), (-1,-1)

                if pixel[0] == 231 and pixel[1] == 71 and pixel[2] == 29: #pixel color is red
                    board[row][col] = 1 #setting board position of apple as 1
                    apple = (row,col)
                elif np.array_equal(pixel,[78, 124, 246]) or np.array_equal(pixel,[77,123,249]):
                    board[row][col] = 0.5
                    head = (row,col)
                elif pixel[2] >= 190: #pixel color is blue
                    board[row][col] = 0.3 #setting body of snake as 0.3

        return board, apple, head

    def step(self,action_choice, prev_apple, prev_head=(7,3), i=''):
        done = False
        screen_region = (576, 302, 748, 660)
        actions = ['','left','right','up','down']
        # if action != 0:
        #     pyautogui.press(actions[action_choice])
        #image = np.array(pyautogui.screenshot(fr'.\Screenshot\currentBoard.png', region=screen_region))# image[y][x] (height (y) = 660, width (x) = 748)
        cur_state, cur_apple, snake_head = self.sim.move(actions[action_choice]) #Env.__state_from_image(image)


        # debug statement
        # with open("board.txt", "a") as file:
        #     file.write(f"{cur_state}\n{cur_apple} {snake_head}\n\n")
        if cur_state.size == 0:
            reward = -100
            done = True
            cur_state = np.zeros((15,17),dtype='float32')
        elif prev_apple != (-1, -1) and cur_apple != prev_apple:
            reward = 200
        else:
            reward = 10
            old_dist = abs(prev_head[0] - prev_apple[0]) + abs(prev_head[1] - prev_apple[1])
            new_dist = abs(snake_head[0] - cur_apple[0]) + abs(snake_head[1] - cur_apple[1])
            if new_dist < old_dist:
                reward +=5
            elif new_dist > old_dist:
                reward -=5
        direction_map = {
            (-1,0): [1,0,0,0], #up
            (1,0): [0,1,0,0], #down
            (0,-1): [0,0,1,0], #left
            (0,1):[0,0,0,1] #right
        }

        movement = (
            snake_head[0] - prev_head[0],
            snake_head[1] - prev_head[1]
        )

        self.direction = direction_map.get(movement) if abs(movement[0] + movement[1]) == 1 else self.direction
        cur_direction = np.array(self.direction, dtype='float32')

        cur_state = np.concatenate([cur_state.flatten(), cur_direction.flatten()])
        return torch.tensor(cur_state, dtype=torch.float32).flatten().unsqueeze(0), reward, done, cur_apple, snake_head


    def reset(self):
        self.sim = Simulation()
        screen_region = (576, 302, 748, 660)
        direction = np.array(
            [0,0,0,1], dtype='float32'
        )
        #pyautogui.press('Enter')
        #time.sleep(2)
        #image = np.array(pyautogui.screenshot(fr'.\Screenshot\startingBoard.png', region=screen_region))# image[y][x] (height (y) = 660, width (x) = 748)
        cur_state, cur_apple, head = self.sim.board, self.sim.apple, self.sim.snake_head #Env.__state_from_image(image)
        cur_state = np.concatenate([cur_state.flatten(), direction.flatten()])

        return torch.tensor(cur_state, dtype=torch.float32).flatten().unsqueeze(0), cur_apple, head

