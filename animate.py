import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap, BoundaryNorm
import torch
import numpy as np

class Animate:
    def __init__(self, rows=15, cols=17, interval=100, game=-1): #interval = ms per frame (100 = 100ms per frame = 10fps)
        self.rows = rows
        self.cols = cols
        self.game = game
        self.interval = interval
        self.frames = []

        self.cmap = ListedColormap(["#3498db", "#2ecc71", "#3498db", "#e74c3c"])
        self.norm = BoundaryNorm([-1.5, -0.5, 0.25, 0.75, 1.5], self.cmap.N)

    def _to_2d(self, state, rows=None, cols=None):
        r = self.rows if rows is None else rows
        c = self.cols if cols is None else cols

        if isinstance(state, torch.Tensor):
            if state.numel() == (r * c) + 4:
                state = state.flatten()[:-4]
            return state.reshape(r, c).detach().cpu().numpy()

        return np.array(state).reshape(r, c)

    def add_frame(self, state):
        self.frames.append(self._to_2d(state))

    def add_frames(self, state_list):
        for state in state_list:
            self.add_frame(state)

    def clear(self):
        self.frames = []

    def show(self, save_path=None, display=False):
        if not self.frames:
            print("No frames to animate.")
            return

        fig, ax = plt.subplots()
        im = ax.imshow(self.frames[0], cmap=self.cmap, norm=self.norm)
        title = ax.set_title(f"Game: {self.game}")

        # subtle grid lines between grass tiles
        ax.set_xticks(np.arange(-0.5, self.cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, self.rows, 1), minor=True)
        ax.grid(which="minor", color="#27ae60", linestyle="-", linewidth=1)
        ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)

        def update(frame_idx):
            im.set_data(self.frames[frame_idx])

        self.ani = animation.FuncAnimation(
            fig,
            update,
            frames=len(self.frames),
            interval=self.interval,
            blit=False
        )

        if save_path:
            fps = max(1, int(1000 / self.interval))
            self.ani.save(save_path, writer="pillow", fps=fps)

        if display:
            plt.show()

        plt.close()
