import numpy as np


class WifiProblem:
    def __init__(self, room_size=(100, 100), n_routers=3, signal_radius=30):
        self.width, self.height = room_size
        self.n_routers = n_routers
        self.radius = signal_radius

        self.walls = [
            (40, 0, 5, 60),
            (40, 80, 5, 20),
            (0, 50, 30, 5)
        ]

        x = np.linspace(0, self.width, 50)
        y = np.linspace(0, self.height, 50)
        self.grid_x, self.grid_y = np.meshgrid(x, y)

        self.grid_points = np.column_stack((self.grid_x.ravel(), self.grid_y.ravel()))

    def get_bounds(self):
        bounds = []
        for _ in range(self.n_routers):
            bounds.append((0, self.width))
            bounds.append((0, self.height))
        return bounds

    def fitness_function(self, particle_position):
        routers = particle_position.reshape((self.n_routers, 2))
        diff = self.grid_points[:, np.newaxis, :] - routers[np.newaxis, :, :]

        dists_sq = np.sum(diff ** 2, axis=2)

        r_sq = self.radius ** 2
        min_dists_sq = np.min(dists_sq, axis=1)
        uncovered_count = np.sum(min_dists_sq > r_sq)
        return uncovered_count