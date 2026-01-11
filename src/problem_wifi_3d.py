import numpy as np


class WifiProblem3D:
    def __init__(self, room_size=(100, 100, 100), n_routers=3, signal_radius=35):
        """
        room_size: (Width, Depth, Height)
        """
        self.width, self.depth, self.height = room_size
        self.n_routers = n_routers
        self.radius = signal_radius

        self.walls = [
            (45, 55, 0, 60, 0, 100)
        ]

        x = np.linspace(0, self.width, 20)
        y = np.linspace(0, self.depth, 20)
        z = np.linspace(0, self.height, 20)

        self.grid_x, self.grid_y, self.grid_z = np.meshgrid(x, y, z)

        self.grid_points = np.column_stack((
            self.grid_x.ravel(),
            self.grid_y.ravel(),
            self.grid_z.ravel()
        ))

    def get_bounds(self):

        bounds = []
        for _ in range(self.n_routers):
            bounds.append((0, self.width))  # X
            bounds.append((0, self.depth))  # Y
            bounds.append((0, self.height))  # Z
        return bounds

    def fitness_function(self, particle_position):

        routers = particle_position.reshape((self.n_routers, 3))

        diff = self.grid_points[:, np.newaxis, :] - routers[np.newaxis, :, :]

        dists_sq = np.sum(diff ** 2, axis=2)

        r_sq = self.radius ** 2


        min_dists_sq = np.min(dists_sq, axis=1)

        uncovered_count = np.sum(min_dists_sq > r_sq)

        return uncovered_count