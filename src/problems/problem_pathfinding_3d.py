import numpy as np

class PathfindingProblem3D:
    def __init__(self, start_pos, end_pos, num_waypoints):
        self.start = np.array(start_pos)
        self.end = np.array(end_pos)
        self.num_waypoints = num_waypoints

        self.obstacles = [
            (50, 50, 50, 20),
            (20, 20, 20, 15),
            (80, 80, 80, 15),
            (20, 80, 50, 10)
        ]

    def get_bounds(self):
        bounds = []
        for _ in range(self.num_waypoints):
            bounds.append((0, 100))
            bounds.append((0, 100))
            bounds.append((0, 100))
        return bounds

    def _check_collision(self, p1, p2):
        for (ox, oy, oz, r) in self.obstacles:
            obstacle_center = np.array([ox, oy, oz])

            d_vec = p2 - p1
            f_vec = p1 - obstacle_center

            d2 = np.dot(d_vec, d_vec)
            if d2 == 0: continue

            t = -np.dot(f_vec, d_vec) / d2
            t = np.clip(t, 0, 1)

            closest_point = p1 + t * d_vec
            distance = np.linalg.norm(closest_point - obstacle_center)

            if distance < r:
                return True
        return False

    def fitness_function(self, particle_position):
        waypoints = particle_position.reshape((self.num_waypoints, 3))

        full_path = [self.start] + list(waypoints) + [self.end]

        total_distance = 0
        penalty = 0

        for i in range(len(full_path) - 1):
            p1 = np.array(full_path[i])
            p2 = np.array(full_path[i + 1])

            dist = np.linalg.norm(p2 - p1)
            total_distance += dist

            if self._check_collision(p1, p2):
                penalty += 1000

        return total_distance + penalty