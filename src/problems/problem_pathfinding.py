import numpy as np


class PathfindingProblem:
    def __init__(self, start_pos, end_pos, num_waypoints):
        self.start = np.array(start_pos)
        self.end = np.array(end_pos)
        self.num_waypoints = num_waypoints

        self.obstacles = [
            (30, 30, 10),
            (60, 60, 15),
            (30, 70, 10),
            (70, 20, 10)
        ]

    def get_bounds(self):
        bounds = []
        for _ in range(self.num_waypoints):
            bounds.append((0, 100))
            bounds.append((0, 100))
        return bounds

    def _is_point_in_obstacle(self, point):
        for (ox, oy, r) in self.obstacles:
            if np.linalg.norm(point - np.array([ox, oy])) < r:
                return True
        return False

    def fitness_function(self, particle_position):
        waypoints = particle_position.reshape((self.num_waypoints, 2))
        full_path = [self.start] + list(waypoints) + [self.end]

        total_distance = 0
        penalty = 0

        for i in range(len(full_path) - 1):
            p1 = np.array(full_path[i])
            p2 = np.array(full_path[i + 1])

            dist = np.linalg.norm(p2 - p1)
            total_distance += dist

            if self._check_collision(p1, p2):
                samples = 5
                for t in np.linspace(0, 1, samples):
                    sample_point = p1 + t * (p2 - p1)
                    if self._is_point_in_obstacle(sample_point):
                        penalty += 200

        return total_distance + penalty