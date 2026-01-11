import numpy as np
import copy


class Particle:
    def __init__(self, bounds, dimension):
        self.position = np.array([np.random.uniform(b[0], b[1]) for b in bounds])
        self.velocity = np.zeros(dimension)
        self.best_position = copy.deepcopy(self.position)
        self.best_value = float('inf')
        self.current_value = float('inf')


class PSO:
    def __init__(self, objective_function, bounds, num_particles, max_iter,
                 w=0.729, c1=1.49, c2=1.49, topology='global', neighbor_size=3):
        self.fitness_func = objective_function
        self.bounds = bounds
        self.dim = len(bounds)
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.topology = topology
        self.neighbor_size = neighbor_size

        self.v_max = np.array([0.2 * (b[1] - b[0]) for b in bounds])

        self.swarm = [Particle(bounds, self.dim) for _ in range(num_particles)]
        self.global_best_position = np.zeros(self.dim)
        self.global_best_value = float('inf')

        self.history = []
        self.cost_history = []

    def _get_social_target(self, particle_index):
        if self.topology == 'global':
            return self.global_best_position

        neighbors_indices = []
        if self.topology == 'social':
            start = particle_index - (self.neighbor_size // 2)
            for i in range(start, start + self.neighbor_size):
                neighbors_indices.append(i % self.num_particles)

        elif self.topology == 'geographic':
            distances = []
            current_pos = self.swarm[particle_index].position
            for i, p in enumerate(self.swarm):
                dist = np.linalg.norm(current_pos - p.position)
                distances.append((dist, i))
            distances.sort(key=lambda x: x[0])
            neighbors_indices = [x[1] for x in distances[:self.neighbor_size]]

        best_neighbor_val = float('inf')
        best_neighbor_pos = self.swarm[particle_index].best_position

        for idx in neighbors_indices:
            if self.swarm[idx].best_value < best_neighbor_val:
                best_neighbor_val = self.swarm[idx].best_value
                best_neighbor_pos = self.swarm[idx].best_position

        return best_neighbor_pos

    def optimize(self):
        w_start = 0.9
        w_end = 0.4

        for iteration in range(self.max_iter):
            self.w = w_start - (w_start - w_end) * (iteration / self.max_iter)
            self.history.append([p.position.copy() for p in self.swarm])

            for i, particle in enumerate(self.swarm):
                fitness = self.fitness_func(particle.position)
                particle.current_value = fitness

                if fitness < particle.best_value:
                    particle.best_value = fitness
                    particle.best_position = particle.position.copy()

                if fitness < self.global_best_value:
                    self.global_best_value = fitness
                    self.global_best_position = particle.position.copy()

            self.cost_history.append(self.global_best_value)
            for i, particle in enumerate(self.swarm):
                target_social = self._get_social_target(i)
                r1 = np.random.random(self.dim)
                r2 = np.random.random(self.dim)

                cognitive = self.c1 * r1 * (particle.best_position - particle.position)
                social = self.c2 * r2 * (target_social - particle.position)

                particle.velocity = (self.w * particle.velocity) + cognitive + social
                particle.velocity = np.clip(particle.velocity, -self.v_max, self.v_max)
                particle.position += particle.velocity
                for d in range(self.dim):
                    particle.position[d] = max(self.bounds[d][0], min(particle.position[d], self.bounds[d][1]))


        return self.global_best_position, self.global_best_value, self.history, self.cost_history