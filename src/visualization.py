import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import numpy as np

def _draw_base_map(ax, problem):
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle='--', alpha=0.6)

    for (ox, oy, r) in problem.obstacles:
        circle = patches.Circle((ox, oy), r, edgecolor='black', facecolor='gray', alpha=0.5, zorder=2)
        ax.add_patch(circle)

    ax.scatter(problem.start[0], problem.start[1], c='green', s=150, marker='s', zorder=5)
    ax.scatter(problem.end[0], problem.end[1], c='red', s=150, marker='X', zorder=5)


def plot_pathfinding(history, problem, best_pos, title="PSO Pathfinding - Rezultat Final"):
    plt.figure(figsize=(8, 8))
    ax = plt.gca()
    _draw_base_map(ax, problem)
    plt.title(title)

    waypoints = best_pos.reshape((problem.num_waypoints, 2))
    full_path = np.vstack([problem.start, waypoints, problem.end])

    plt.plot(full_path[:, 0], full_path[:, 1], c='blue', linewidth=3, label='Traseu Optim', zorder=4)
    plt.scatter(waypoints[:, 0], waypoints[:, 1], c='blue', s=50, zorder=5)

    plt.legend()
    plt.show()


def animate_pathfinding_evolution(history, problem):
    fig, ax = plt.subplots(figsize=(8, 8))

    def update(frame_idx):
        ax.clear()
        _draw_base_map(ax, problem)
        ax.set_title(f"Evoluție Roi - Iterația {frame_idx}")

        current_generation_positions = history[frame_idx]

        for particle_pos in current_generation_positions:
            waypoints = particle_pos.reshape((problem.num_waypoints, 2))
            full_path = np.vstack([problem.start, waypoints, problem.end])

            ax.plot(full_path[:, 0], full_path[:, 1], c='blue', alpha=0.15, linewidth=1, zorder=3)

    anim = animation.FuncAnimation(fig, update, frames=len(history), interval=100, repeat=False)

    plt.show()


def _draw_wifi_base(ax, problem):
    ax.set_xlim(0, problem.width)
    ax.set_ylim(0, problem.height)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_aspect('equal')

    for (wx, wy, w, h) in problem.walls:
        rect = patches.Rectangle((wx, wy), w, h, linewidth=1, edgecolor='black', facecolor='black', zorder=1)
        ax.add_patch(rect)


def plot_wifi_coverage(history, problem, best_pos, title="Wi-Fi Optimization - Final"):
    plt.figure(figsize=(10, 8))
    plt.title(title)

    routers = best_pos.reshape((problem.n_routers, 2))
    signal_map = np.zeros(problem.grid_x.shape)

    for i in range(problem.grid_x.shape[0]):
        for j in range(problem.grid_x.shape[1]):
            px = problem.grid_x[i, j]
            py = problem.grid_y[i, j]
            dists = [np.linalg.norm(np.array([px, py]) - r) for r in routers]
            min_dist = min(dists)

            if min_dist <= problem.radius:
                signal_map[i, j] = 1 - (min_dist / problem.radius)
            else:
                signal_map[i, j] = 0

    plt.imshow(signal_map, extent=(0, problem.width, 0, problem.height),
               origin='lower', cmap='YlGnBu', alpha=0.6, zorder=0)
    plt.colorbar(label='Intensitate Semnal')

    ax = plt.gca()
    _draw_wifi_base(ax, problem)

    for i, router in enumerate(routers):
        ax.scatter(router[0], router[1], c='red', s=150, marker='^', zorder=5, label=f'Router {i + 1}')
        circle = patches.Circle((router[0], router[1]), problem.radius, fill=False, edgecolor='red', linestyle='--',
                                alpha=0.5)
        ax.add_patch(circle)

    plt.legend()
    plt.show()


def animate_wifi_evolution(history, problem):
    fig, ax = plt.subplots(figsize=(8, 8))

    def update(frame_idx):
        ax.clear()
        _draw_wifi_base(ax, problem)
        ax.set_title(f"Optimizare Wi-Fi - Iterația {frame_idx}")

        current_gen = history[frame_idx]
        all_routers_x = []
        all_routers_y = []

        for particle_pos in current_gen:
            routers = particle_pos.reshape((problem.n_routers, 2))
            all_routers_x.extend(routers[:, 0])
            all_routers_y.extend(routers[:, 1])

        ax.scatter(all_routers_x, all_routers_y, c='blue', alpha=0.3, s=30)

    anim = animation.FuncAnimation(fig, update, frames=len(history), interval=100, repeat=False)
    plt.show()