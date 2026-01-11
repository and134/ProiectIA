import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def _draw_base_3d(ax, problem):
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_zlim(0, 100)

    ax.scatter(problem.start[0], problem.start[1], problem.start[2], c='green', s=100, marker='o', label='Start')
    ax.scatter(problem.end[0], problem.end[1], problem.end[2], c='red', s=100, marker='^', label='End')

    for (ox, oy, oz, r) in problem.obstacles:
        ax.scatter(ox, oy, oz, s=r * 100, c='gray', alpha=0.2)

        u, v = np.mgrid[0:2 * np.pi:10j, 0:np.pi:10j]
        x = ox + r * np.cos(u) * np.sin(v)
        y = oy + r * np.sin(u) * np.sin(v)
        z = oz + r * np.cos(v)
        ax.plot_wireframe(x, y, z, color="black", alpha=0.1)


def plot_pathfinding_3d_static(history, problem, best_pos):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    _draw_base_3d(ax, problem)

    waypoints = best_pos.reshape((problem.num_waypoints, 3))
    full_path = np.vstack([problem.start, waypoints, problem.end])

    ax.plot(full_path[:, 0], full_path[:, 1], full_path[:, 2], c='blue', linewidth=3, label='Traseu Optim')
    ax.scatter(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2], c='blue', s=50)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.legend()
    plt.show()


def animate_pathfinding_3d(history, problem):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    def update(frame_idx):
        ax.clear()
        _draw_base_3d(ax, problem)
        ax.set_title(f"3D Pathfinding - Iterația {frame_idx}")

        current_gen = history[frame_idx]

        for particle_pos in current_gen:
            waypoints = particle_pos.reshape((problem.num_waypoints, 3))
            full_path = np.vstack([problem.start, waypoints, problem.end])

            ax.plot(full_path[:, 0], full_path[:, 1], full_path[:, 2],
                    c='blue', alpha=0.15, linewidth=1)

    anim = animation.FuncAnimation(fig, update, frames=len(history), interval=200, repeat=False)
    plt.show()

def _draw_wifi_base_3d(ax, problem):
    ax.set_xlim(0, problem.width)
    ax.set_ylim(0, problem.depth)
    ax.set_zlim(0, problem.height)

    for (x1, x2, y1, y2, z1, z2) in problem.walls:
        xx, yy, zz = np.meshgrid(np.linspace(x1, x2, 2), np.linspace(y1, y2, 5), np.linspace(z1, z2, 5))
        ax.scatter(xx, yy, zz, c='black', alpha=0.5, s=20)


def plot_wifi_3d_static(history, problem, best_pos):

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Acoperire Wi-Fi 3D - Final")

    _draw_wifi_base_3d(ax, problem)

    routers = best_pos.reshape((problem.n_routers, 3))

    u = np.linspace(0, 2 * np.pi, 15)
    v = np.linspace(0, np.pi, 15)

    for i, (rx, ry, rz) in enumerate(routers):
        ax.scatter(rx, ry, rz, c='red', s=100, marker='^', label=f'Router {i + 1}')

        x = rx + problem.radius * np.outer(np.cos(u), np.sin(v))
        y = ry + problem.radius * np.outer(np.sin(u), np.sin(v))
        z = rz + problem.radius * np.outer(np.ones(np.size(u)), np.cos(v))

        ax.plot_wireframe(x, y, z, color='blue', alpha=0.2)

    ax.set_xlabel('X (Latime)')
    ax.set_ylabel('Y (Adancime)')
    ax.set_zlabel('Z (Inaltime)')
    plt.legend()
    plt.show()


def animate_wifi_3d(history, problem):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    def update(frame_idx):
        ax.clear()
        _draw_wifi_base_3d(ax, problem)
        ax.set_title(f"Optimizare Wi-Fi 3D - Iterația {frame_idx}")

        current_gen = history[frame_idx]

        xs, ys, zs = [], [], []
        for particle in current_gen:
            routers = particle.reshape((problem.n_routers, 3))
            xs.extend(routers[:, 0])
            ys.extend(routers[:, 1])
            zs.extend(routers[:, 2])

        ax.scatter(xs, ys, zs, c='blue', alpha=0.3, s=20)

    anim = animation.FuncAnimation(fig, update, frames=len(history), interval=100, repeat=False)
    plt.show()