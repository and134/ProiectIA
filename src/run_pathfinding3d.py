from pso_algorithm import PSO
from problem_pathfinding_3d import PathfindingProblem3D
from visualization_3d import plot_pathfinding_3d_static, animate_pathfinding_3d

if __name__ == "__main__":
    print("=== PSO Pathfinding 3D ===")

    start = (5, 5, 5)
    end = (95, 95, 95)
    nr_waypoints = 5

    problem = PathfindingProblem3D(start, end, nr_waypoints)
    pso = PSO(
        objective_function=problem.fitness_function,
        bounds=problem.get_bounds(),
        num_particles=50,
        max_iter=80,
        topology='global'
    )

    best_pos, best_val, history, cost_history = pso.optimize()

    print(f"Cost Final: {best_val:.2f}")

    print("1. Generare Animatie 3D...")
    animate_pathfinding_3d(history, problem)

    print("2. Afisare Rezultat Final...")
    plot_pathfinding_3d_static(history, problem, best_pos)