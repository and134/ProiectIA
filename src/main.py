import numpy as np
from pso_algorithm import PSO
from problem_pathfinding import PathfindingProblem
from visualization import plot_pathfinding, animate_pathfinding_evolution

if __name__ == "__main__":
    start = (5, 5)
    end = (95, 95)
    nr_puncte_control = 10

    problem = PathfindingProblem(start, end, nr_puncte_control)

    print("=== Rulare PSO pentru Pathfinding ===")

    pso = PSO(
        objective_function=problem.fitness_function,
        bounds=problem.get_bounds(),
        num_particles=100,
        max_iter=250,
        topology='global',
        neighbor_size=10
    )

    best_pos, best_val, history, cost_history = pso.optimize()

    print(f"Cel mai bun cost: {best_val:.2f}")

    print("Se generează animația...")

    animate_pathfinding_evolution(history, problem)
    print("Rezultatul optim")
    plot_pathfinding(history, problem, best_pos, title="PSO Pathfinding - Rezultat Final")