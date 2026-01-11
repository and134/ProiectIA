import numpy as np
from pso_algorithm import PSO
from problem_wifi import WifiProblem
from visualization import plot_wifi_coverage, animate_wifi_evolution

if __name__ == "__main__":
    print("=== Rulare PSO pentru Acoperire Wi-Fi ===")

    problem = WifiProblem(room_size=(100, 100), n_routers=3, signal_radius=35)

    pso = PSO(
        objective_function=problem.fitness_function,
        bounds=problem.get_bounds(),
        num_particles=40,
        max_iter=60,
        topology='global'
    )

    best_pos, best_val, history, cost_history = pso.optimize()

    total_puncte = len(problem.grid_points)
    neacoperite = int(best_val)
    procent_acoperire = 100 * (1 - neacoperite / total_puncte)

    print(f"Cost final (Puncte moarte): {neacoperite} din {total_puncte}")
    print(f"Grad de acoperire: {procent_acoperire:.2f}%")

    print("1. Pornire Animație (închide fereastra pentru a vedea rezultatul final)...")
    animate_wifi_evolution(history, problem)

    print("2. Afișare Heatmap final...")
    plot_wifi_coverage(history, problem, best_pos)