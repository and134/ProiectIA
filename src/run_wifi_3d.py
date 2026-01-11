from pso_algorithm import PSO
from problem_wifi_3d import WifiProblem3D
from visualization_3d import plot_wifi_3d_static, animate_wifi_3d

def run():
    print("=== PSO Wi-Fi Coverage 3D ===")

    problem = WifiProblem3D(room_size=(100, 100, 100), n_routers=3, signal_radius=45)

    pso = PSO(
        objective_function=problem.fitness_function,
        bounds=problem.get_bounds(),
        num_particles=40,
        max_iter=60,
        topology='global'
    )

    best_pos, best_val, history, cost_history = pso.optimize()

    print(f"Puncte neacoperite (aprox): {int(best_val)}")

    print("1. Animație evoluție routere 3D...")
    animate_wifi_3d(history, problem)

    print("2. Rezultat Final (Sfere de semnal)...")
    plot_wifi_3d_static(history, problem, best_pos)

    if __name__ == "__main__":
        run()