import matplotlib.pyplot as plt
from pso_algorithm import PSO
from problem_pathfinding import PathfindingProblem


def run_comparison():
    start = (5, 5)
    end = (95, 95)
    nr_puncte_control = 5
    problem = PathfindingProblem(start, end, nr_puncte_control)

    N_PARTICULE = 50
    N_ITERATII = 100

    topologies = [
        ('global', 0),
        ('social', 5),
        ('geographic', 5)
    ]

    results = {}

    print(f"=== Incepe Comparația ({N_ITERATII} iteratii, {N_PARTICULE} particule) ===")

    for top_name, n_size in topologies:
        print(f"Rulare topologie: {top_name.upper()}...")

        pso = PSO(
            objective_function=problem.fitness_function,
            bounds=problem.get_bounds(),
            num_particles=N_PARTICULE,
            max_iter=N_ITERATII,
            topology=top_name,
            neighbor_size=n_size
        )

        _, best_val, _, cost_history = pso.optimize()

        results[top_name] = cost_history
        print(f"-> Cost final {top_name}: {best_val:.2f}")

    plt.figure(figsize=(10, 6))
    plt.title(f"Analiza Convergenței PSO: Global vs Local\n(Pathfinding Problem)")
    plt.xlabel("Iterații (Timp)")
    plt.ylabel("Cost (Fitness)")
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.plot(results['global'], label='Global (gbest)', color='red', linewidth=2)
    plt.plot(results['social'], label='Local Social (lbest)', color='green', linestyle='--', linewidth=2)
    plt.plot(results['geographic'], label='Local Geographic (lbest)', color='blue', linestyle='-.', linewidth=2)

    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_comparison()