import tkinter as tk
from tkinter import ttk
import threading
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.animation as animation
import matplotlib.patches as patches

try:
    from pso_algorithm import PSO
    from problem_pathfinding import PathfindingProblem
    from problem_pathfinding_3d import PathfindingProblem3D
    from problem_wifi import WifiProblem
    from problem_wifi_3d import WifiProblem3D
except ImportError:
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from src.pso_algorithm import PSO
    from src.problem_pathfinding import PathfindingProblem
    from src.problem_pathfinding_3d import PathfindingProblem3D
    from src.problem_wifi import WifiProblem
    from src.problem_wifi_3d import WifiProblem3D


class PSOInterface:
    def __init__(self, root):
        self.root = root
        self.root.title("PSO Visualization Studio - v2.1 (Stable)")
        self.root.geometry("1200x850")

        self.is_running = False
        self.current_anim = None
        self.history = []
        self.best_pos = None
        self.problem_instance = None
        self.problem_mode = tk.StringVar(value="Pathfinding 2D")

        self.var_part = tk.DoubleVar(value=40)
        self.var_iter = tk.DoubleVar(value=100)
        self.var_complex = tk.DoubleVar(value=5)

        self._setup_ui()

    def _setup_ui(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        control_panel = ttk.LabelFrame(main_frame, text="Panou Control", width=300)
        control_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        control_panel.pack_propagate(False)

        ttk.Label(control_panel, text="Problemă:").pack(pady=(15, 5), anchor="w", padx=10)
        prob_cb = ttk.Combobox(control_panel, textvariable=self.problem_mode, state="readonly")
        prob_cb['values'] = ("Pathfinding 2D", "Pathfinding 3D", "Wi-Fi 2D", "Wi-Fi 3D")
        prob_cb.pack(fill=tk.X, padx=10)
        prob_cb.bind("<<ComboboxSelected>>", self._update_labels)

        self.lbl_part = ttk.Label(control_panel, text=f"Particule: {int(self.var_part.get())}")
        self.lbl_part.pack(pady=(20, 5), anchor="w", padx=10)
        s_part = ttk.Scale(control_panel, from_=10, to=100, variable=self.var_part, command=self._update_lbl_part)
        s_part.pack(fill=tk.X, padx=10)

        self.lbl_iter = ttk.Label(control_panel, text=f"Iterații: {int(self.var_iter.get())}")
        self.lbl_iter.pack(pady=(15, 5), anchor="w", padx=10)
        s_iter = ttk.Scale(control_panel, from_=20, to=300, variable=self.var_iter, command=self._update_lbl_iter)
        s_iter.pack(fill=tk.X, padx=10)

        self.lbl_complex = ttk.Label(control_panel, text=f"Nr. Waypoints: {int(self.var_complex.get())}")
        self.lbl_complex.pack(pady=(15, 5), anchor="w", padx=10)
        s_complex = ttk.Scale(control_panel, from_=3, to=15, variable=self.var_complex,
                              command=self._update_lbl_complex)
        s_complex.pack(fill=tk.X, padx=10)

        self.btn_start = tk.Button(control_panel, text="START SIMULARE", bg="#4CAF50", fg="white",
                                   font=("Arial", 12, "bold"), command=self.start_thread)
        self.btn_start.pack(fill=tk.X, pady=30, padx=10, ipady=5)

        ttk.Label(control_panel, text="Jurnal Execuție:").pack(anchor="w", padx=10)
        self.log_text = tk.Text(control_panel, height=20, width=30, state="disabled", bg="#f0f0f0",
                                font=("Consolas", 9))
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        plot_frame = ttk.Frame(main_frame)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.fig = Figure(figsize=(5, 5), dpi=100)
        self.ax = None

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)

    def _update_lbl_part(self, val):
        self.lbl_part.config(text=f"Particule: {int(float(val))}")

    def _update_lbl_iter(self, val):
        self.lbl_iter.config(text=f"Iterații: {int(float(val))}")

    def _update_lbl_complex(self, val):
        mode = self.problem_mode.get()
        txt = "Nr. Waypoints:" if "Pathfinding" in mode else "Nr. Routere:"
        self.lbl_complex.config(text=f"{txt} {int(float(val))}")

    def _update_labels(self, event=None):
        self._update_lbl_complex(self.var_complex.get())

    def log(self, msg):
        self.log_text.config(state="normal")
        self.log_text.insert(tk.END, f"> {msg}\n")
        self.log_text.see(tk.END)
        self.log_text.config(state="disabled")

    def start_thread(self):
        if self.is_running: return

        try:
            if self.current_anim and getattr(self.current_anim, 'event_source', None):
                self.current_anim.event_source.stop()
        except Exception:
            pass
        self.current_anim = None

        self.fig.clf()
        self.canvas.draw()

        self.log_text.config(state="normal");
        self.log_text.delete(1.0, tk.END);
        self.log_text.config(state="disabled")

        self.btn_start.config(state="disabled", text="Se calculează...", bg="#FFC107")
        self.is_running = True

        threading.Thread(target=self.run_algorithm, daemon=True).start()

    def run_algorithm(self):
        try:
            mode = self.problem_mode.get()
            n_part = int(self.var_part.get())
            n_iter = int(self.var_iter.get())
            complexity = int(self.var_complex.get())

            self.log(f"Config: {mode}")
            self.log(f"P: {n_part}, I: {n_iter}, C: {complexity}")

            if mode == "Pathfinding 2D":
                self.problem_instance = PathfindingProblem((5, 5), (95, 95), complexity)
            elif mode == "Pathfinding 3D":
                self.problem_instance = PathfindingProblem3D((5, 5, 5), (95, 95, 95), complexity)
            elif mode == "Wi-Fi 2D":
                self.problem_instance = WifiProblem(n_routers=complexity, signal_radius=35)
            elif mode == "Wi-Fi 3D":
                self.problem_instance = WifiProblem3D(n_routers=complexity, signal_radius=45)

            pso = PSO(
                objective_function=self.problem_instance.fitness_function,
                bounds=self.problem_instance.get_bounds(),
                num_particles=n_part,
                max_iter=n_iter,
                topology='global'
            )

            best_pos, best_val, history, _ = pso.optimize()
            self.history = history
            self.best_pos = best_pos

            self.log(f"Calcul complet. Cost Final: {best_val:.2f}")
            self.root.after(0, lambda: self.start_animation(mode))

        except Exception as e:
            self.root.after(0, lambda: self.show_error(str(e)))

    def show_error(self, msg):
        self.log(f"ERR: {msg}")
        self.is_running = False
        self.btn_start.config(state="normal", text="START SIMULARE", bg="#4CAF50")

    def start_animation(self, mode):
        self.log("Pornire animație...")

        if "3D" in mode:
            self.ax = self.fig.add_subplot(111, projection='3d')
        else:
            self.ax = self.fig.add_subplot(111)

        if mode == "Pathfinding 2D":
            self._anim_path_2d()
        elif mode == "Pathfinding 3D":
            self._anim_path_3d()
        elif mode == "Wi-Fi 2D":
            self._anim_wifi_2d()
        elif mode == "Wi-Fi 3D":
            self._anim_wifi_3d()

        self.canvas.draw()

    def finish_sequence(self):
        self.is_running = False
        self.btn_start.config(state="normal", text="START SIMULARE", bg="#4CAF50")
        self.log("Animație finalizată.")
        self.log("Generare Raport Final...")

        mode = self.problem_mode.get()
        self.fig.clf()

        if "3D" in mode:
            self.ax = self.fig.add_subplot(111, projection='3d')
        else:
            self.ax = self.fig.add_subplot(111)

        if mode == "Wi-Fi 2D":
            self._draw_final_wifi_2d()
        elif mode == "Wi-Fi 3D":
            self._draw_final_wifi_3d()
        elif mode == "Pathfinding 2D":
            self._draw_final_path_2d()
        elif mode == "Pathfinding 3D":
            self._draw_final_path_3d()

        self.canvas.draw()

    def _anim_path_2d(self):
        prob = self.problem_instance
        self.ax.set_title("Evoluție Pathfinding 2D")
        self.ax.set_xlim(0, 100);
        self.ax.set_ylim(0, 100)
        self.ax.grid(True, linestyle='--')

        for (ox, oy, r) in prob.obstacles:
            self.ax.add_patch(patches.Circle((ox, oy), r, color='gray', alpha=0.5))
        self.ax.plot(*prob.start, 'gs', ms=10);
        self.ax.plot(*prob.end, 'rx', ms=10)

        lines = [self.ax.plot([], [], 'c-', alpha=0.2)[0] for _ in range(len(self.history[0]))]

        def update(frame):
            if frame == len(self.history) - 1:
                self.root.after(100, self.finish_sequence)

            pop = self.history[frame]
            for i, p in enumerate(pop):
                pts = p.reshape((prob.num_waypoints, 2))
                full = np.vstack([prob.start, pts, prob.end])
                lines[i].set_data(full[:, 0], full[:, 1])
            self.ax.set_title(f"Iter: {frame}")

        self.current_anim = animation.FuncAnimation(self.fig, update, frames=len(self.history), interval=30,
                                                    repeat=False)

    def _draw_final_path_2d(self):
        prob = self.problem_instance
        self.ax.set_title("Rezultat Final Pathfinding 2D")
        self.ax.set_xlim(0, 100);
        self.ax.set_ylim(0, 100)

        for (ox, oy, r) in prob.obstacles:
            self.ax.add_patch(patches.Circle((ox, oy), r, color='gray', alpha=0.5))

        pts = self.best_pos.reshape((prob.num_waypoints, 2))
        full = np.vstack([prob.start, pts, prob.end])
        self.ax.plot(full[:, 0], full[:, 1], 'b-', linewidth=3, label="Traseu Optim")
        self.ax.plot(*prob.start, 'gs', ms=10);
        self.ax.plot(*prob.end, 'rx', ms=10)
        self.ax.legend()

    def _anim_wifi_2d(self):
        prob = self.problem_instance
        self.ax.set_title("Optimizare Wi-Fi...")
        self.ax.set_xlim(0, 100);
        self.ax.set_ylim(0, 100)

        for (wx, wy, w, h) in prob.walls:
            self.ax.add_patch(patches.Rectangle((wx, wy), w, h, facecolor='black'))

        scat = self.ax.scatter([], [], c='blue')

        def update(frame):
            if frame == len(self.history) - 1:
                self.root.after(100, self.finish_sequence)

            pop = self.history[frame]
            xs, ys = [], []
            for p in pop:
                r = p.reshape((prob.n_routers, 2))
                xs.extend(r[:, 0]);
                ys.extend(r[:, 1])
            scat.set_offsets(np.c_[xs, ys])

        self.current_anim = animation.FuncAnimation(self.fig, update, frames=len(self.history), interval=50,
                                                    repeat=False)

    def _draw_final_wifi_2d(self):
        prob = self.problem_instance
        self.ax.set_title("Heatmap Acoperire Wi-Fi")
        self.ax.set_xlim(0, 100);
        self.ax.set_ylim(0, 100)

        res = 100
        x = np.linspace(0, 100, res);
        y = np.linspace(0, 100, res)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)

        routers = self.best_pos.reshape((prob.n_routers, 2))

        for i in range(res):
            for j in range(res):
                px, py = X[i, j], Y[i, j]
                dists = [np.linalg.norm([px - r[0], py - r[1]]) for r in routers]
                md = min(dists)
                if md < prob.radius: Z[i, j] = 1 - (md / prob.radius)

        self.ax.imshow(Z, extent=(0, 100, 0, 100), origin='lower', cmap='YlGnBu', alpha=0.7)

        for (wx, wy, w, h) in prob.walls:
            self.ax.add_patch(patches.Rectangle((wx, wy), w, h, facecolor='black'))

        for r in routers:
            self.ax.scatter(*r, c='red', marker='^', s=100, edgecolors='white')
            self.ax.add_patch(patches.Circle(r, prob.radius, fill=False, edgecolor='red', linestyle='--'))

    def _anim_path_3d(self):
        prob = self.problem_instance
        self.ax.set_title("Pathfinding 3D")
        self.ax.set_xlim(0, 100);
        self.ax.set_ylim(0, 100);
        self.ax.set_zlim(0, 100)

        for (ox, oy, oz, r) in prob.obstacles:
            self.ax.scatter(ox, oy, oz, s=r * 20, c='gray', alpha=0.2)

        lines = [self.ax.plot([], [], [], 'b-', alpha=0.2)[0] for _ in range(min(20, len(self.history[0])))]

        def update(frame):
            if frame == len(self.history) - 1:
                self.root.after(100, self.finish_sequence)

            pop = self.history[frame]
            for i, l in enumerate(lines):
                if i >= len(pop): break
                pts = pop[i].reshape((prob.num_waypoints, 3))
                full = np.vstack([prob.start, pts, prob.end])
                l.set_data(full[:, 0], full[:, 1])
                l.set_3d_properties(full[:, 2])

        self.current_anim = animation.FuncAnimation(self.fig, update, frames=len(self.history), interval=60,
                                                    repeat=False)

    def _draw_final_path_3d(self):
        prob = self.problem_instance
        self.ax.set_title("Rezultat Final 3D")

        for (ox, oy, oz, r) in prob.obstacles:
            u, v = np.mgrid[0:2 * np.pi:15j, 0:np.pi:15j]
            x = ox + r * np.cos(u) * np.sin(v)
            y = oy + r * np.sin(u) * np.sin(v)
            z = oz + r * np.cos(v)
            self.ax.plot_wireframe(x, y, z, color="black", alpha=0.1)

        pts = self.best_pos.reshape((prob.num_waypoints, 3))
        full = np.vstack([prob.start, pts, prob.end])
        self.ax.plot(full[:, 0], full[:, 1], full[:, 2], 'b-', linewidth=3)
        self.ax.scatter(*prob.start, c='green', s=100)
        self.ax.scatter(*prob.end, c='red', s=100)

    def _anim_wifi_3d(self):
        prob = self.problem_instance
        self.ax.set_title("Wi-Fi 3D")
        self.ax.set_xlim(0, 100);
        self.ax.set_ylim(0, 100);
        self.ax.set_zlim(0, 100)

        scat = self.ax.scatter([], [], [], c='red')

        def update(frame):
            if frame == len(self.history) - 1:
                self.root.after(100, self.finish_sequence)

            pop = self.history[frame]
            xs, ys, zs = [], [], []
            for p in pop:
                r = p.reshape((prob.n_routers, 3))
                xs.extend(r[:, 0]);
                ys.extend(r[:, 1]);
                zs.extend(r[:, 2])
            scat._offsets3d = (xs, ys, zs)

        self.current_anim = animation.FuncAnimation(self.fig, update, frames=len(self.history), interval=60,
                                                    repeat=False)

    def _draw_final_wifi_3d(self):
        prob = self.problem_instance
        self.ax.set_title("Sfere Acoperire 3D")

        routers = self.best_pos.reshape((prob.n_routers, 3))
        u = np.linspace(0, 2 * np.pi, 15)
        v = np.linspace(0, np.pi, 15)

        for r in routers:
            self.ax.scatter(*r, c='red', s=100, marker='^')
            x = r[0] + prob.radius * np.outer(np.cos(u), np.sin(v))
            y = r[1] + prob.radius * np.outer(np.sin(u), np.sin(v))
            z = r[2] + prob.radius * np.outer(np.ones(np.size(u)), np.cos(v))
            self.ax.plot_wireframe(x, y, z, color='blue', alpha=0.2)


if __name__ == "__main__":
    root = tk.Tk()
    app = PSOInterface(root)
    root.mainloop()