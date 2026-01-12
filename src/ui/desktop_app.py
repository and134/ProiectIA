import tkinter as tk
from tkinter import ttk
import threading
import numpy as np

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.animation as animation
import matplotlib.patches as patches

try:
    from src.core.pso_algorithm import PSO
    from src.problems.problem_pathfinding import PathfindingProblem
    from src.problems.problem_pathfinding_3d import PathfindingProblem3D
    from src.problems.problem_wifi import WifiProblem
    from src.problems.problem_wifi_3d import WifiProblem3D
except ImportError:
    import sys, os

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from src.core.pso_algorithm import PSO
    from src.problems.problem_pathfinding import PathfindingProblem
    from src.problems.problem_pathfinding_3d import PathfindingProblem3D
    from src.problems.problem_wifi import WifiProblem
    from src.problems.problem_wifi_3d import WifiProblem3D

C = {
    "bg_main": "#2b2b2b",
    "bg_side": "#3c3f41",
    "fg_text": "#ffffff",
    "accent": "#4a90e2",
    "btn_bg": "#5c8a8a",
    "btn_fg": "#ffffff",
    "plot_bg": "#2b2b2b"
}


class PSOInterface:
    def __init__(self, root):
        self.root = root
        self.root.title("PSO Studio")
        self.root.geometry("1300x850")
        self.root.configure(bg=C["bg_main"])

        self.problem_mode = tk.StringVar(value="Pathfinding 2D")
        self.topology_mode = tk.StringVar(value="global")

        self.var_part = tk.DoubleVar(value=40)
        self.var_iter = tk.DoubleVar(value=100)
        self.var_complex = tk.DoubleVar(value=5)

        self.is_running = False
        self.current_anim = None
        self.history = []
        self.best_pos = None
        self.problem_instance = None
        self.comparison_results = {}

        self._configure_styles()
        self._setup_ui()

    def _configure_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TFrame", background=C["bg_main"])
        style.configure("Sidebar.TFrame", background=C["bg_side"])

        style.configure("TLabel", background=C["bg_side"], foreground=C["fg_text"], font=("Helvetica", 11))

        style.configure("TButton",
                        background=C["btn_bg"],
                        foreground=C["btn_fg"],
                        font=("Helvetica", 11, "bold"),
                        borderwidth=0, focuscolor="none")
        style.map("TButton", background=[("active", "#4a6fa5")])  # Hover effect

        style.configure("TCombobox", fieldbackground="#555", background="#777", foreground="white", arrowcolor="white")
        style.map("TCombobox", fieldbackground=[("readonly", "#555")], selectbackground=[("readonly", C["accent"])])

        style.configure("Horizontal.TScale", background=C["bg_side"], troughcolor="#555", sliderthickness=15)

    def _setup_ui(self):
        sidebar = ttk.Frame(self.root, style="Sidebar.TFrame", width=320)
        sidebar.pack(side=tk.LEFT, fill=tk.Y)
        sidebar.pack_propagate(False)

        lbl_head = tk.Label(sidebar, text="CONFIGURARE", bg=C["bg_side"], fg="#aaa", font=("Helvetica", 12, "bold"))
        lbl_head.pack(pady=(20, 10), padx=15, anchor="w")

        self._add_label(sidebar, "Scenariu:")
        prob_cb = ttk.Combobox(sidebar, textvariable=self.problem_mode, state="readonly")
        prob_cb['values'] = ("Pathfinding 2D", "Pathfinding 3D", "Wi-Fi 2D", "Wi-Fi 3D",
                             "STUDIU COMPARATIV (Pathfinding)")
        prob_cb.pack(fill=tk.X, padx=15, pady=5)
        prob_cb.bind("<<ComboboxSelected>>", self._on_mode_change)

        self._add_label(sidebar, "Topologie:")
        topo_cb = ttk.Combobox(sidebar, textvariable=self.topology_mode, state="readonly")
        topo_cb['values'] = ("Global", "Social", "Geographic")
        topo_cb.pack(fill=tk.X, padx=15, pady=5)

        tk.Frame(sidebar, bg="#555", height=2).pack(fill=tk.X, padx=15, pady=20)

        self.lbl_part = self._add_label(sidebar, f"Particule: {int(self.var_part.get())}")
        ttk.Scale(sidebar, from_=10, to=100, variable=self.var_part, command=self._update_lbl_part).pack(fill=tk.X,
                                                                                                         padx=15,
                                                                                                         pady=5)

        self.lbl_iter = self._add_label(sidebar, f"Iterații: {int(self.var_iter.get())}")
        ttk.Scale(sidebar, from_=20, to=300, variable=self.var_iter, command=self._update_lbl_iter).pack(fill=tk.X,
                                                                                                         padx=15,
                                                                                                         pady=5)

        self.lbl_complex = self._add_label(sidebar, f"Nr. Waypoints: {int(self.var_complex.get())}")
        ttk.Scale(sidebar, from_=3, to=15, variable=self.var_complex, command=self._update_lbl_complex).pack(fill=tk.X,
                                                                                                             padx=15,
                                                                                                             pady=5)

        tk.Frame(sidebar, bg=C["bg_side"], height=30).pack()

        self.btn_start = ttk.Button(sidebar, text="START SIMULARE", command=self.start_thread)
        self.btn_start.pack(fill=tk.X, padx=15, ipady=10)

        self._add_label(sidebar, "Jurnal:")
        self.log_text = tk.Text(sidebar, height=12, bg="#222", fg="#00ff00",
                                font=("Menlo", 10), relief="flat", highlightthickness=0)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=15, pady=(5, 15))

        plot_area = tk.Frame(self.root, bg=C["bg_main"])
        plot_area.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.fig = Figure(figsize=(5, 5), dpi=100)
        self.fig.patch.set_facecolor(C["bg_main"])

        self.ax = None

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_area)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.get_tk_widget().configure(highlightthickness=0, borderwidth=0)

        toolbar = NavigationToolbar2Tk(self.canvas, plot_area)
        toolbar.config(background="#ddd")
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)

    def _add_label(self, parent, text):
        lbl = tk.Label(parent, text=text, bg=C["bg_side"], fg="white", font=("Helvetica", 10), anchor="w")
        lbl.pack(fill=tk.X, padx=15, pady=(10, 0))
        return lbl

    def _update_lbl_part(self, val):
        self.lbl_part.config(text=f"Particule: {int(float(val))}")

    def _update_lbl_iter(self, val):
        self.lbl_iter.config(text=f"Iterații: {int(float(val))}")

    def _update_lbl_complex(self, val):
        mode = self.problem_mode.get()
        txt = "Nr. Waypoints:" if "Pathfinding" in mode else "Nr. Routere:"
        if "COMPARATIV" in mode: txt = "Nr. Waypoints (Test):"
        self.lbl_complex.config(text=f"{txt} {int(float(val))}")

    def _on_mode_change(self, event):
        self._update_lbl_complex(self.var_complex.get())
        if "COMPARATIV" in self.problem_mode.get():
            self.btn_start.config(text="GENEREAZĂ GRAFIC")
        else:
            self.btn_start.config(text="START SIMULARE")

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
        except:
            pass
        self.current_anim = None

        self.fig.clf()
        self.canvas.draw()
        self.log_text.config(state="normal");
        self.log_text.delete(1.0, tk.END);
        self.log_text.config(state="disabled")

        self.btn_start.config(state="disabled", text="CALCULEAZĂ...")
        self.is_running = True

        mode = self.problem_mode.get()
        if "COMPARATIV" in mode:
            threading.Thread(target=self.run_comparison_logic, daemon=True).start()
        else:
            threading.Thread(target=self.run_simulation_logic, daemon=True).start()

    def run_simulation_logic(self):
        try:
            mode = self.problem_mode.get()
            topo = self.topology_mode.get()

            n_part = int(self.var_part.get())
            n_iter = int(self.var_iter.get())
            comp = int(self.var_complex.get())
            self.log(f"Simulare: {mode}")

            if mode == "Pathfinding 2D":
                self.problem_instance = PathfindingProblem((5, 5), (95, 95), comp)
            elif mode == "Pathfinding 3D":
                self.problem_instance = PathfindingProblem3D((5, 5, 5), (95, 95, 95), comp)
            elif mode == "Wi-Fi 2D":
                self.problem_instance = WifiProblem(n_routers=comp, signal_radius=35)
            elif mode == "Wi-Fi 3D":
                self.problem_instance = WifiProblem3D(n_routers=comp, signal_radius=45)

            pso = PSO(self.problem_instance.fitness_function,
                      self.problem_instance.get_bounds(),
                      n_part,
                      n_iter,
                      topology=topo,
                      neighbor_size=5)

            best_pos, best_val, history, _ = pso.optimize()
            self.history = history;
            self.best_pos = best_pos

            self.log(f"Cost Final: {best_val:.2f}")
            self.root.after(0, lambda: self.start_animation(mode))
        except Exception as e:
            self.root.after(0, lambda: self.show_error(str(e)))

    def run_comparison_logic(self):
        try:
            self.log("Studiu Comparativ...")
            problem = PathfindingProblem((5, 5), (95, 95), int(self.var_complex.get()))
            topologies = [('global', 0), ('social', 5), ('geographic', 5)]
            self.comparison_results = {}
            for top, size in topologies:
                self.log(f"Rulare: {top}...")
                pso = PSO(problem.fitness_function, problem.get_bounds(), int(self.var_part.get()),
                          int(self.var_iter.get()), topology=top, neighbor_size=size)
                _, best, _, hist = pso.optimize()
                self.comparison_results[top] = hist
                self.log(f"-> {top}: {best:.2f}")
            self.root.after(0, self.draw_comparison_chart)
        except Exception as e:
            self.root.after(0, lambda: self.show_error(str(e)))

    def show_error(self, msg):
        self.log(f"Eroare: {msg}")
        self.is_running = False
        self.btn_start.config(state="normal", text="EROARE")

    def _style_axes(self, ax, title, is_3d=False):
        ax.set_facecolor(C["bg_main"])
        ax.set_title(title, color="white", fontsize=14, pad=15)

        if not is_3d:
            ax.tick_params(colors='white')
            for spine in ax.spines.values(): spine.set_color('#666')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.grid(True, linestyle=':', alpha=0.4, color="#777")
        else:
            ax.tick_params(axis='x', colors='white');
            ax.tick_params(axis='y', colors='white');
            ax.tick_params(axis='z', colors='white')
            ax.xaxis.label.set_color('white');
            ax.yaxis.label.set_color('white');
            ax.zaxis.label.set_color('white')
            ax.xaxis.set_pane_color((0.2, 0.2, 0.2, 1.0))
            ax.yaxis.set_pane_color((0.2, 0.2, 0.2, 1.0))
            ax.zaxis.set_pane_color((0.2, 0.2, 0.2, 1.0))
            ax.grid(True, linestyle=':', alpha=0.4, color="#777")

    def start_animation(self, mode):
        is_3d = "3D" in mode
        self.ax = self.fig.add_subplot(111, projection='3d' if is_3d else None)

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
        self.btn_start.config(state="normal", text="START SIMULARE")
        self.log("Finalizat.")

        mode = self.problem_mode.get()
        self.fig.clf()
        is_3d = "3D" in mode
        self.ax = self.fig.add_subplot(111, projection='3d' if is_3d else None)

        if mode == "Wi-Fi 2D":
            self._draw_final_wifi_2d()
        elif mode == "Wi-Fi 3D":
            self._draw_final_wifi_3d()
        elif mode == "Pathfinding 2D":
            self._draw_final_path_2d()
        elif mode == "Pathfinding 3D":
            self._draw_final_path_3d()
        self.canvas.draw()

    def draw_comparison_chart(self):
        self.is_running = False
        self.btn_start.config(state="normal", text="GENEREAZĂ GRAFIC")
        self.ax = self.fig.add_subplot(111)
        self._style_axes(self.ax, "Convergență PSO: Global vs Local")
        self.ax.set_xlabel("Iterații");
        self.ax.set_ylabel("Cost")

        styles = {'global': ('#ff5252', '-'), 'social': ('#69f0ae', '--'), 'geographic': ('#448aff', '-.')}
        for top, data in self.comparison_results.items():
            color, style = styles.get(top, ('white', '-'))
            self.ax.plot(data, label=top.capitalize(), color=color, linestyle=style, lw=2)
        self.ax.legend(facecolor="#444", labelcolor="white", edgecolor="#555")
        self.canvas.draw()

    def _anim_path_2d(self):
        prob = self.problem_instance;
        self._style_axes(self.ax, "Simulare Pathfinding 2D")
        self.ax.set_xlim(0, 100);
        self.ax.set_ylim(0, 100)
        for (ox, oy, r) in prob.obstacles: self.ax.add_patch(patches.Circle((ox, oy), r, color='#555', alpha=0.8))
        self.ax.plot(*prob.start, 'gs', ms=10, zorder=5);
        self.ax.plot(*prob.end, 'rx', ms=10, zorder=5)
        lines = [self.ax.plot([], [], color=C["accent"], alpha=0.3)[0] for _ in range(len(self.history[0]))]

        def update(frame):
            if frame == len(self.history) - 1: self.root.after(100, self.finish_sequence)
            for i, p in enumerate(self.history[frame]):
                full = np.vstack([prob.start, p.reshape((prob.num_waypoints, 2)), prob.end])
                lines[i].set_data(full[:, 0], full[:, 1])

        self.current_anim = animation.FuncAnimation(self.fig, update, frames=len(self.history), interval=30,
                                                    repeat=False)

    def _anim_path_3d(self):
        prob = self.problem_instance;
        self._style_axes(self.ax, "Simulare Pathfinding 3D", True)
        self.ax.set_xlim(0, 100);
        self.ax.set_ylim(0, 100);
        self.ax.set_zlim(0, 100)
        for (ox, oy, oz, r) in prob.obstacles: self.ax.scatter(ox, oy, oz, s=r * 20, c='#555', alpha=0.3)
        lines = [self.ax.plot([], [], [], color=C["accent"], alpha=0.3)[0] for _ in
                 range(min(20, len(self.history[0])))]

        def update(frame):
            if frame == len(self.history) - 1: self.root.after(100, self.finish_sequence)
            for i, l in enumerate(lines):
                if i >= len(self.history[frame]): break
                full = np.vstack([prob.start, self.history[frame][i].reshape((prob.num_waypoints, 3)), prob.end])
                l.set_data(full[:, 0], full[:, 1]);
                l.set_3d_properties(full[:, 2])

        self.current_anim = animation.FuncAnimation(self.fig, update, frames=len(self.history), interval=60,
                                                    repeat=False)

    def _anim_wifi_2d(self):
        prob = self.problem_instance;
        self._style_axes(self.ax, "Simulare Wi-Fi")
        self.ax.set_xlim(0, 100);
        self.ax.set_ylim(0, 100)
        for (wx, wy, w, h) in prob.walls: self.ax.add_patch(patches.Rectangle((wx, wy), w, h, facecolor='#666'))
        scat = self.ax.scatter([], [], c=C["accent"])

        def update(frame):
            if frame == len(self.history) - 1: self.root.after(100, self.finish_sequence)
            pts = [p.reshape((prob.n_routers, 2)) for p in self.history[frame]]
            scat.set_offsets(np.vstack(pts))

        self.current_anim = animation.FuncAnimation(self.fig, update, frames=len(self.history), interval=50,
                                                    repeat=False)

    def _anim_wifi_3d(self):
        prob = self.problem_instance;
        self._style_axes(self.ax, "Simulare Wi-Fi 3D", True)
        self.ax.set_xlim(0, 100);
        self.ax.set_ylim(0, 100);
        self.ax.set_zlim(0, 100)
        scat = self.ax.scatter([], [], [], c=C["accent"])

        def update(frame):
            if frame == len(self.history) - 1: self.root.after(100, self.finish_sequence)
            all_r = np.vstack([p.reshape((prob.n_routers, 3)) for p in self.history[frame]])
            scat._offsets3d = (all_r[:, 0], all_r[:, 1], all_r[:, 2])

        self.current_anim = animation.FuncAnimation(self.fig, update, frames=len(self.history), interval=60,
                                                    repeat=False)

    def _draw_final_path_2d(self):
        prob = self.problem_instance;
        self._style_axes(self.ax, "Rezultat Final")
        self.ax.set_xlim(0, 100);
        self.ax.set_ylim(0, 100)
        for (ox, oy, r) in prob.obstacles: self.ax.add_patch(patches.Circle((ox, oy), r, color='#555', alpha=0.9))
        full = np.vstack([prob.start, self.best_pos.reshape((prob.num_waypoints, 2)), prob.end])
        self.ax.plot(full[:, 0], full[:, 1], color=C["accent"], lw=3, label="Traseu Optim")
        self.ax.plot(*prob.start, 'gs', ms=12);
        self.ax.plot(*prob.end, 'rx', ms=12)
        self.ax.legend(facecolor=C["bg_side"], labelcolor="white")

    def _draw_final_path_3d(self):
        prob = self.problem_instance;
        self._style_axes(self.ax, "Rezultat Final 3D", True)
        for (ox, oy, oz, r) in prob.obstacles:
            u, v = np.mgrid[0:2 * np.pi:15j, 0:np.pi:15j]
            self.ax.plot_wireframe(ox + r * np.cos(u) * np.sin(v), oy + r * np.sin(u) * np.sin(v), oz + r * np.cos(v),
                                   color="#555", alpha=0.2)
        full = np.vstack([prob.start, self.best_pos.reshape((prob.num_waypoints, 3)), prob.end])
        self.ax.plot(full[:, 0], full[:, 1], full[:, 2], color=C["accent"], lw=3)
        self.ax.scatter(*prob.start, c='green', s=100);
        self.ax.scatter(*prob.end, c='red', s=100)

    def _draw_final_wifi_2d(self):
        prob = self.problem_instance;
        self._style_axes(self.ax, "Heatmap Final")
        self.ax.set_xlim(0, 100);
        self.ax.set_ylim(0, 100)
        x = np.linspace(0, 100, 100);
        y = np.linspace(0, 100, 100);
        X, Y = np.meshgrid(x, y);
        Z = np.zeros_like(X)
        routers = self.best_pos.reshape((prob.n_routers, 2))
        for i in range(100):
            for j in range(100):
                md = min([np.linalg.norm([X[i, j] - r[0], Y[i, j] - r[1]]) for r in routers])
                if md < prob.radius: Z[i, j] = 1 - (md / prob.radius)
        self.ax.imshow(Z, extent=(0, 100, 0, 100), origin='lower', cmap='viridis', alpha=0.8)
        for (wx, wy, w, h) in prob.walls: self.ax.add_patch(patches.Rectangle((wx, wy), w, h, facecolor='#444'))
        for r in routers:
            self.ax.scatter(*r, c='red', marker='^', s=100, edgecolors='white')
            self.ax.add_patch(patches.Circle(r, prob.radius, fill=False, edgecolor='white', ls='--', alpha=0.5))

    def _draw_final_wifi_3d(self):
        prob = self.problem_instance;
        self._style_axes(self.ax, "Acoperire 3D Final", True)
        routers = self.best_pos.reshape((prob.n_routers, 3))
        u = np.linspace(0, 2 * np.pi, 15);
        v = np.linspace(0, np.pi, 15)
        for r in routers:
            self.ax.scatter(*r, c='red', s=100, marker='^')
            x = r[0] + prob.radius * np.outer(np.cos(u), np.sin(v))
            y = r[1] + prob.radius * np.outer(np.sin(u), np.sin(v))
            z = r[2] + prob.radius * np.outer(np.ones(np.size(u)), np.cos(v))
            self.ax.plot_wireframe(x, y, z, color=C["accent"], alpha=0.2)


if __name__ == "__main__":
    root = tk.Tk()
    app = PSOInterface(root)
    root.mainloop()