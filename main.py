import sys
import os
import tkinter as tk
from tkinter import ttk

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.ui.desktop_app import PSOInterface

if __name__ == "__main__":
    root = tk.Tk()

    style = ttk.Style()
    style.theme_use('clam')

    app = PSOInterface(root)
    root.mainloop()