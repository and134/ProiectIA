import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src import main as app_pathfinding_2d
from src import run_pathfinding_3d
from src import run_wifi
from src import run_wifi_3d
from src import compare

def main():
    while True:
        print("\n=== MENIU PRINCIPAL ===")
        print("1. Pathfinding 2D (Animație)")
        print("2. Pathfinding 3D (Animație)")
        print("3. Wi-Fi Coverage 2D (Heatmap)")
        print("4. Wi-Fi Coverage 3D (Sfere Semnal)")
        print("5. Grafic Comparativ (Global vs Local)")
        print("0. Ieșire")

        choice = input("\nAlege o optiune: ")

        if choice == '1':
            app_pathfinding_2d.main()
        elif choice == '2':
            run_pathfinding_3d.run()
        elif choice == '3':
            run_wifi.run()
        elif choice == '4':
            run_wifi_3d.run()
        elif choice == '5':
            compare.run_comparison()
        elif choice == '0':
            print("Ceao")
            break
        else:
            print("Invalid!")


if __name__ == "__main__":
    main()