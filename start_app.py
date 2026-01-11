import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

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
            os.system('python src/main.py')
        elif choice == '2':
            os.system('python src/run_pathfinding_3d.py')
        elif choice == '3':
            os.system('python src/run_wifi.py')
        elif choice == '4':
            os.system('python src/run_wifi_3d.py')
        elif choice == '5':
            os.system('python src/compare.py')
        elif choice == '0':
            print("Ceao")
            break
        else:
            print("Invalid!")


if __name__ == "__main__":
    main()