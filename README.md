# PSO Visualization Studio

**PSO Visualization Studio** este o aplicație desktop interactivă, dezvoltată în Python, care demonstrează puterea algoritmului **Particle Swarm Optimization (PSO)**. Aplicația permite vizualizarea în timp real a modului în care un roi de particule rezolvă probleme complexe de optimizare în spații 2D și 3D.

## Funcționalități Principale

### 1. Scenarii de Simulare
* **Pathfinding (2D & 3D):** Găsirea celui mai scurt traseu între două puncte, ocolind obstacole statice.
* **Wi-Fi Coverage (2D & 3D):** Optimizarea poziționării routerelor într-o incintă pentru a maximiza acoperirea semnalului.

### 2. Configurare Avansată
Interfața permite modificarea parametrilor algoritmului în timp real:
* **Număr Particule:** Dimensiunea roiului.
* **Număr Iterații:** Durata simulării.
* **Complexitate:** Numărul de obstacole sau routere din scenariu.

### 3. Topologii PSO
Suport pentru diferite strategii de comunicare între particule:
* **Global:** Toate particulele comunică între ele (convergență rapidă).
* **Social:** Particulele comunică doar cu vecinii din index (rețea inelară).
* **Geographic:** Particulele comunică doar cu vecinii fizici apropiați (menține diversitatea).

### 4. Studiu Comparativ
Un modul dedicat care rulează automat algoritmul pe toate cele 3 topologii și generează un grafic comparativ al costului (fitness) în funcție de iterații.

## Cerințe de Sistem

* **Python 3.10+**
* Librării necesare (instalate via `pip`):
    * `numpy`
    * `matplotlib`
    * `tkinter` (inclus de obicei în Python standard)

## Instalare

1.  **Clonează proiectul:**
    ```bash
    git clone <repository-url>
    cd ProiectIA
    ```

2.  **Creează un mediu virtual (recomandat):**
    ```bash
    python -m venv .venv
    # Windows:
    .venv\Scripts\activate
    # macOS/Linux:
    source .venv/bin/activate
    ```

3.  **Instalează dependențele:**
    ```bash
    pip install -r requirements.txt
    ```

## Utilizare

Pentru a porni aplicația, rulează scriptul principal din rădăcina proiectului:

```bash
python main.py
