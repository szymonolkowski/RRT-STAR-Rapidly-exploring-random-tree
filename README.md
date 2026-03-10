Oto zaktualizowana wersja pliku `README.md`, w której zintegrowałem Twoje wizualizacje. Użyłem standardowej składni Markdown dla obrazów, korzystając z nazw plików, które przesłałeś.

Aby obrazy wyświetlały się poprawnie na GitHubie, upewnij się, że pliki `image_41b119.png` oraz `image_41b0dd.png` znajdują się w tym samym folderze co Twój plik `README.md` (lub zaktualizuj ścieżki, jeśli wrzucisz je do folderu np. `docs/` lub `img/`).

---

# RRT* Path Planner ze Wsparciem dla Pól Potencjałowych 🚀

Zaawansowana implementacja algorytmu **RRT*** (Rapidly-exploring Random Tree Star) w języku Python. Algorytm służy do wyznaczania optymalnej i bezkolizyjnej ścieżki w przestrzeni 2D (współrzędne NED - North-East), dedykowany dla robotyki mobilnej, dronów lub pojazdów autonomicznych (np. ASV - Autonomous Surface Vehicles).

*Rozrost drzewa RRT* (zielone punkty) z uwzględnieniem aktywnych obszarów (błękitne wielokąty). Czerwona linia prezentuje ostateczną, wygładzoną ścieżkę omijającą strefy wpływu przeszkód.*

Wyróżnia się na tle klasycznego RRT* zintegrowanym mechanizmem Sztucznych Pól Potencjałowych (Artificial Potential Fields) oraz inteligentnym próbkowaniem przestrzeni z wykorzystaniem triangulacji Delaunay'a.

## ✨ Kluczowe funkcjonalności

* **Asymptotyczna optymalność (RRT*)**: Algorytm nie tylko szuka pierwszej lepszej ścieżki (jak standardowe RRT), ale z czasem optymalizuje ją, "przepinając" węzły (rewiring) w celu minimalizacji całkowitego kosztu.
* **Sztuczne Pola Potencjałowe (Potential Fields)**: Przeszkody generują "strefy wpływu" (influence zones). Ścieżki przebiegające zbyt blisko przeszkód otrzymują drastyczne kary do kosztu przejścia, co naturalnie odpycha ścieżkę od zagrożeń.
* **Inteligentne próbkowanie (Delaunay Heuristics)**: Zamiast losować punkty w całej przestrzeni, algorytm potrafi wykorzystać podaną siatkę Delaunay'a z wagami, skupiając się na trójkątach leżących na kierunku do celu.

*Zastosowanie algorytmu do nawigacji pomiędzy punktami (np. bojami). Widoczna siatka triangulacji Delaunay'a ograniczająca przestrzeń poszukiwań oraz wektory kierunku (heading) na wygenerowanej ścieżce.*

* **Dynamiczny limit węzłów (Target Node Count)**: Kalkuluje optymalną liczbę węzłów docelowych na podstawie powierzchni obszaru poszukiwań i zadanego kroku (`step_size`).
* **Early Stopping**: Algorytm potrafi zatrzymać się przed osiągnięciem limitu węzłów, jeśli koszt przestał się poprawiać przez odpowiednio długi czas, oszczędzając zasoby obliczeniowe.
* **Wygładzanie trasy (Path Smoothing)**: Moduł post-processingu usuwający niepotrzebne "zygzaki" typowe dla RRT* za pomocą raycastingu i redukcji gęsto upakowanych punktów.
* **Tolerancja kolizji startowej**: Jeśli punkt startowy znajduje się wewnątrz przeszkody, algorytm ignoruje tę konkretną przeszkodę tylko dla pierwszego ruchu, pozwalając robotowi "wyjechać" z kolizji.

## 🛠 Wymagania

Algorytm wykorzystuje standardowe biblioteki środowiska Python:

* `numpy`
* `math` (wbudowana)
* `random` (wbudowana)

## 📦 Struktura danych

Aby algorytm zadziałał, wymaga przekazania obiektów w specyficznym formacie:

1. **Start i Cel (`start`, `goal`)**: Obiekty muszą posiadać atrybut `ned`, w którym znajdują się współrzędne `n` (North) i `e` (East).
2. **Przeszkody (`potential_fields`)**: Lista obiektów posiadających współrzędne środka `n`, `e` oraz promień `r`.

## 🚀 Użycie (Szybki Start)

```python
import numpy as np
# Zakładając, że klasy Node i RRTStar znajdują się w pliku rrt_star.py
from rrt_star import RRTStar

# 1. Przygotowanie mockowych struktur dla startu i celu
class Coordinate:
    def __init__(self, n, e):
        self.n = n
        self.e = e

class Waypoint:
    def __init__(self, n, e):
        self.ned = Coordinate(n, e)

class Obstacle:
    def __init__(self, n, e, r):
        self.n = n
        self.e = e
        self.r = r

start_node = Waypoint(0.0, 0.0)
goal_node = Waypoint(20.0, 20.0)
obstacles = [Obstacle(10.0, 10.0, 3.0), Obstacle(5.0, 15.0, 2.0)]

# 2. Inicjalizacja plannera
planner = RRTStar(
    start=start_node,
    goal=goal_node,
    delaunay=None,            # Opcjonalny obiekt z triangulacją scipy.spatial.Delaunay
    weighted_random=None,     # Opcjonalne wagi dla trójkątów
    potential_fields=obstacles,
    step_size=1.0,
    iter_scale=10,
    safety_margin=0.5,        # Dodatkowy bufor bezpieczeństwa wokół przeszkód
    repulsive_weight=20.0     # Siła "odpychania" pól potencjałowych
)

# 3. Uruchomienie planowania (z opcjonalnym wygładzaniem ścieżki)
planner.plan(post_processing=True)

# 4. Pobranie wyników
if planner.path:
    print(f"Ścieżka znaleziona! Długość (węzły): {len(planner.path)}")
    print(f"Najlepszy koszt: {planner.best_cost:.2f}")
    for node in planner.path:
        print(f" -> N: {node.n:.2f}, E: {node.e:.2f}")
else:
    print("Nie udało się znaleźć ścieżki.")

```

## ⚙️ Parametryzacja klasy `RRTStar`

* `step_size`: (float) Maksymalna długość pojedynczej gałęzi drzewa.
* `iter_scale`: (int) Mnożnik wpływający na dynamiczne wyliczanie docelowej liczby węzłów.
* `safety_margin`: (float) Margines bezpieczeństwa dodawany do promienia każdej przeszkody.
* `repulsive_weight`: (float) Waga kary dla funkcji pól potencjałowych. Wpływa na wzór kosztu: $penalty = 0.5 \times weight \times term^2$.
* `goal_region_radius`: (float) Promień akceptacji celu. Węzeł w tym promieniu uznawany jest za sukces.
* `search_radius`: (float) Promień w jakim RRT* szuka sąsiadów do procedury "rewiring".

---

Czy układ tych obrazów Ci odpowiada, czy wolałbyś przenieść któryś z nich na sam dół do specjalnej sekcji "Galeria" / "Przykłady użycia"?
