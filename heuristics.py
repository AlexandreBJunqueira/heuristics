import math
import random
from typing import List, Tuple


import math
from typing import List

class ConstructiveHeuristics:
    def tour_length(self, tour: List[int], D: List[List[float]]) -> float:
        return sum(D[tour[i]][tour[i+1]] for i in range(len(tour)-1))

    # -----------------------------
    # 1) Nearest Neighbor (NN)
    # -----------------------------
    def nearest_neighbor(self, D: List[List[float]], start: int = 0) -> List[int]:
        n = len(D)
        unvisited = set(range(n))
        tour = [start]
        unvisited.remove(start)

        while unvisited:
            last = tour[-1]
            next_city = min(unvisited, key=lambda j: D[last][j])
            tour.append(next_city)
            unvisited.remove(next_city)

        tour.append(start)  # fechar ciclo
        return tour

    # -----------------------------
    # 2) Cheapest Insertion (CI)
    # -----------------------------
    def cheapest_insertion(self, D: List[List[float]]) -> List[int]:
        n = len(D)
        # 1. começa com o par mais próximo
        min_pair = min(((i, j) for i in range(n) for j in range(n) if i != j),
                       key=lambda x: D[x[0]][x[1]])
        tour = [min_pair[0], min_pair[1], min_pair[0]]
        unvisited = set(range(n)) - set(min_pair)

        # 2. insere cidades até completar o tour
        while unvisited:
            best_increase = math.inf
            best_city, best_pos = None, None
            for city in unvisited:
                # tenta inserir city entre (i, j)
                for i in range(len(tour)-1):
                    a, b = tour[i], tour[i+1]
                    delta = D[a][city] + D[city][b] - D[a][b]
                    if delta < best_increase:
                        best_increase = delta
                        best_city, best_pos = city, i+1
            tour.insert(best_pos, best_city)
            unvisited.remove(best_city)

        return tour

    # -----------------------------
    # 3) Farthest Insertion (FI)
    # -----------------------------
    def farthest_insertion(self, D: List[List[float]]) -> List[int]:
        n = len(D)

        # 1. começa com o par mais distante
        max_pair = max(((i, j) for i in range(n) for j in range(n) if i != j),
                       key=lambda x: D[x[0]][x[1]])
        tour = [max_pair[0], max_pair[1], max_pair[0]]
        unvisited = set(range(n)) - set(max_pair)

        # 2. insere cidades até completar o tour
        while unvisited:
            # a próxima cidade é a mais distante de qualquer cidade já no tour
            farthest_city = max(unvisited, key=lambda c: min(D[c][t] for t in tour))

            # insere onde o aumento de custo for mínimo
            best_increase = math.inf
            best_pos = None
            for i in range(len(tour)-1):
                a, b = tour[i], tour[i+1]
                delta = D[a][farthest_city] + D[farthest_city][b] - D[a][b]
                if delta < best_increase:
                    best_increase = delta
                    best_pos = i+1

            tour.insert(best_pos, farthest_city)
            unvisited.remove(farthest_city)

        return tour


# -----------------------------
# Teste das heurísticas
# -----------------------------
if __name__ == "__main__":
    from tsp import TSP
    N = 20
    constructive_heuristics, tsp = ConstructiveHeuristics(), TSP()
    cities = tsp.generate_cities(N)
    D = tsp.build_distance_matrix(cities, True)

    # Heurística 1: Nearest Neighbor
    tour_nn = constructive_heuristics.nearest_neighbor(D, start=0)
    cost_nn = constructive_heuristics.tour_length(tour_nn, D)
    print("Nearest Neighbor:")
    print(f"Rota: {tour_nn}")
    print(f"Custo: {cost_nn:.2f}\n")

    # Heurística 2: Cheapest Insertion
    tour_ci = constructive_heuristics.cheapest_insertion(D)
    cost_ci = constructive_heuristics.tour_length(tour_ci, D)
    print("Cheapest Insertion:")
    print(f"Rota: {tour_ci}")
    print(f"Custo: {cost_ci:.2f}\n")

    # Heurística 3: Farthest Insertion
    tour_ci = constructive_heuristics.farthest_insertion(D)
    cost_ci = constructive_heuristics.tour_length(tour_ci, D)
    print("Farthest Insertion:")
    print(f"Rota: {tour_ci}")
    print(f"Custo: {cost_ci:.2f}\n")