import math
import random
from typing import List, Tuple

class ImprovementHeuristics:
    # -------------------------------------
    # Função de utilidade: calcula o custo do tour
    # -------------------------------------
    def tour_length(self, tour: List[int], D: List[List[float]]) -> float:
        return sum(D[tour[i]][tour[i+1]] for i in range(len(tour) - 1))

    # -------------------------------------
    # 1) Heurística de melhoria 2-opt
    # -------------------------------------
    def two_opt(self, tour: List[int], D: List[List[float]]) -> Tuple[List[int], float]:
        n = len(tour)
        best_tour = tour[:]
        best_cost = self.tour_length(tour, D)
        improved = True

        while improved:
            improved = False
            for i in range(1, n - 2):
                for j in range(i + 1, n - 1):
                    new_tour = best_tour[:i] + best_tour[i:j][::-1] + best_tour[j:]
                    new_cost = self.tour_length(new_tour, D)
                    if new_cost < best_cost - 1e-9:
                        best_tour, best_cost = new_tour, new_cost
                        improved = True
                        break
                if improved:
                    break
        return best_tour, best_cost

    # -------------------------------------
    # 2) Heurística de melhoria 3-opt
    # -------------------------------------
    def three_opt(self, tour: List[int], D: List[List[float]]) -> Tuple[List[int], float]:
        n = len(tour)
        best_tour = tour[:]
        best_cost = self.tour_length(tour, D)
        improved = True

        while improved:
            improved = False
            for i in range(1, n - 4):
                for j in range(i + 1, n - 3):
                    for k in range(j + 1, n - 2):
                        segments = [
                            best_tour[:i] + best_tour[i:j][::-1] + best_tour[j:k][::-1] + best_tour[k:],
                            best_tour[:i] + best_tour[j:k] + best_tour[i:j] + best_tour[k:],
                            best_tour[:i] + best_tour[j:k] + best_tour[i:j][::-1] + best_tour[k:],
                            best_tour[:i] + best_tour[i:j] + best_tour[j:k][::-1] + best_tour[k:]
                        ]
                        for new_tour in segments:
                            new_cost = self.tour_length(new_tour, D)
                            if new_cost < best_cost - 1e-9:
                                best_tour, best_cost = new_tour, new_cost
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        return best_tour, best_cost

    # -------------------------------------
    # 3) Pequena perturbação aleatória
    # -------------------------------------
    def perturbation(self, tour: List[int]) -> List[int]:
        new_tour = tour[:]
        n = len(new_tour)
        i, j = sorted(random.sample(range(1, n - 1), 2))
        # aumentar chance de perturbação mais forte
        if random.random() < 0.5:
            new_tour[i:j] = reversed(new_tour[i:j])
        else:
            for _ in range(3):  # faz múltiplas trocas aleatórias
                a, b = random.sample(range(1, n - 1), 2)
                new_tour[a], new_tour[b] = new_tour[b], new_tour[a]
        return new_tour

    # -------------------------------------
    # 4) Iterated Local Search (heurística + perturbação)
    # -------------------------------------
    def iterated_improvement(self, tour: List[int], D: List[List[float]], 
                             heuristic: str = "two_opt", num_iteracoes: int = 10) -> Tuple[List[int], float]:
        if heuristic not in ["two_opt", "three_opt"]:
            raise ValueError("Heurística deve ser 'two_opt' ou 'three_opt'.")

        best_tour = tour[:]
        best_cost = self.tour_length(best_tour, D)

        for it in range(num_iteracoes):
            if heuristic == "two_opt":
                improved_tour, improved_cost = self.two_opt(best_tour, D)
            else:
                improved_tour, improved_cost = self.three_opt(best_tour, D)

            # Atualiza se melhorou
            if improved_cost < best_cost:
                best_tour, best_cost = improved_tour, improved_cost

            # Pequena perturbação para escapar de ótimo local
            best_tour = self.perturbation(best_tour)

            print(f"[Iter {it+1}] Custo atual: {best_cost:.2f}")

        return best_tour, best_cost


# -------------------------------------
# Teste das heurísticas de melhoria com perturbação
# -------------------------------------
if __name__ == "__main__":
    from tsp import TSP
    from heuristics import ConstructiveHeuristics

    N = 20
    num_iteracoes = 10  # ← você pode alterar aqui

    tsp = TSP()
    heur = ConstructiveHeuristics()
    improver = ImprovementHeuristics()

    cities = tsp.generate_cities(N)
    D = tsp.build_distance_matrix(cities, list=True)


    #Selecionar qual heurística construtiva usar
    tour_nn = heur.farthest_insertion(D)
    cost_nn = heur.tour_length(tour_nn, D)
    print(f"Custo inicial (NN): {cost_nn:.2f}")

    # Aplicar busca iterada (2-opt com perturbação)
    tour_final, cost_final = improver.iterated_improvement(
        tour_nn, D, heuristic="three_opt", num_iteracoes=num_iteracoes
    )

    print(f"\nCusto final após {num_iteracoes} iterações: {cost_final:.2f}")
