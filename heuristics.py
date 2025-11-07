import math
import random
from typing import List, Tuple
import math

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
    # 3) Lin-Kernighan simplificado (usa 2-opt e 3-opt)
    # -------------------------------------
    def lin_kernighan(self, tour: List[int], D: List[List[float]], max_iter: int = 50) -> Tuple[List[int], float, List[float]]:
        """
        Versão simplificada do Lin-Kernighan para melhoria de tour.

        Estratégia:
        - Iterativamente tenta melhorar o tour aplicando 2-opt e, se não houver
          melhoria, tenta 3-opt. Repete até não haver melhorias ou atingir
          max_iter iterações.

        Retorna (melhor_tour, melhor_custo, historico_de_costs)
        """
        n = len(tour)
        best_tour = tour[:]
        best_cost = self.tour_length(best_tour, D)
        history = [best_cost]

        for it in range(max_iter):
            improved = False

            # 1) tenta 2-opt (melhorias locais rápidas)
            new_tour, new_cost = self.two_opt(best_tour, D)
            if new_cost < best_cost - 1e-9:
                best_tour, best_cost = new_tour, new_cost
                history.append(best_cost)
                improved = True
                # volta ao começo para procurar mais melhorias a partir do novo tour
                continue

            # 2) se não houve melhoria com 2-opt, tenta 3-opt (mais potência)
            new_tour, new_cost = self.three_opt(best_tour, D)
            if new_cost < best_cost - 1e-9:
                best_tour, best_cost = new_tour, new_cost
                history.append(best_cost)
                improved = True
                continue

            # 3) nenhuma melhoria por 2-opt/3-opt: encerra
            if not improved:
                break

        return best_tour, best_cost, history

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
    

    """ def perturbation(self, tour: List[int]) -> List[int]:
        
        #Gera uma perturbação moderada/forte para escapar de ótimos locais.
        #Combina reversões, remoções e reinserções aleatórias de cidades.
        
        new_tour = tour[:]
        n = len(new_tour)

        # 1️⃣ - Decide a intensidade (5% a 20% do tamanho do tour)
        k = max(2, int(0.1 * n + random.random() * 0.1 * n))

        # 2️⃣ - Escolhe aleatoriamente k cidades para remover
        removed_indices = sorted(random.sample(range(1, n - 1), k), reverse=True)
        removed_cities = [new_tour[i] for i in removed_indices]
        for idx in removed_indices:
            del new_tour[idx]

        # 3️⃣ - Reinsere as cidades removidas em posições aleatórias
        for city in removed_cities:
            pos = random.randint(1, len(new_tour) - 1)
            new_tour.insert(pos, city)

        # 4️⃣ - Opcional: aplica uma reversão aleatória em um segmento grande
        if random.random() < 0.7:
            i, j = sorted(random.sample(range(1, n - 1), 2))
            new_tour[i:j] = reversed(new_tour[i:j])

        # 5️⃣ - Pequenas trocas adicionais (ruído extra)
        for _ in range(random.randint(1, 4)):
            a, b = random.sample(range(1, n - 1), 2)
            new_tour[a], new_tour[b] = new_tour[b], new_tour[a]

        return new_tour """

    # -------------------------------------
    # 4) Iterated Local Search (heurística + perturbação)
    # -------------------------------------
    def iterated_improvement(self, tour: List[int], D: List[List[float]], 
                             heuristic: str = "two_opt", num_iteracoes: int = 10,
                             lin_kernighan_max_iter: int = 50) -> Tuple[List[int], float, List[float]]:
        BEST_COSTS = []
        if heuristic not in ["two_opt", "three_opt", "lin_kernighan"]:
            raise ValueError("Heurística deve ser 'two_opt', 'three_opt' ou 'lin_kernighan'.")

        best_tour = tour[:]
        best_cost = self.tour_length(best_tour, D)

        for it in range(num_iteracoes):
            if heuristic == "two_opt":
                improved_tour, improved_cost = self.two_opt(best_tour, D)
            elif heuristic == "three_opt":
                improved_tour, improved_cost = self.three_opt(best_tour, D)
            else:  # lin_kernighan
                improved_tour, improved_cost, _history = self.lin_kernighan(best_tour, D, max_iter=lin_kernighan_max_iter)

            # Atualiza se melhorou
            if improved_cost < best_cost:
                best_tour, best_cost = improved_tour, improved_cost

            # Pequena perturbação para escapar de ótimo local
            best_tour = self.perturbation(best_tour)

            print(f"[Iter {it+1}] Custo atual: {best_cost:.2f}")
            BEST_COSTS.append(round(best_cost,2))

        return best_tour, best_cost, BEST_COSTS

# -----------------------------
# Teste das heurísticas
# -----------------------------
if __name__ == "__main__":
    from tsp import TSP
    from analysis import Analysis
    N = 20
    constructive_heuristics, tsp, analysis = ConstructiveHeuristics(), TSP(), Analysis()
    cities = tsp.generate_cities(N)
    D = tsp.build_distance_matrix(cities, True)

    # Heurística 1: Nearest Neighbor
    tour_nn = constructive_heuristics.nearest_neighbor(D, start=0)
    cost_nn = constructive_heuristics.tour_length(tour_nn, D)
    analysis.plot_tour(cities, tour_nn, 'Nearest Neighbor')
    print("Nearest Neighbor:")
    print(f"Rota: {tour_nn}")
    print(f"Custo: {cost_nn:.2f}\n")

    # Heurística 2: Cheapest Insertion
    tour_ci = constructive_heuristics.cheapest_insertion(D)
    cost_ci = constructive_heuristics.tour_length(tour_ci, D)
    analysis.plot_tour(cities, tour_ci, 'Cheapest Insertion')
    print("Cheapest Insertion:")
    print(f"Rota: {tour_ci}")
    print(f"Custo: {cost_ci:.2f}\n")

    # Heurística 3: Farthest Insertion
    tour_fi = constructive_heuristics.farthest_insertion(D)
    cost_fi = constructive_heuristics.tour_length(tour_fi, D)
    analysis.plot_tour(cities, tour_fi, 'Farthest Insertion')
    print("Farthest Insertion:")
    print(f"Rota: {tour_fi}")
    print(f"Custo: {cost_fi:.2f}\n")