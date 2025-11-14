####### TSP #############
import gurobipy as gp
from gurobipy import GRB
import math
import random
import time

class TSP:
    # -----------------------------
    # 1) Gerar dados
    # -----------------------------
    def generate_cities(self, n, seed=42, x_range=(0, 100), y_range=(0, 100)):
        random.seed(seed)
        cities = [(random.uniform(*x_range), random.uniform(*y_range)) for _ in range(n)]
        return cities

    def euclidean(self, a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    # -----------------------------
    # 2) Construir matriz de distâncias
    # -----------------------------
    def build_distance_matrix(self, cities, list=False):
        if list:
            n = len(cities)
            D = [[0.0]*n for _ in range(n)]
            for i in range(n):
                for j in range(i+1, n):
                    d = self.euclidean(cities[i], cities[j])
                    D[i][j] = d
                    D[j][i] = d
            return D
        else:
            n = len(cities)
            dist = {(i, j): self.euclidean(cities[i], cities[j]) if i != j else 0 for i in range(n) for j in range(n)}
            return dist

    # -----------------------------
    # 3) Resolver TSP com Gurobi (MTZ formulation)
    # -----------------------------
    def solve_tsp_gurobi(self, D):
        n = len(D)
        model = gp.Model("TSP")
        model.Params.OutputFlag = 0  # silenciar

        x = model.addVars(n, n, vtype=GRB.BINARY, name="x")
        model.setObjective(gp.quicksum(D[i][j]*x[i,j] for i in range(n) for j in range(n)), GRB.MINIMIZE)

        for i in range(n):
            model.addConstr(gp.quicksum(x[i,j] for j in range(n) if j != i) == 1)
            model.addConstr(gp.quicksum(x[j,i] for j in range(n) if j != i) == 1)

        u = model.addVars(n, vtype=GRB.CONTINUOUS, lb=0, ub=n-1)
        for i in range(1, n):
            for j in range(1, n):
                if i != j:
                    model.addConstr(u[i] - u[j] + (n-1)*x[i,j] <= n-2)

        t0 = time.time()
        model.optimize()
        t1 = time.time()

        if model.status == GRB.OPTIMAL:
            edges = [(i,j) for i in range(n) for j in range(n) if x[i,j].X > 0.5]
            tour = [0]
            current = 0
            while len(tour) < n:
                for (a,b) in edges:
                    if a == current and b not in tour:
                        tour.append(b)
                        current = b
                        break
            tour.append(0)
            return model.objVal, tour, t1 - t0
        else:
            return None, None, None

########### NEarest Neighbor Heuristic #############

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


################# Heurísticas de Melhoria #################
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
        GRAFICO = []

        while improved:
            improved = False
            for i in range(1, n - 2):
                for j in range(i + 1, n - 1):
                    new_tour = best_tour[:i] + best_tour[i:j][::-1] + best_tour[j:]
                    new_cost = self.tour_length(new_tour, D)
                    if new_cost < best_cost - 1e-9:
                        best_tour, best_cost = new_tour, new_cost
                        improved = True
                        GRAFICO.append(best_cost)
                        break
                if improved:
                    break
        return best_tour, best_cost, GRAFICO

    # -------------------------------------
    # 2) Heurística de melhoria 3-opt
    # -------------------------------------
    def three_opt(self, tour: List[int], D: List[List[float]]) -> Tuple[List[int], float]:
        n = len(tour)
        best_tour = tour[:]
        best_cost = self.tour_length(tour, D)
        improved = True
        GRAFICO = []

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
                                GRAFICO.append(best_cost)
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        return best_tour, best_cost, GRAFICO

    def fob(self, tour: List[int], D: List[List[float]], num_iteracoes: int = 50, *args, **kwargs) -> Tuple[List[int], float, List[float]]:
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
        history = []
        descarte = []

        for it in range(num_iteracoes):
            improved = False

            # 1) tenta 2-opt (melhorias locais rápidas)
            new_tour, new_cost, descarte = self.two_opt(best_tour, D)
            if new_cost < best_cost - 1e-9:
                best_tour, best_cost = new_tour, new_cost
                history.append(best_cost)
                improved = True
                # volta ao começo para procurar mais melhorias a partir do novo tour
                continue

            # 2) se não houve melhoria com 2-opt, tenta 3-opt (mais potência)
            new_tour, new_cost, descarte = self.three_opt(best_tour, D)
            if new_cost < best_cost - 1e-9:
                best_tour, best_cost = new_tour, new_cost
                history.append(best_cost)
                improved = True
                continue

            # 3) nenhuma melhoria por 2-opt/3-opt: encerra
            if not improved:
                break

        return best_tour, best_cost, history




########### MAIN #############

if __name__ == "__main__":
    import pandas as pd
    df_1_iteração = pd.DataFrame(columns=["melhor_valor"])

    N=40
    tsp = TSP()
    cities = tsp.generate_cities(N)
    D = tsp.build_distance_matrix(cities, list=True)
    # Aplicar Heurística Nearest Neighbor e registrar custo
    heur = ConstructiveHeuristics()
    tour_nn = heur.nearest_neighbor(D, start=0)
    cost_nn = heur.tour_length(tour_nn, D)

    improv = ImprovementHeuristics()
    tour_fob, cost_fob, GRAFICO = improv.fob(tour_nn, D)
    GRAFICO.insert(0, cost_nn)
    df_1_iteração = pd.DataFrame(GRAFICO, columns=["melhor_valor"])
    print(df_1_iteração)
    df_1_iteração.to_csv("data/grafico_1_interacao_fob.csv", index=False)