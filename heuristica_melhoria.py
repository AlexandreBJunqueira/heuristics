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
    """ def perturbation(self, tour: List[int]) -> List[int]:
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
        return new_tour """
    

    def perturbation(self, tour: List[int]) -> List[int]:
        """
        Gera uma perturbação moderada/forte para escapar de ótimos locais.
        Combina reversões, remoções e reinserções aleatórias de cidades.
        """
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

        return new_tour

    # -------------------------------------
    # 4) Iterated Local Search (heurística + perturbação)
    # -------------------------------------
    def iterated_improvement(self, tour: List[int], D: List[List[float]], 
                             heuristic: str = "two_opt", num_iteracoes: int = 10) -> Tuple[List[int], float]:
        BEST_COSTS=[]
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
            BEST_COSTS.append(round(best_cost,2))

        return best_tour, best_cost, BEST_COSTS


# -------------------------------------
# Teste das heurísticas de melhoria com perturbação
# -------------------------------------
if __name__ == "__main__":
    from tsp import TSP
    from heuristics import ConstructiveHeuristics
    import os

    N = 40
    num_iteracoes =20 # ← você pode alterar aqui

    tsp = TSP()
    heur = ConstructiveHeuristics()
    improver = ImprovementHeuristics()

    cities = tsp.generate_cities(N)
    D = tsp.build_distance_matrix(cities, list=True)


    #Selecionar qual heurística construtiva usar
    tour_nn = heur.nearest_neighbor(D)
    cost_nn = heur.tour_length(tour_nn, D)
    print(f"Custo inicial (NN): {cost_nn:.2f}")

    # Aplicar busca iterada
    tour_final, cost_final ,lista_custos = improver.iterated_improvement(
        tour_nn, D, heuristic="two_opt", num_iteracoes=num_iteracoes
    )

    print(f"\nCusto final após {num_iteracoes} iterações: {cost_final:.2f}")
    lista_custos.insert(0,round(cost_nn,2))
    print(lista_custos)



    ###### Plotando gráficos

    x=[i for i in range(len(lista_custos))]
    y=lista_custos

    import matplotlib.pyplot as plt
    os.makedirs("imagens_melhoria", exist_ok=True)
    plt.plot(x, y, marker='o')
    plt.ylim(0, max(y)*1.05)
    plt.xlim(0, len(lista_custos))
    plt.xlabel("Iteração")
    plt.ylabel("Custo")
    plt.title("Evolução do custo durante Iterated Improvement")
    plt.grid(True)
    filename = os.path.join("imagens_melhoria", "custo_iteracoes.png")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"Gráfico salvo em: {filename}")
    plt.close()

    