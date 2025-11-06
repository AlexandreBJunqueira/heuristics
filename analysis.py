import os
import math
import time
import csv
import matplotlib.pyplot as plt
import pandas as pd
from heuristics import ConstructiveHeuristics
from heuristica_melhoria import ImprovementHeuristics
from tsp import TSP


class Analysis:
    def __init__(self):
        self.tsp = TSP()
        self.h = ConstructiveHeuristics()
        self.i = ImprovementHeuristics()

        os.makedirs("images", exist_ok=True)
        os.makedirs("data", exist_ok=True)

    def _combine_methods(self, D, construction_method, improvement_method, type):
        tour = construction_method(D)
        results = improvement_method(tour, D, heuristic=type, num_iteracoes=10)[:2]
        return results[0], results[1]

    # -----------------------------
    # 1) Visualização
    # -----------------------------
    def plot_tour(self, points, tour, title, ax):
        xs = [points[i][0] for i in tour]
        ys = [points[i][1] for i in tour]
        ax.plot(xs, ys, '-o')
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    # -----------------------------
    # 2) Comparação única (para um n)
    # -----------------------------
    def compare_all(self, n=12, seed=42, save_outputs=True, use_gurobi=True):
        print(f"\n=== COMPARAÇÃO HEURÍSTICAS x ÓTIMO (n={n}) ===\n")

        points = self.tsp.generate_cities(n, seed)
        D = self.tsp.build_distance_matrix(points, True)
        h = self.h
        i = self.i
        results = []

        # -----------------------------
        # Ótimo via Gurobi
        # -----------------------------
        tour_opt = None
        cost_opt = None
        if use_gurobi:
            try:
                cost_opt, tour_opt, t_opt = self.tsp.solve_tsp_gurobi(D)
                results.append((seed, n, "Ótimo (Gurobi)", cost_opt, t_opt, 0.0))
                print(f"ÓTIMO (Gurobi): custo = {cost_opt:.2f} | tempo = {t_opt:.3f}s\n")
            except Exception as e:
                print(f"⚠️ Erro Gurobi em n={n}: {e}")
        else:
            print("⚙️ Gurobi desativado para este experimento.")
            cost_opt, tour_opt, t_opt = None, None, None

        # -----------------------------
        # Heurísticas construtivas
        # -----------------------------
        methods = {
            "Nearest Neighbor": h.nearest_neighbor,
            "Nearest Neighbor + 2-opt": [h.nearest_neighbor, i.iterated_improvement, 'two_opt'],
            "Nearest Neighbor + 3-opt": [h.nearest_neighbor, i.iterated_improvement, 'three_opt'],
            # "Cheapest Insertion": h.cheapest_insertion,
            # "Cheapest Insertion + 2-opt": [h.cheapest_insertion, i.iterated_improvement, 'two_opt'],
            # "Cheapest Insertion + 3-opt": [h.cheapest_insertion, i.iterated_improvement, 'three_opt'],
            # "Farthest Insertion": h.farthest_insertion,
            # "Farthest Insertion + 2-opt": [h.farthest_insertion, i.iterated_improvement, 'two_opt'],
            # "Farthest Insertion + 3-opt": [h.farthest_insertion, i.iterated_improvement, 'three_opt'],
        }

        for name, func in methods.items():
            t0 = time.time()
            if isinstance(func, list):
                tour, cost = self._combine_methods(D, *func)
            else:
                tour = func(D)
                cost = h.tour_length(tour, D)
            t1 = time.time()

            if cost_opt is not None:
                gap = 100 * (cost - cost_opt) / cost_opt
            else:
                gap = float("nan")

            results.append((seed, n, name, cost, t1 - t0, gap))
            print(f"{name:20s} | custo = {cost:.2f} | tempo = {t1 - t0:.3f}s | gap = {gap:.2f}%")

        # -----------------------------
        # Salvar CSV incremental sem duplicar resultados
        # -----------------------------
        if save_outputs:
            csv_path = "data/results.csv"
            file_exists = os.path.exists(csv_path)

            # Carregar CSV existente, se houver
            existing = pd.read_csv(csv_path) if file_exists else pd.DataFrame(
                columns=["seed", "n", "Método", "Custo", "Tempo (s)", "Gap (%)"]
            )

            with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["seed", "n", "Método", "Custo", "Tempo (s)", "Gap (%)"])

                for r in results:
                    seed_r, n_r, method_r = r[0], r[1], r[2]
                    # Verificar se já existe entrada igual
                    if not (
                        ((existing["seed"] == seed_r)
                         & (existing["n"] == n_r)
                         & (existing["Método"] == method_r)).any()
                    ):
                        writer.writerow(r)

        return results

    # -----------------------------
    # 3) Rodar experimentos
    # -----------------------------
    def run_experiments(self, n_min=2, n_max=50, seed=42, use_gurobi=True):
        all_results = []
        for n in range(n_min, n_max + 1):
            try:
                res = self.compare_all(n=n, seed=seed, save_outputs=True, use_gurobi=use_gurobi)
                all_results.extend(res)
            except Exception as e:
                print(f"⚠️ Falha em n={n}: {e}")

        print("\n✅ Experimentos concluídos.")
        print("Resultados salvos em 'data/results.csv'.")
        return all_results

    # -----------------------------
    # 4) Gráficos de médias por método
    # -----------------------------
    def plot_average_bars(self):
        df = pd.read_csv("data/results.csv")
        df = df[df["Método"] != "Ótimo (Gurobi)"]  # ignorar ótimo
        avg = df.groupby("Método").agg({"Custo": "mean", "Tempo (s)": "mean", "Gap (%)": "mean"}).reset_index()

        fig, ax = plt.subplots(1, 3, figsize=(14, 4))
        ax[0].bar(avg["Método"], avg["Custo"], color="skyblue")
        ax[1].bar(avg["Método"], avg["Tempo (s)"], color="lightgreen")
        ax[2].bar(avg["Método"], avg["Gap (%)"], color="salmon")

        ax[0].set_title("Custo médio")
        ax[1].set_title("Tempo médio (s)")
        ax[2].set_title("Gap médio (%)")
        for a in ax:
            a.tick_params(axis='x', rotation=15)
        plt.suptitle("Médias das heurísticas construtivas")
        plt.tight_layout()
        plt.savefig("images/average_bars.png", dpi=300)
        plt.show()

    # -----------------------------
    # 5) Evolução das métricas por n
    # -----------------------------
    def plot_evolution(self):
        df = pd.read_csv("data/results.csv")

        heuristics = [m for m in df["Método"].unique() if m != "Ótimo (Gurobi)"]

        fig, axs = plt.subplots(1, 3, figsize=(16, 4))
        metrics = ["Custo", "Tempo (s)", "Gap (%)"]
        colors = ["#007acc", "#2ca02c", "#ff7f0e"]

        for k, metric in enumerate(metrics):
            for i, hname in enumerate(heuristics):
                sub = df[df["Método"] == hname].groupby("n")[metric].mean().reset_index()
                axs[k].plot(sub["n"], sub[metric], label=hname, color=colors[i % len(colors)], linewidth=2)
            axs[k].set_title(metric)
            axs[k].set_xlabel("Número de cidades (n)")
            axs[k].grid(True)
            if metric == "Gap (%)":
                axs[k].legend()

        plt.suptitle("Evolução das métricas por heurística")
        plt.tight_layout()
        plt.savefig("images/evolution_metrics.png", dpi=300)
        plt.show()


# -----------------------------
# Execução principal
# -----------------------------
if __name__ == "__main__":
    analysis = Analysis()

    # # Exemplo 1: Rodar experimentos (sem Gurobi)
    # analysis.run_experiments(n_min=2, n_max=100, seed=42, use_gurobi=False)

    # # Exemplo 2: Rodar experimentos com Gurobi
    # analysis.run_experiments(n_min=2, n_max=100, seed=42, use_gurobi=True)

    # Exemplo 3: Gerar gráficos de médias e evolução a partir do CSV
    analysis.plot_average_bars()
    analysis.plot_evolution()