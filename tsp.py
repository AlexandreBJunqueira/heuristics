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

# -----------------------------
# 5) Exemplo de uso
# -----------------------------
if __name__ == "__main__":
    N = 40  # número de cidades
    tsp = TSP()
    cities = tsp.generate_cities(N)
    cost, tour, delta_t = tsp.solve_tsp_gurobi(cities)
