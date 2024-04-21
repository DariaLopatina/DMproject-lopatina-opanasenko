import numpy as np
import time

def generate_matrix(n, p, k):
    adj_matrix = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            if np.random.rand() <= p:
                weight = np.random.randint(1, k + 1)
                adj_matrix[i][j] = adj_matrix[j][i] = weight

    return adj_matrix

class kruskalalgorithm:
    def __init__(self, adj_matrix, n):
        self.adj_matrix = adj_matrix
        self.len = n
        self.result_matrix = np.zeros((n, n), dtype=int)
        self.visited = [False] * self.len
        self.result_edge = []

    def find_minedge(self):
        min_weight = float('inf')
        min_edge = None

        for i in range(self.len):
            for j in range(i + 1, self.len):
                if (not (self.visited[i] and self.visited[j]) or not ((i,j) in self.result_edge)) and self.adj_matrix[i][j] != 0:
                    if self.adj_matrix[i][j] < min_weight:
                        min_weight = self.adj_matrix[i][j]
                        min_edge = (i, j)

        self.result_edge.append(min_edge)
        return min_edge, min_weight

    def update(self, a, b, min_weight):
        self.visited[a] = True
        self.visited[b] = True

        self.result_matrix[a][b] = self.result_matrix[b][a] = min_weight

    def dfs(self, visited, start):
        stack = [start]
        while stack:
            node = stack.pop()
            if not visited[node]:
                visited[node] = True
                for i in range(self.len):
                    if self.result_matrix[node][i] != 0 and not visited[i]:
                        stack.append(i)

    def сonnectivity(self):
        visited = [False] * self.len
        self.dfs(visited, 0)
        return all(visited)

    def kruskal(self):
        total_weight = 0

        while not all(self.visited) or not(self.сonnectivity()):
            min_edge, min_weight = self.find_minedge()
            if min_edge is None:
                break

            a, b = min_edge
            self.update(a, b, min_weight)
            total_weight += min_weight

        return self.result_matrix, total_weight

numbers = range(25, 201, 25) #вказує на кількість вершин в графі
density = [0.5, 0.6, 0.7, 0.8, 0.9, 1] #вказує на щільність
k = 10 #вказує на проміжок, в якому буде генеруватися вага для ребер

for n in numbers:
    for p in density:
        avrg_time = 0
        for i in range(20):
            adj_matrix = generate_matrix(n, p, k)
            start_time = time.time()
            kruskal_init = kruskalalgorithm(adj_matrix, n)
            result_matrix, total_weight = kruskal_init.kruskal()

            elapsed_time = time.time() - start_time
            avrg_time += elapsed_time
        avrg_time /= 20
        print(f"size: {n}, density: {p}, average time: {avrg_time:.6f} s")





