#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <limits.h>
#include <time.h>

// Функция для поиска вершины с минимальным "весом" (ключом) среди непосещённых
int minweight(int *weight, bool *visited, int n) {
    int min = INT_MAX;
    int min_index = -1;

    for (int v = 0; v < n; v++) {
        if (!visited[v] && weight[v] < min) {
            min = weight[v];
            min_index = v;
        }
    }
    return min_index;
}

// Функция для печати результата (рёбра MST и их суммарная стоимость)
void printMST(int *parent, int **graph, int n) {
    int totalCost = 0;
    printf("\nMST for the generated graph:\n");
    printf("Edge   Weight\n");
    for (int i = 1; i < n; i++) {
        printf("%d - %d    %d\n", parent[i], i, graph[i][parent[i]]);
        totalCost += graph[i][parent[i]];
    }
    printf("Total cost of MST: %d\n", totalCost);
}

// Функция построения MST алгоритмом Прима (на последовательном коде)
void primMST(int **graph, int n) {
    // Массив для хранения MST
    int *parent = (int *)malloc(n * sizeof(int));
    // "Веса" (или ключи), используемые для выбора минимального ребра
    int *weight = (int *)malloc(n * sizeof(int));
    // Массив, отмечающий, включена ли вершина в MST
    bool *visited = (bool *)malloc(n * sizeof(bool));

    // Инициализация
    for (int i = 0; i < n; i++) {
        weight[i] = INT_MAX;
        visited[i] = false;
    }

    // Всегда включаем вершину 0 первой (можно любую другую)
    weight[0] = 0;   // чтобы выбрать её первой
    parent[0] = -1;  // у корневой вершины нет родителя

    // На каждом шаге добавляем по одной вершине в MST (всего n-1 добавлений)
    for (int count = 0; count < n - 1; count++) {
        // 1. Выбираем непосещённую вершину с минимальным значением weight[]
        int u = minweight(weight, visited, n);
        // 2. Помечаем её как посещённую
        visited[u] = true;

        // 3. Обновляем weight[] и parent[] для смежных вершин, 
        //    которые ещё не в MST
        for (int v = 0; v < n; v++) {
            // Если есть ребро (u,v), и v не посещён, и вес (u,v) меньше текущего weight[v]
            // то обновляем
            if (graph[u][v] != 0 && !visited[v] && graph[u][v] < weight[v]) {
                parent[v] = u;
                weight[v] = graph[u][v];
            }
        }
    }

    // Печатаем результат
    printMST(parent, graph, n);

    // Освобождаем ресурсы
    free(parent);
    free(weight);
    free(visited);
}

// Функция генерации псевдослучайного неориентированного взвешенного графа
// (матрица смежности). Веса от 1 до 100, диагональ = 0.
int **generateGraph(int n) {
    // Выделяем память под матрицу смежности
    int **graph = (int **)malloc(n * sizeof(int *));
    for (int i = 0; i < n; i++) {
        graph[i] = (int *)malloc(n * sizeof(int));
    }

    srand((unsigned)time(NULL));

    // Заполняем матрицу
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            if (i == j) {
                graph[i][j] = 0;
            } else {
                int w = rand() % 100 + 1; // случайный вес 1..100
                graph[i][j] = w;
                graph[j][i] = w; // симметрично, т.к. граф неориентированный
            }
        }
    }
    return graph;
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Usage: %s <num_vertices>\n", argv[0]);
        return 1;
    }

    int n = atoi(argv[1]); // число вершин

    // Генерируем граф
    int **graph = generateGraph(n);

    // Замер времени начала
    clock_t start = clock();

    // Запускаем алгоритм Прима
    primMST(graph, n);

    // Замер времени окончания
    clock_t end = clock();

    // Выводим затраченное время
    double total_time = (double)(end - start) / CLOCKS_PER_SEC;
    printf("\nNumber of vertices: %d\n", n);
    printf("Time taken by Prim's algorithm: %.6f seconds\n", total_time);

    // Освобождаем память под матрицу
    for (int i = 0; i < n; i++) {
        free(graph[i]);
    }
    free(graph);

    return 0;
}
