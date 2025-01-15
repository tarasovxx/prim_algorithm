#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <stdbool.h>
#include <omp.h>
#include <mpi.h>

#define INF INT_MAX

//------------------------------------------------------------------------------
// Функции генерации и освобождения памяти
//------------------------------------------------------------------------------
int** gen_empty_graph(int size)
{
    int** graph = (int**)calloc(size, sizeof(int*));
    if (!graph) {
        fprintf(stderr, "Error: cannot allocate graph\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    for (int i = 0; i < size; ++i) {
        graph[i] = (int*)calloc(size, sizeof(int));
        if (!graph[i]) {
            fprintf(stderr, "Error: cannot allocate graph[%d]\n", i);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    return graph;
}

// Пример генерации взвешенного полного графа, похожего на ваш
int** gen_graph(int size)
{
    int** graph = gen_empty_graph(size);
    for (int i = 0; i < size; ++i)
    {
        for (int j = i + 1; j < size; ++j)
        {
            // Пример: вес = (size - j)
            int w = size - j;
            graph[i][j] = w;
            graph[j][i] = w;
        }
    }
    return graph;
}

// Освобождаем память
void free_graph(int** graph, int size)
{
    if (!graph) return;
    for (int i = 0; i < size; i++) {
        free(graph[i]);
    }
    free(graph);
}

//------------------------------------------------------------------------------
// Функция поиска вершины с минимальным ключом (используется в Приме)
//------------------------------------------------------------------------------
int find_min_key(int* key, bool* mst_set, int vertices)
{
    int min_val = INF, min_index = -1;
    for (int v = 0; v < vertices; v++)
    {
        if (!mst_set[v] && key[v] < min_val)
        {
            min_val = key[v];
            min_index = v;
        }
    }
    return min_index;
}

//------------------------------------------------------------------------------
// Параллельный алгоритм Прима (MPI + OpenMP)
//------------------------------------------------------------------------------
double parallel_prim_mst(int** graph, int vertices, int rank, int world_size)
{
    double start = MPI_Wtime();

    // Выделяем память под массивы
    int*  parent  = (int*) malloc(vertices * sizeof(int));
    int*  key     = (int*) malloc(vertices * sizeof(int));
    bool* mst_set = (bool*)malloc(vertices * sizeof(bool));

    // Инициализация
    for (int i = 0; i < vertices; i++)
    {
        key[i] = INF;
        mst_set[i] = false;
        parent[i] = -1;
    }
    // Начинаем с вершины 0
    key[0] = 0;
    parent[0] = -1;

    // Основной цикл Прима (vertices - 1 итерация)
    for (int count = 0; count < vertices - 1; count++)
    {
        int min_index = -1;

        // Только rank=0 находит вершину с минимальным key
        if (rank == 0)
            min_index = find_min_key(key, mst_set, vertices);

        // Раздаём min_index всем процессам
        MPI_Bcast(&min_index, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // Помечаем её как посещённую
        mst_set[min_index] = true;

        // Параллельно (OpenMP) обновляем key для оставшихся вершин
        #pragma omp parallel for
        for (int v = 0; v < vertices; v++)
        {
            if (graph[min_index][v] != 0 && !mst_set[v] && graph[min_index][v] < key[v])
            {
                key[v] = graph[min_index][v];
                parent[v] = min_index;
            }
        }
    }

    double finish = MPI_Wtime();

    free(parent);
    free(key);
    free(mst_set);

    return finish - start;  // Возвращаем время (в секундах)
}

//------------------------------------------------------------------------------
// Последовательный алгоритм Прима (работает ТОЛЬКО на rank=0)
//------------------------------------------------------------------------------
double linear_mst(int** graph, int vertices)
{
    double start = MPI_Wtime();

    int* parent  = (int*) malloc(vertices * sizeof(int));
    int* key     = (int*) malloc(vertices * sizeof(int));
    bool* mst_set= (bool*)malloc(vertices * sizeof(bool));

    for (int i = 0; i < vertices; i++)
    {
        key[i] = INF;
        mst_set[i] = false;
        parent[i] = -1;
    }
    key[0] = 0;

    for (int count = 0; count < vertices - 1; count++)
    {
        int min_index = find_min_key(key, mst_set, vertices);
        mst_set[min_index] = true;

        for (int v = 0; v < vertices; v++)
        {
            if (graph[min_index][v] && !mst_set[v] && graph[min_index][v] < key[v])
            {
                key[v] = graph[min_index][v];
                parent[v] = min_index;
            }
        }
    }
    double finish = MPI_Wtime();

    free(parent);
    free(key);
    free(mst_set);

    return finish - start; // сек
}

//------------------------------------------------------------------------------
// Основная функция
//------------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    // Инициализация MPI
    MPI_Init(&argc, &argv);

    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Для воспроизводимости можно установить кол-во потоков OpenMP
    // или задать через переменную окружения OMP_NUM_THREADS
    omp_set_num_threads(4);

    // Набор размеров графа для эксперимента
    // Можете менять или читать из argv
    int testSizes[] = {10, 100, 200, 400, 800, 1000, 10000, 100000, 1000000};
    int numTests = sizeof(testSizes) / sizeof(testSizes[0]);

    // Чтобы rank=0 выводил результаты в CSV-формате
    if (rank == 0) {
        // Заголовок CSV
        printf("n,parallel_time,linear_time\n");
    }

    // Прогоняем несколько тестов
    for (int i = 0; i < numTests; i++)
    {
        int n = testSizes[i];

        // Rank=0 генерирует матрицу
        int** graph = NULL;
        if (rank == 0) {
            graph = gen_graph(n);
        }

        // Рассылаем всем (n) и саму матрицу [n x n]
        // Сначала рассылаем размер
        MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // Создаём локальный graph у всех процессов
        // (даже если rank!=0, чтобы не падало при обращении).
        if (rank != 0) {
            graph = gen_empty_graph(n);
        }

        // Рассылаем (n) строк, каждая из (n) int
        for (int row = 0; row < n; row++) {
            MPI_Bcast(graph[row], n, MPI_INT, 0, MPI_COMM_WORLD);
        }

        // Измеряем параллельный алгоритм
        double timeParallel = parallel_prim_mst(graph, n, rank, world_size);

        // Измеряем последовательный алгоритм
        double timeLinear = 0.0;
        if (rank == 0) {
            // Только на rank=0 имеет смысл
            timeLinear = linear_mst(graph, n);
        }

        // С Rank=0 собираем и печатаем результат
        // Остальные rank просто освобождают память
        if (rank == 0) {
            printf("%d,%.6f,%.6f\n", n, timeParallel, timeLinear);
        }

        // Освободим память
        free_graph(graph, n);

        // Синхронизация между тестами (необязательно)
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // Завершаем
    MPI_Finalize();
    return 0;
}
