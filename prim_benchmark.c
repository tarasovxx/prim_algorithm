#include <stdio.h>
#include <stdlib.h>
#include <time.h>       // для rand()/srand()
#include <limits.h>
#include <stdbool.h>
#include <omp.h>
#include <mpi.h>

#define INF INT_MAX

//------------------------------------------------------------------------------
// Функция генерации "пустого" (нулевого) графа размером n x n
//------------------------------------------------------------------------------
int** gen_empty_graph(int n)
{
    int** graph = (int**)calloc(n, sizeof(int*));
    if (!graph) {
        fprintf(stderr, "[Error] cannot allocate graph\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    for (int i = 0; i < n; ++i) {
        graph[i] = (int*)calloc(n, sizeof(int));
        if (!graph[i]) {
            fprintf(stderr, "[Error] cannot allocate graph[%d]\n", i);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    return graph;
}

//------------------------------------------------------------------------------
// Генерация случайного графа c плотностью edges_density [0..1]
// и весами рёбер от 1 до maxWeight.
// graph[i][j] = 0, если ребра нет, иначе вес > 0
//------------------------------------------------------------------------------
int** gen_random_graph(int n, double edges_density, int maxWeight)
{
    // Создаём пустую матрицу n x n
    int** graph = gen_empty_graph(n);

    // Заполняем случайные рёбра
    // rand()/(double)RAND_MAX даёт [0..1)
    for (int i = 0; i < n; i++) {
        for (int j = i+1; j < n; j++) {
            double r = rand() / (double)RAND_MAX;
            if (r < edges_density) {
                // создаём ребро со случайным весом [1..maxWeight]
                int w = 1 + rand() % maxWeight;
                graph[i][j] = w;
                graph[j][i] = w;  // неориентированный граф
            } else {
                // 0 => нет ребра
                graph[i][j] = 0;
                graph[j][i] = 0;
            }
        }
    }
    return graph;
}

//------------------------------------------------------------------------------
// Освобождаем память
//------------------------------------------------------------------------------
void free_graph(int** graph, int n)
{
    if (!graph) return;
    for (int i = 0; i < n; i++) {
        free(graph[i]);
    }
    free(graph);
}

//------------------------------------------------------------------------------
// Функция поиска вершины с минимальным ключом (используется в Приме)
//------------------------------------------------------------------------------
int find_min_key(int* key, bool* mst_set, int n)
{
    int min_val = INF;
    int min_index = -1;
    for (int v = 0; v < n; v++)
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
double parallel_prim_mst(int** graph, int n, int rank, int world_size)
{
    double start = MPI_Wtime();

    // Выделяем память под массивы
    int*  parent  = (int*) malloc(n * sizeof(int));
    int*  key     = (int*) malloc(n * sizeof(int));
    bool* mst_set = (bool*)malloc(n * sizeof(bool));

    // Инициализация
    for (int i = 0; i < n; i++)
    {
        key[i] = INF;
        mst_set[i] = false;
        parent[i] = -1;
    }
    // Начинаем с вершины 0
    key[0] = 0;

    // (n - 1) итерация (алгоритм Прима)
    for (int count = 0; count < n - 1; count++)
    {
        int min_index = -1;

        // Только rank=0 ищет вершину с минимальным ключом
        if (rank == 0) {
            min_index = find_min_key(key, mst_set, n);
        }

        // Широковещательная рассылка выбранной вершины
        MPI_Bcast(&min_index, 1, MPI_INT, 0, MPI_COMM_WORLD);

        mst_set[min_index] = true;

        // Параллельно обновляем key[]
        #pragma omp parallel for
        for (int v = 0; v < n; v++)
        {
            int cost = graph[min_index][v];
            if (cost > 0 && !mst_set[v] && cost < key[v]) {
                key[v] = cost;
                parent[v] = min_index;
            }
        }
    }

    double finish = MPI_Wtime();

    free(parent);
    free(key);
    free(mst_set);

    return (finish - start);
}

//------------------------------------------------------------------------------
// Последовательный алгоритм Прима
//------------------------------------------------------------------------------
double linear_mst(int** graph, int n)
{
    double start = MPI_Wtime();

    int* parent  = (int*) malloc(n * sizeof(int));
    int* key     = (int*) malloc(n * sizeof(int));
    bool* mst_set= (bool*)malloc(n * sizeof(bool));

    for (int i = 0; i < n; i++)
    {
        key[i] = INF;
        mst_set[i] = false;
        parent[i] = -1;
    }
    key[0] = 0;

    for (int count = 0; count < n - 1; count++)
    {
        int min_index = find_min_key(key, mst_set, n);
        mst_set[min_index] = true;

        for (int v = 0; v < n; v++)
        {
            int cost = graph[min_index][v];
            if (cost > 0 && !mst_set[v] && cost < key[v]) {
                key[v] = cost;
                parent[v] = min_index;
            }
        }
    }

    double finish = MPI_Wtime();

    free(parent);
    free(key);
    free(mst_set);

    return (finish - start);
}

//------------------------------------------------------------------------------
// Основная функция
//------------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Можно переопределить OMP_NUM_THREADS снаружи
    // или здесь задать фиксированное число потоков
    // omp_set_num_threads(4);

    // Для "настоящей" случайности и чтобы разные ранги
    // не генерировали одинаковые числа
    srand(time(NULL) + rank * 1234);

    // Параметры генерации графа
    double density = 0.2;    // Плотность 20%
    int maxWeight  = 100;    // Макс. вес ребра

    // Набор размеров графа
    int testSizes[] = {10, 100, 200, 500, 1000, 5000, 10000, 100000, 1000000};
    int numTests = sizeof(testSizes) / sizeof(testSizes[0]);

    // Rank=0 будет печатать заголовки CSV
    if (rank == 0) {
        printf("n,parallel_time,linear_time\n");
    }

    // Цикл по наборам размеров
    for (int i = 0; i < numTests; i++)
    {
        int n = testSizes[i];

        int** graph = NULL;
        if (rank == 0) {
            // Генерируем случайный граф
            graph = gen_random_graph(n, density, maxWeight);
        }

        // Рассылка размера n
        MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // Создаём локальный граф (пусть будет для всех процессов)
        // чтобы не падать при обращении
        if (rank != 0) {
            graph = gen_empty_graph(n);
        }

        // Рассылаем сам граф
        for (int row = 0; row < n; row++) {
            MPI_Bcast(graph[row], n, MPI_INT, 0, MPI_COMM_WORLD);
        }

        // Запускаем параллельный алгоритм
        double timeParallel = parallel_prim_mst(graph, n, rank, world_size);

        // Последовательный — только на rank=0
        double timeLinear = 0.0;
        if (rank == 0) {
            timeLinear = linear_mst(graph, n);
        }

        // Вывод результатов
        if (rank == 0) {
            // CSV формат
            // n, timeParallel, timeLinear
            printf("%d,%.6f,%.6f\n", n, timeParallel, timeLinear);
        }

        free_graph(graph, n);

        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
