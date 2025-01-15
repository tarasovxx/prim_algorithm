/******************************************************************************
 * Пример (учебный!) распараллеленного (MPI + OpenMP) алгоритма Прима с
 * распределённым хранением графа (adjacency list).
 *
 * Компиляция (пример):
 *   mpicc -fopenmp -O2 -o prim_mpi_omp_dist prim_mpi_omp_dist.c
 *
 * Запуск (пример):
 *   mpirun -np 4 ./prim_mpi_omp_dist 1000 20000
 *      где 4 - количество MPI-процессов
 *          1000 - общее число вершин
 *          20000 - общее число рёбер (примерно)
 *
 * Логика:
 *  - Каждый процесс хранит ТОЛЬКО часть вершин и исходящие из них рёбра
 *    (т. е. распределённый граф).
 *  - В каждом процессе OpenMP параллелит поиск минимального рёбра среди локальных.
 *  - С помощью MPI происходит согласование выбора "глобального" минимального рёбра.
 *
 * ПРИМЕЧАНИЕ:
 *  - Для больших масштабов и максимальной производительности
 *    на практике применяют более сложные структуры данных (приоритетные очереди,
 *    улучшенный Union-Find, асинхронные операции и т.д.).
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <mpi.h>
#include <omp.h>

#define INF 1000000000

// ------------------ Структура для списка рёбер -------------------
typedef struct {
    int src;    // от какой вершины
    int dst;    // к какой вершине
    int weight; // вес
} Edge;


// ----------------------------------------------------------------
// Генерация "случайных" рёбер локально в каждом процессе.
// Каждый процесс генерирует рёбра только для "своих" вершин
// (блок вершин от localStart до localEnd-1).
// Аргумент totalEdges - это общее желаемое кол-во рёбер (примерно),
// распределим его поровну между процессами.
// ----------------------------------------------------------------
Edge* generateLocalEdges(int rank, int size, int n, int totalEdges,
                         int localStart, int localEnd,
                         int *localEdgeCountOut)
{
    // Простейший способ: пусть каждый процесс сгенерирует примерно (totalEdges / size) рёбер
    int localCount = totalEdges / size;
    // (можно учесть остаток, но сейчас для краткости опустим)

    Edge *localEdges = (Edge*)malloc(localCount * sizeof(Edge));

    srand((unsigned int)(time(NULL) + rank * 12345));

    for (int i = 0; i < localCount; i++) {
        // src в диапазоне [localStart..localEnd-1],
        // т.к. рёбра принадлежат «локальным» вершинам
        int src = localStart + (rand() % (localEnd - localStart));
        int dst = rand() % n; // во весь диапазон вершин
        int w = 1 + (rand() % 1000); // вес 1..1000

        // На всякий случай следим, чтобы src != dst
        if (dst == src) {
            dst = (dst + 1) % n;
        }

        // Заполним структуру
        localEdges[i].src = src;
        localEdges[i].dst = dst;
        localEdges[i].weight = w;
    }

    *localEdgeCountOut = localCount;
    return localEdges;
}

// ----------------------------------------------------------------
// (Вспомогательная) Функция для распределения вершин на процессы по схеме block.
// Например, если n=100, size=4, то:
//   rank=0 -> [0..24]
//   rank=1 -> [25..49]
//   rank=2 -> [50..74]
//   rank=3 -> [75..99]
// ----------------------------------------------------------------
void getLocalVertexRange(int rank, int size, int n, int *start, int *end)
{
    int verticesPerProc = n / size;
    int remainder = n % size;
    // Распределим "остаток" по первым процессам
    if (rank < remainder) {
        *start = rank * (verticesPerProc + 1);
        *end   = *start + (verticesPerProc + 1);
    } else {
        *start = remainder * (verticesPerProc + 1)
                 + (rank - remainder) * verticesPerProc;
        *end   = *start + verticesPerProc;
    }
    if (*end > n) *end = n; // на всякий случай
}

// ----------------------------------------------------------------
// Глобальный алгоритм Прима (в демонстрационной параллельной форме):
//   - все процессы совместно поддерживают массив visited[n]
//   - процесс, который "выигрывает" выбор минимального ребра,
//     добавляет вершину в остов
//   - все синхронизируются (MPI_Bcast visited, и т.д.)
// Продолжаем, пока не выберем (n-1) рёбер (или все вершины не посещены).
// ----------------------------------------------------------------
int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 3) {
        if (rank == 0) {
            fprintf(stderr,
               "Usage: mpirun -np <p> %s <num_vertices> <num_edges>\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    // Общее число вершин и рёбер
    int n = atoi(argv[1]);
    int totalEdges = atoi(argv[2]);

    // Распределение вершин по процессам
    int localStart, localEnd;
    getLocalVertexRange(rank, size, n, &localStart, &localEnd);
    int localVertexCount = localEnd - localStart;

    // Генерация "локальной" части рёбер
    int localEdgeCount;
    Edge *localEdges = generateLocalEdges(rank, size, n, totalEdges,
                                          localStart, localEnd,
                                          &localEdgeCount);

    // Для оценки времени
    double gen_time_start = MPI_Wtime();

    // Собственно, генерация уже сделана, отсечём время
    double gen_time_end = MPI_Wtime();

    // Общий массив visited (на всех процессах одинаковый)
    // Для простоты хранения реплицируем: каждый процесс держит visited[n].
    // Это не критично по памяти, если n не сверхбольшое.
    bool *visited = (bool *)malloc(n * sizeof(bool));
    for (int i = 0; i < n; i++) visited[i] = false;

    // Начинаем измерять время Прима
    double start_time = MPI_Wtime();

    // Выбираем вершину 0 в качестве стартовой (можно другую)
    // Процесс, который владеет вершиной 0, активирует её
    if (rank == 0 && localStart <= 0 && localEnd > 0) {
        visited[0] = true;
    }

    // Синхронизируем visited
    MPI_Bcast(visited, n, MPI_C_BOOL, /*root=*/0, MPI_COMM_WORLD);

    long long mstCost = 0; // суммарная стоимость MST
    int edgesChosen = 0;   // сколько рёбер уже добавили (до n-1)

    // Будем (n-1) раз искать новое ребро (или пока не посещены все вершины)
    while (edgesChosen < n - 1) {

        // Каждый процесс ищет локальное минимальное ребро (minEdge),
        // ведущее из посещённой вершины в непосещённую
        int localMinWeight = INF;
        int localMinSrc = -1;
        int localMinDst = -1;

        // ПАРАЛЛЕЛИЗУЕМ перебор локальных рёбер OpenMP (если localEdgeCount велик)
        #pragma omp parallel
        {
            // Локальные переменные для потока
            int thrMinW = INF;
            int thrMinS = -1;
            int thrMinD = -1;

            #pragma omp for nowait
            for (int i = 0; i < localEdgeCount; i++) {
                Edge e = localEdges[i];
                // Интересуют только рёбра, где src уже посещён,
                // а dst ещё не посещён (или наоборот, если хотим неориентированный вариант)
                // Предположим, что ребро (src->dst) достаточно,
                // если visited[src] = true, visited[dst] = false
                // (Можно также проверить e.dst->e.src, если хранение рёбер неориентированное)
                if (visited[e.src] && !visited[e.dst]) {
                    if (e.weight < thrMinW) {
                        thrMinW = e.weight;
                        thrMinS = e.src;
                        thrMinD = e.dst;
                    }
                }
                else if (visited[e.dst] && !visited[e.src]) {
                    // Если ребро фактически неориентированное,
                    // то можно рассматривать и обратный вариант
                    if (e.weight < thrMinW) {
                        thrMinW = e.weight;
                        thrMinS = e.dst;
                        thrMinD = e.src;
                    }
                }
            }

            // Синхронизируем результаты внутри процесса
            #pragma omp critical
            {
                if (thrMinW < localMinWeight) {
                    localMinWeight = thrMinW;
                    localMinSrc    = thrMinS;
                    localMinDst    = thrMinD;
                }
            }
        } // end omp parallel

        // Теперь у процесса есть localMinEdge
        // Найдём глобально "лучшее" ребро через MPI
        struct {
            int weight;
            int src;
            int dst;
        } localMin, globalMin;

        localMin.weight = localMinWeight;
        localMin.src    = localMinSrc;
        localMin.dst    = localMinDst;

        // MPI_Allreduce: нужно взять минимум по полю weight,
        // но при этом сохранить src/dst соответствующего минимума
        // Можно сделать пользовательский MPI_Op, но для примера
        // часто используют трюк: сначала сравниваем вес, затем
        // берём (src, dst).
        // Сделаем это в два шага (для наглядности).

        // 1) Собираем globalMinWeight
        int sendBuf[2], recvBuf[2];
        sendBuf[0] = localMin.weight;
        sendBuf[1] = rank;  // чтобы понять, кто "выиграл"
        MPI_Allreduce(sendBuf, recvBuf, 1, MPI_2INT, MPI_MINLOC, MPI_COMM_WORLD);
        // Теперь recvBuf[0] = globalMinWeight, recvBuf[1] = rankWinner

        int globalMinWeight  = recvBuf[0];
        int globalMinProcess = recvBuf[1];

        // 2) Процесс-победитель рассылает (src, dst)
        //    (либо через MPI_Bcast, либо через MPI_Gather/MPI_Scatter)
        int bestSrc, bestDst;
        if (rank == globalMinProcess) {
            bestSrc = localMin.src;
            bestDst = localMin.dst;
        }
        // Рассылаем
        MPI_Bcast(&bestSrc, 1, MPI_INT, globalMinProcess, MPI_COMM_WORLD);
        MPI_Bcast(&bestDst, 1, MPI_INT, globalMinProcess, MPI_COMM_WORLD);

        // Если globalMinWeight == INF, значит ребра больше нет
        if (globalMinWeight == INF || bestSrc < 0 || bestDst < 0) {
            // MST уже построено (или граф был несвязным)
            break;
        }

        // Добавляем ребро к MST (считаем его вес) - делаем это 1 раз (например, на ранк 0)
        // либо можем "добавлять" везде, но обычно итоговую стоимость собираем в одном месте.
        if (rank == 0) {
            mstCost += globalMinWeight;
        }

        // Помечаем вершину bestDst как посещённую (или bestSrc - смотря что добавляем)
        visited[bestDst] = true;

        edgesChosen++;

        // Распространяем updated visited[] по всем
        // (Можно было бы делать инкрементальные передачи,
        //  но для простоты Bcast всего массива bool)
        MPI_Bcast(visited, n, MPI_C_BOOL, 0, MPI_COMM_WORLD);

        // Продолжаем, пока не выберем (n-1) рёбер
    }

    // Замер времени окончания
    double end_time = MPI_Wtime();

    // Выводим результаты (только на rank=0, чтобы не дублировать)
    if (rank == 0) {
        printf("=============================================\n");
        printf("Number of vertices: %d\n", n);
        printf("Number of edges (total): %d\n", totalEdges);
        printf("Time for graph generation: %.6f s\n", gen_time_end - gen_time_start);
        printf("Time for Prim's algorithm: %.6f s\n", end_time - start_time);
        printf("MST total cost = %lld\n", mstCost);
        printf("Edges chosen = %d (should be ~ %d)\n", edgesChosen, n - 1);
        printf("=============================================\n");
    }

    // Освобождаем ресурсы
    free(localEdges);
    free(visited);

    MPI_Finalize();
    return 0;
}
