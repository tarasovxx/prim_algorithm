#include <stdio.h>
#include <limits.h>
#include <omp.h>
#include <time.h>
#include <stdlib.h>

#define V 10000

int t_cost = 0;

int minKey(int key[], int visited[])
{
    int min = INT_MAX, index, i;

#pragma omp parallel
    {
        int local_min = INT_MAX;
        int local_index = -1;

#pragma omp for
        for (i = 0; i < V; i++)
        {
            if (visited[i] == 0 && key[i] < local_min)
            {
                local_min = key[i];
                local_index = i;
            }
        }

#pragma omp critical
        {
            if (local_min < min)
            {
                min = local_min;
                index = local_index;
            }
        }
    }
    return index;
}

void printMST(int from[], int n, int **graph)
{
    printf("\n\n Edge   Weight\n\n");
    for (int i = 1; i < V; i++)
    {
        t_cost += graph[i][from[i]];
        printf("%d <---->  %d  =  %d \n", from[i], i, graph[i][from[i]]);
    }

    printf("\n\tMinimum cost = %d\n", t_cost);
}

void primMST(int **graph)
{
    int from[V];
    int key[V];
    int visited[V];

    for (int i = 0; i < V; i++)
    {
        key[i] = INT_MAX;
        visited[i] = 0;
    }

    key[0] = 0;
    from[0] = -1;

    for (int count = 0; count < V - 1; count++)
    {
        int u = minKey(key, visited);
        visited[u] = 1;

#pragma omp parallel for schedule(dynamic, 10)
        for (int v = 0; v < V; v++)
        {
            if (graph[u][v] && visited[v] == 0 && graph[u][v] < key[v])
            {
                from[v] = u;
                key[v] = graph[u][v];
            }
        }
    }
    // printMST(from, V, graph);
}

int main()
{
    printf("\n\n____Implementing parallel PRIM's using OMP_____\n\n");

    int **graph = (int **)malloc(V * sizeof(int *));
    for (int x = 0; x < V; x++)
        graph[x] = (int *)malloc(V * sizeof(int));

    srand(time(NULL));

    for (int i = 0; i < V; i++)
        for (int j = 0; j < V; j++)
            graph[i][j] = rand() % 10;

    for (int i = 0; i < V; i++)
        graph[i][i] = 0;

    for (int i = 0; i < V; i++)
        for (int j = 0; j < V; j++)
            graph[j][i] = graph[i][j];

    printf("Adjacency matrix:\n");
    for (int i = 0; i < V; i++)
    {
        for (int j = 0; j < V; j++)
        {
            printf("%d ", graph[i][j]);
        }
        printf("\n");
    }

    for (int threads = 1; threads <= 16; threads++)
    {
        printf("\nRunning with %d threads:\n", threads);
        omp_set_dynamic(0);
        omp_set_num_threads(threads);

        t_cost = 0;
        double start = omp_get_wtime();
        primMST(graph);
        double end = omp_get_wtime();

        printf("\tTime = %f seconds\n", end - start);
    }

    for (int i = 0; i < V; i++)
        free(graph[i]);
    free(graph);

    return 0;
}
