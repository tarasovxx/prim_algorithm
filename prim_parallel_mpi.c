#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <stdbool.h>
#include <omp.h>
#include <mpi.h>

#define INF INT_MAX

int** gen_empty_graph(int size)
{
    int** graph = (int**)calloc(size, sizeof(int*));
    for (int i = 0; i < size; ++i)
        graph[i] = (int*)calloc(size, sizeof(int));
    return graph;
}

int** gen_graph(int size)
{
    int** graph = gen_empty_graph(size);
    for (int i = 0; i < size; ++i)
    {
        for (int j = i + 1; j < size; ++j)
        {
            graph[i][j] = size - j;
            graph[j][i] = size - j;
        }

    }
    return graph;
}

// Function to find the vertex with the minimum key value
int find_min_key(int* key, bool* mst_set, int vertices) 
{
    int min = INF, min_index = -1;
    for (int v = 0; v < vertices; v++) 
    {
        if (!mst_set[v] && key[v] < min) 
        {
            min = key[v];
            min_index = v;
        }
    }
    return min_index;
}

double parallel_prim_mst(int** graph, int vertices, int rank, int size) 
{
    double start = MPI_Wtime();
    int* parent = (int*)malloc(vertices * sizeof(int)); // Stores MST
    int* key = (int*)malloc(vertices * sizeof(int));    // Minimum weight edge
    bool* mst_set = (bool*)malloc(vertices * sizeof(bool));

    // Initialize keys as infinite and mst_set[] as false
    for (int i = 0; i < vertices; i++)
    {
        key[i] = INF;
        mst_set[i] = false;
    }

    printf("%d setting env\n", rank);

    key[0] = 0;     // Start with the first vertex
    parent[0] = -1; // First node is always the root of MST

    for (int count = 0; count < vertices - 1; count++)
    {
        int min_index;
        if (rank == 0)
            min_index = find_min_key(key, mst_set, vertices);

        // Broadcast the chosen vertex to all processes
        MPI_Bcast(&min_index, 1, MPI_INT, 0, MPI_COMM_WORLD);
        mst_set[min_index] = true;

        // Update key and parent arrays using OpenMP
        #pragma omp parallel for
        for (int v = 0; v < vertices; v++) 
        {
            if (graph[min_index][v] && !mst_set[v] && graph[min_index][v] < key[v]) 
            {
                key[v] = graph[min_index][v];
                parent[v] = min_index;
            }
        }
    }

    // if (!rank) 
    // {
    //     printf("Edge \tWeight\n");
    //     for (int i = 1; i < vertices; i++)
    //         printf("%d - %d \t%d\n", parent[i], i, graph[i][parent[i]]);
    // }
    double finish = MPI_Wtime();

    free(parent);
    free(key);
    free(mst_set);
    return finish - start;
}

double linear_mst(int** graph, int vertices)
{
    double start = MPI_Wtime();    
    int* parent = (int*)malloc(vertices * sizeof(int)); // Stores MST
    int* key = (int*)malloc(vertices * sizeof(int));    // Minimum weight edge
    bool* mst_set = (bool*)malloc(vertices * sizeof(bool));

    // Initialize keys as infinite and mst_set[] as false
    for (int i = 0; i < vertices; i++)
    {
        key[i] = INF;
        mst_set[i] = false;
    }
    key[0] = 0;     // Start with the first vertex
    parent[0] = -1; // First node is always the root of MST

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
    
    printf("Edge \tWeight\n");
    for (int i = 1; i < vertices; i++)
        printf("%d - %d \t%d\n", parent[i], i, graph[i][parent[i]]);
    
    free(parent);
    free(key);
    free(mst_set);
    return finish - start;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int vertices = 1000;
    int** graph  = gen_empty_graph(vertices);
    if (rank == 0) graph = gen_graph(vertices);

    printf("%d before bcast\n", rank);

    // Broadcast the graph to all processes
    for (int i = 0; i < vertices; i++) {
        MPI_Bcast(graph[i], vertices, MPI_INT, 0, MPI_COMM_WORLD);
    }

    double par_t = parallel_prim_mst(graph, vertices, rank, size);
    double lin_t = linear_mst(graph, vertices);
    if (!rank)
    {
        printf("parallel %lf\n", par_t);
        printf("linear%lf\n", linear_mst(graph, vertices));
    }

    for (int i = 0; i < vertices; i++) {
        free(graph[i]);
    }
    free(graph);

    MPI_Finalize();
    return 0;
}