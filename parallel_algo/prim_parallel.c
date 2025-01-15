#include <limits.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

typedef struct Edge {
    int start, end, weight;
} Edge;

void createMPIEdgeType(MPI_Datatype *MPI_EDGE) {

    int blocklengths[3] = {1, 1, 1};
    MPI_Datatype types[3] = {MPI_INT, MPI_INT, MPI_INT};
    MPI_Aint offsets[3];

    offsets[0] = offsetof(Edge, weight);
    offsets[1] = offsetof(Edge, start);
    offsets[2] = offsetof(Edge, end);

    MPI_Type_create_struct(3, blocklengths, offsets, types, MPI_EDGE);
    MPI_Type_commit(MPI_EDGE);
}

void minWeightOp(void *in, void *inout, int *len, MPI_Datatype *dptr) {
    Edge *inEdges = (Edge *)in;
    Edge *inoutEdge = (Edge *)inout;

    if (inEdges->weight < inoutEdge->weight ||
        (inEdges->weight == inoutEdge->weight && inEdges->start < inoutEdge->start)) {
        *inoutEdge = *inEdges;
    }
}



int findEdgeWithMinKey(bool mstSet[], Edge edges[], int edgeCount) {
    int min = INT_MAX, min_index = -1;

    #pragma omp parallel for
    for (int i = 0; i < edgeCount; i++) {
        if ((mstSet[edges[i].start] || mstSet[edges[i].end]) && edges[i].weight < min) {
            if (!(mstSet[edges[i].start] && mstSet[edges[i].end])) {
                #pragma omp critical
                {
                    if (edges[i].weight < min) {
                        min = edges[i].weight;
                        min_index = i;
                    }
                }
            }
        }
    }

    return min_index;
}


Edge* parallelPrimMST(Edge edges[], int edgeCount, int rows, int rank, int size) {
    MPI_Datatype MPI_EDGE;
    createMPIEdgeType(&MPI_EDGE);
    MPI_Op minWeightMPIOp;
    MPI_Op_create((MPI_User_function *)minWeightOp, 1, &minWeightMPIOp);
    Edge* mst = NULL;
    if (rank == 0) {
        mst = (Edge*)malloc((rows - 1) * sizeof(Edge));
        if (mst == NULL) {
            fprintf(stderr, "Errore nell'allocazione di memoria per MST\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    bool mstSet[rows];
    int *sendcounts = (int *)malloc(size * sizeof(int));
    int *displs = (int *)malloc(size * sizeof(int));
    
    for (int i = 0; i < rows; i++) {
        mstSet[i] = false;
    }
    mstSet[0] = true;

    int localEdgeCount = edgeCount / size;
    int extraSize = edgeCount % size;

    if(size == 1){
        sendcounts[0] = localEdgeCount;
        displs[0] = 0;
    }
    else{
        for (int i = 0; i < size - 1; i++){
            sendcounts[i] = localEdgeCount;

            if(i == 0){
                displs[i] = 0;
            }
            else{
                displs[i] = localEdgeCount + displs[i-1];
            }
        }

        sendcounts[size-1] = localEdgeCount + extraSize;
        displs[size-1] = localEdgeCount + displs[size-2];  
    }

    Edge *localEdges = (Edge *)malloc(sendcounts[rank] * sizeof(Edge));
    MPI_Bcast(sendcounts, size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(displs, size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&mstSet, rows, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    Edge globalMinEdge;
    MPI_Scatterv(edges, sendcounts, displs, MPI_EDGE, localEdges, sendcounts[rank], MPI_EDGE, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    for (int count = 0, i = 0; count < rows-1; count++) {
        

        int localMinIndex = findEdgeWithMinKey(mstSet, localEdges, sendcounts[rank]);
        MPI_Barrier(MPI_COMM_WORLD);


        MPI_Barrier(MPI_COMM_WORLD);
        if(localMinIndex!= -1){
            localMinIndex;
            MPI_Allreduce(&localEdges[localMinIndex], &globalMinEdge, 1, MPI_EDGE, minWeightMPIOp, MPI_COMM_WORLD);
        }
        else{
            Edge noEdgeFound;
            noEdgeFound.weight = INT_MAX;
            noEdgeFound.end = 0;
            noEdgeFound.start = 1; 
            MPI_Allreduce(&noEdgeFound,&globalMinEdge, 1, MPI_EDGE, minWeightMPIOp, MPI_COMM_WORLD);
        }
       
        

        if (rank == 0) {
            mstSet[globalMinEdge.start] = true;
            mstSet[globalMinEdge.end] = true;
            mst[i++] = globalMinEdge;

        }
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Bcast(&mstSet, rows, MPI_C_BOOL, 0, MPI_COMM_WORLD);

    }


    free(sendcounts);
    free(displs);
    free(localEdges);
    MPI_Op_free(&minWeightMPIOp);
    return mst;
}



int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    double start_time;
    double end_time;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    char *inputFilePath = argv[1];

    FILE *file = fopen(inputFilePath, "r");
    if (file == NULL) {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int rows= atoi(argv[2]);

    int total_elements = rows * rows;
    int *matrix = (int *)malloc(total_elements * sizeof(int));

    for (int i = 0; i < total_elements; i++) {
        if(fscanf(file, "%d", &matrix[i])!=1){
            return 1;
        };
    }

    fclose(file);


    Edge *edges = (Edge *)malloc((rows * (rows - 1) / 2) * sizeof(Edge));
    int edgeCount = 0;

    for (int i = 0; i < rows; i++) {
        for (int j = i + 1; j < rows; j++) {
            if (matrix[i * rows + j] != 0) {
                edges[edgeCount].start = i;
                edges[edgeCount].end = j;
                edges[edgeCount].weight = matrix[i * rows + j];
                edgeCount++;
            }
        }
    }
    free(matrix);

    MPI_Barrier(MPI_COMM_WORLD);
    double times[10];
    int averageTime;
    double elapsed;
    for(int i=0;i<10;i++){
    if (rank == 0) {
        start_time = MPI_Wtime();
    }

    Edge* mst= parallelPrimMST(edges, edgeCount, rows, rank, size);

    
    if (rank == 0) {
        end_time = MPI_Wtime();
        
    }
    elapsed= end_time - start_time;
    
    if(rank==0){
        int weight = 0;
        for (int i = 0; i < rows - 1; i++) {
            weight += mst[i].weight;
        }
        printf("\nTotal Weight:%d\n", weight);
    }
    times[i]=elapsed;
    }
    double total=0;
    for(int i=0;i<10;i++){
        total+=times[i];
    }
    if(rank==0){
            printf("MPI Time : %f", (double)(total/10));
    }
    free(edges);

    MPI_Finalize();

    return 0;
}