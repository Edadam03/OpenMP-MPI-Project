#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define SIZE 30 // Number of data points per city

// Function to read temperature data from CSV file
void read_csv(float city1[], float city2[], float city3[], const char *filename)
{
    FILE *file = fopen(filename, "r");
    

    char line[256];
    int i = 0;
    fgets(line, sizeof(line), file); // Skip header line

    while (fgets(line, sizeof(line), file) && i < SIZE)
    {
        sscanf(line, "%*[^,],%f,%f,%f", &city1[i], &city2[i], &city3[i]);
        i++;
    }

    fclose(file);
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (rank == 0) printf(" Total MPI Processes (Threads): %d", size);

    double start_time, end_time;
    start_time = MPI_Wtime();  // Start execution timer

    // Allow program to run with 1 to 3 processes
    if (rank == 0 && size < 3) {
        printf(" Warning: Running with %d process(es). Some cities may not be analyzed.", size);
    }

    float city1[SIZE], city2[SIZE], city3[SIZE];
    float local_data[SIZE];
    float results[3][SIZE]; // To store results if rank 0 processes multiple cities

    if (rank == 0) {
        printf("[Rank 0] Reading CSV file...");
        read_csv(city1, city2, city3, "Weather_Data_Analyzer_Dataset.csv");

        if (size > 1) {
            printf("[Rank 0] Sending city2 to Rank 1...");
            MPI_Send(city2, SIZE, MPI_FLOAT, 1, 0, MPI_COMM_WORLD);
        } else {
            printf("[Rank 0] No Rank 1, processing city2 locally.");
        }

        if (size > 2) {
            printf("[Rank 0] Sending city3 to Rank 2...");
            MPI_Send(city3, SIZE, MPI_FLOAT, 2, 0, MPI_COMM_WORLD);
        } else {
            printf("[Rank 0] No Rank 2, processing city3 locally.");
        }

        memcpy(results[0], city1, sizeof(float) * SIZE);
        if (size == 1) {
                memcpy(results[1], city2, sizeof(float) * SIZE);
                memcpy(results[2], city3, sizeof(float) * SIZE);
        } else if (size == 2) {
                memcpy(results[2], city3, sizeof(float) * SIZE);
        }
        memcpy(local_data, results[rank], sizeof(float) * SIZE);
        printf(" [Rank 0] Processing city1 (Kuala Lumpur)...\n");
    } else if (rank == 1 && size > 1) {
        printf("[Rank 1] Receiving city2 from Rank 0...");
        MPI_Recv(local_data, SIZE, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("[Rank 1] Processing city2 (La Paz)...");
    } else if (rank == 2 && size > 2) {
        printf("[Rank 2] Receiving city3 from Rank 0...");
        MPI_Recv(local_data, SIZE, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("[Rank 2] Processing city3 (Lisbon)...");
    }

    float min = local_data[0], max = local_data[0], sum = 0;
    for (int i = 0; i < SIZE; i++) {
            if (local_data[i] < min)
                min = local_data[i];
            if (local_data[i] > max)
                max = local_data[i];
            sum += local_data[i];
    }

    float avg = sum / SIZE;
    float sq_sum = 0;
    for (int i = 0; i < SIZE; i++) {
            float diff = local_data[i] - avg;
            sq_sum += diff * diff;
    }
    float stddev = sqrt(sq_sum / SIZE);

    int classification = 0;
    if (avg > 30.0) classification = 1; // Heatwave
    else if (avg < 13.0) classification = 2; // Cold Snap

    printf(" Computation done: Min=%.1f Max=%.1f Avg=%.2f StdDev=%.2f Class=%d\n",
           rank, min, max, avg, stddev, classification);

    float min_results[3], max_results[3], avg_results[3], stddev_results[3];
    int classification_results[3];

    printf("Participating in MPI_Gather...\n", rank);

    MPI_Gather(&min, 1, MPI_FLOAT, min_results, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Gather(&max, 1, MPI_FLOAT, max_results, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Gather(&avg, 1, MPI_FLOAT, avg_results, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Gather(&stddev, 1, MPI_FLOAT, stddev_results, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Gather(&classification, 1, MPI_INT, classification_results, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
            printf("\U0001F4E5 [Rank 0] Gathering complete. Final results:\n\n");

            const char *city_names[] = {
                "Kuala Lumpur, Malaysia",
                "La Paz, Bolivia",
                "Lisbon, Portugal"};

            for (int i = 0; i < 3; i++)
            {
                printf("City %d (%s):\n", i + 1, city_names[i]);
                printf("  Min = %.1fC\n", min_results[i]);
                printf("  Max = %.1fC\n", max_results[i]);
                printf("  Avg = %.2fC\n", avg_results[i]);
                printf("  StdDev = %.2f\n", stddev_results[i]);

                if (classification_results[i] == 1)
                    printf("  Heatwave classified (Avg > 30\u00b0C)\n");
                else if (classification_results[i] == 2)
                    printf("  Cold Snap classified (Avg < 13\u00b0C)\n");
                else
                    printf("  Normal temperature\n");

                printf("--------------------------------------------------\n");
            }

            end_time = MPI_Wtime();
            printf("Total Execution Time: %.6f seconds\n", end_time - start_time);
    }

    // Finalize MPI
    MPI_Finalize();
    return 0;
}
