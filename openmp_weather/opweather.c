#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <omp.h>

#define SIZE 30

// Function to read CSV data
void read_csv(float city1[], float city2[], float city3[], const char *filename) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        printf("\u274c Error: Cannot open %s\n", filename);
        exit(1);
    }
    char line[256];
    int i = 0;
    fgets(line, sizeof(line), file); // Skip header
    while (fgets(line, sizeof(line), file) && i < SIZE) {
        sscanf(line, "%*[^,],%f,%f,%f", &city1[i], &city2[i], &city3[i]);
        i++;
    }
    fclose(file);
}

int main() {
    float city1[SIZE], city2[SIZE], city3[SIZE];
    read_csv(city1, city2, city3, "Weather_Data_Analyzer_Dataset.csv");

    float min_vals[3] = {FLT_MAX, FLT_MAX, FLT_MAX};
    float max_vals[3] = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
    float sum_vals[3] = {0}, sq_diff[3] = {0};
    float avg[3], stddev[3];
    float *cities[3] = {city1, city2, city3};

    // 🔢 Show number of threads
    #pragma omp parallel
    {
        #pragma omp master
        {
            int num_threads = omp_get_num_threads();
            printf("🔢 Number of OpenMP threads: %d\n", num_threads);
        }
    }

    // Start timing
    double start_time = omp_get_wtime();

    // Parallel computation of min, max, and sum using reduction
    #pragma omp parallel for
    for (int city = 0; city < 3; city++) {
        float local_min = FLT_MAX;
        float local_max = -FLT_MAX;
        float local_sum = 0;
        for (int i = 0; i < SIZE; i++) {
            float val = cities[city][i];
            if (val < local_min) local_min = val;
            if (val > local_max) local_max = val;
            local_sum += val;
        }
        #pragma omp critical
        {
            if (local_min < min_vals[city]) min_vals[city] = local_min;
            if (local_max > max_vals[city]) max_vals[city] = local_max;
            sum_vals[city] += local_sum;
        }
    }

    // Compute averages
    for (int i = 0; i < 3; i++) {
        avg[i] = sum_vals[i] / SIZE;
    }

    // Parallel computation of squared differences
    #pragma omp parallel for
    for (int city = 0; city < 3; city++) {
        float local_sq = 0;
        for (int i = 0; i < SIZE; i++) {
            float diff = cities[city][i] - avg[city];
            local_sq += diff * diff;
        }
        #pragma omp critical
        {
            sq_diff[city] += local_sq;
        }
    }

    // Compute standard deviation
    for (int i = 0; i < 3; i++) {
        stddev[i] = sqrt(sq_diff[i] / SIZE);
    }

    // End timing
    double end_time = omp_get_wtime();

    // Print results
    const char* city_names[] = {
        "Kuala Lumpur, Malaysia",
        "La Paz, Bolivia",
        "Lisbon, Portugal"
    };

    for (int i = 0; i < 3; i++) {
        printf("City %d (%s):\n", i + 1, city_names[i]);
        printf("  Min = %.1f C\n", min_vals[i]);
        printf("  Max = %.1f C\n", max_vals[i]);
        printf("  Avg = %.2f C\n", avg[i]);
        printf("  StdDev = %.2f\n", stddev[i]);

        if (avg[i] > 30.0)
            printf("  Heatwave classified based on average > 30 C\n");
        else if (avg[i] < 13.0)
            printf("  Cold Snap classified based on average < 13 C\n");
        else
            printf("  Normal temperature based on average\n");

        printf("--------------------------------------------------\n");
    }

    printf("\n\u23F1\ufe0f Execution Time (OpenMP): %.6f seconds\n", end_time - start_time);

    return 0;
}