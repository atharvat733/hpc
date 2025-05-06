#include <iostream>
#include <omp.h>
#include <vector>
using namespace std;

// Sequential Bubble Sort
void bubbleSortSequential(vector<int> &arr, int n)
{
    for (int i = 0; i < n - 1; i++)
    {
        for (int j = 0; j < n - i - 1; j++)
        {
            if (arr[j] > arr[j + 1])
                swap(arr[j], arr[j + 1]);
        }
    }
}

// Parallel Bubble Sort using OpenMP
void bubbleSortParallel(vector<int> &arr, int n)
{
    for (int i = 0; i < n - 1; i++)
    {
        // Even indexed phase
        if (i % 2 == 0)
        {
#pragma omp parallel for
            for (int j = 0; j < n - 1; j += 2)
            {
                if (arr[j] > arr[j + 1])
                    swap(arr[j], arr[j + 1]);
            }
        }
        // Odd indexed phase
        else
        {
#pragma omp parallel for
            for (int j = 1; j < n - 1; j += 2)
            {
                if (arr[j] > arr[j + 1])
                    swap(arr[j], arr[j + 1]);
            }
        }
    }
}

int main()
{
    int n;
    cout << "Enter number of elements: ";
    cin >> n;

    vector<int> arr(n), arrSeq(n);

    cout << "Enter the elements:\n";
    for (int i = 0; i < n; i++)
    {
        cin >> arr[i];
    }

    arrSeq = arr; // Copy input for sequential sort

    // Sequential timing
    double start = omp_get_wtime();
    bubbleSortSequential(arrSeq, n);
    double end = omp_get_wtime();
    double seqTime = end - start;

    // Parallel timing
    start = omp_get_wtime();
    bubbleSortParallel(arr, n);
    end = omp_get_wtime();
    double parTime = end - start;

    // Output sorted array
    cout << "\nSorted array:\n";
    for (int i = 0; i < n; i++)
        cout << arr[i] << " ";
    cout << "\n";

    cout << "\nSequential Time: " << seqTime << " seconds";
    cout << "\nParallel Time  : " << parTime << " seconds";

    return 0;
}

