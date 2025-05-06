#include <iostream>
#include <omp.h>
#include <vector>
using namespace std;

// Merge two sorted subarrays
void merge(vector<int> &arr, int l, int m, int r)
{
    int n1 = m - l + 1;
    int n2 = r - m;

    vector<int> L(n1), R(n2);
    for (int i = 0; i < n1; i++)
        L[i] = arr[l + i];
    for (int j = 0; j < n2; j++)
        R[j] = arr[m + 1 + j];

    int i = 0, j = 0, k = l;

    while (i < n1 && j < n2)
        arr[k++] = (L[i] <= R[j]) ? L[i++] : R[j++];
    while (i < n1)
        arr[k++] = L[i++];
    while (j < n2)
        arr[k++] = R[j++];
}

// Sequential Merge Sort
void mergeSortSequential(vector<int> &arr, int l, int r)
{
    if (l < r)
    {
        int m = l + (r - l) / 2;
        mergeSortSequential(arr, l, m);
        mergeSortSequential(arr, m + 1, r);
        merge(arr, l, m, r);
    }
}

// Parallel Merge Sort using OpenMP
void mergeSortParallel(vector<int> &arr, int l, int r, int depth = 0)
{
    if (l < r)
    {
        int m = l + (r - l) / 2;

        if (depth < 4)
        { // Limit depth to avoid too many threads
#pragma omp parallel sections
            {
#pragma omp section
                mergeSortParallel(arr, l, m, depth + 1);

#pragma omp section
                mergeSortParallel(arr, m + 1, r, depth + 1);
            }
        }
        else
        {
            mergeSortSequential(arr, l, m);
            mergeSortSequential(arr, m + 1, r);
        }

        merge(arr, l, m, r);
    }
}

int main()
{
    int n;
    cout << "Enter number of elements: ";
    cin >> n;

    vector<int> arr(n), arrSeq(n);

    // User input for array elements
    cout << "Enter the elements:\n";
    for (int i = 0; i < n; i++)
    {
        cin >> arr[i];
    }
    arrSeq = arr; // Copy input for sequential sort

    // Sequential sort timing
    double start = omp_get_wtime();
    mergeSortSequential(arrSeq, 0, n - 1);
    double end = omp_get_wtime();
    double seqTime = end - start;

    // Parallel sort timing
    start = omp_get_wtime();
    mergeSortParallel(arr, 0, n - 1);
    end = omp_get_wtime();
    double parTime = end - start;

    // Output sorted array
    cout << "\nSorted array:\n";
    for (int i = 0; i < n; i++)
        cout << arr[i] << " ";
    cout << "\n";

   // Display metrics

    cout << "\nSequential Time: " << seqTime << " seconds";
    cout << "\nParallel Time  : " << parTime << " seconds";
 
    return 0;
}

