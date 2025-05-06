#include <iostream>
#include <vector>
#include <stack>
#include <omp.h>

using namespace std;

const int MAX = 100000;
vector<int> graph[MAX];
bool visited_seq[MAX];
bool visited_par[MAX];
omp_lock_t lock[MAX];

// Sequential DFS
void dfsSequential(int start_node)
{
    stack<int> s;
    s.push(start_node);

    while (!s.empty())
    {
        int curr_node = s.top();
        s.pop();

        if (!visited_seq[curr_node])
        {
            visited_seq[curr_node] = true;
            cout << curr_node << " ";

            // Add neighbors in reverse order for stack-like behavior
            for (int i = graph[curr_node].size() - 1; i >= 0; i--)
            {
                int adj_node = graph[curr_node][i];
                if (!visited_seq[adj_node])
                    s.push(adj_node);
            }
        }
    }
}

// Parallel DFS
void dfsParallel(int start_node)
{
    stack<int> s;
    s.push(start_node);

    while (!s.empty())
    {
        int curr_node;

#pragma omp critical
        {
            if (!s.empty())
            {
                curr_node = s.top();
                s.pop();
            }
            else
                curr_node = -1;
        }

        if (curr_node == -1)
            continue;

        omp_set_lock(&lock[curr_node]);
        if (!visited_par[curr_node])
        {
            visited_par[curr_node] = true;
            cout << curr_node << " ";
        }
        omp_unset_lock(&lock[curr_node]);

#pragma omp parallel for shared(s)
        for (int i = 0; i < graph[curr_node].size(); i++)
        {
            int adj_node = graph[curr_node][i];

            omp_set_lock(&lock[adj_node]);
            if (!visited_par[adj_node])
            {
#pragma omp critical
                {
                    s.push(adj_node);
                }
            }
            omp_unset_lock(&lock[adj_node]);
        }
    }
}

int main()
{
    int n, m, start_node;

    cout << "Enter number of nodes, edges, and the starting node: ";
    cin >> n >> m >> start_node;

    cout << "Enter pairs of connected edges (u v):\n";
    for (int i = 0; i < m; i++)
    {
        int u, v;
        cin >> u >> v;
        graph[u].push_back(v);
        graph[v].push_back(u); // undirected graph
    }

#pragma omp parallel for
    for (int i = 0; i < n; i++)
    {
        visited_seq[i] = false;
        visited_par[i] = false;
        omp_init_lock(&lock[i]);
    }

    // Sequential DFS
    cout << "\nSequential DFS Traversal:\n";
    double start_seq = omp_get_wtime();
    dfsSequential(start_node);
    double end_seq = omp_get_wtime();
    cout << "\nSequential Execution Time: " << (end_seq - start_seq) << " seconds\n";

    // Parallel DFS
    cout << "\nParallel DFS Traversal:\n";
    double start_par = omp_get_wtime();
    dfsParallel(start_node);
    double end_par = omp_get_wtime();
    cout << "\nParallel Execution Time: " << (end_par - start_par) << " seconds\n";

    // Cleanup
    for (int i = 0; i < n; i++)
    {
        omp_destroy_lock(&lock[i]);
    }

    return 0;
}

