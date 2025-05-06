#include <iostream>
#include <queue>
#include <omp.h> // Required for OpenMP timing and directives
using namespace std;

class Node
{
public:
    Node *left, *right;
    char data;
};

class BreadthFS
{
public:
    Node *insert(Node *root, char data);
    void bfs(Node *root);           // Parallel BFS
    void bfsSequential(Node *root); // Sequential BFS
};

Node *BreadthFS::insert(Node *root, char data)
{
    if (!root)
    {
        root = new Node;
        root->left = nullptr;
        root->right = nullptr;
        root->data = data;
        return root;
    }

    queue<Node *> q;
    q.push(root);

    while (!q.empty())
    {
        Node *current = q.front();
        q.pop();

        if (!current->left)
        {
            current->left = new Node;
            current->left->left = nullptr;
            current->left->right = nullptr;
            current->left->data = data;
            return root;
        }
        else
        {
            q.push(current->left);
        }

        if (!current->right)
        {
            current->right = new Node;
            current->right->left = nullptr;
            current->right->right = nullptr;
            current->right->data = data;
            return root;
        }
        else
        {
            q.push(current->right);
        }
    }
    return root;
}

// Sequential BFS
void BreadthFS::bfsSequential(Node *root)
{
    if (!root)
        return;

    queue<Node *> q;
    q.push(root);

    while (!q.empty())
    {
        Node *current = q.front();
        q.pop();
        cout << current->data << "\t";

        if (current->left)
            q.push(current->left);
        if (current->right)
            q.push(current->right);
    }
}

// Parallel BFS
void BreadthFS::bfs(Node *root)
{
    if (!root)
        return;

    queue<Node *> q;
    q.push(root);

    while (!q.empty())
    {
        int level_size = q.size();

#pragma omp parallel for
        for (int i = 0; i < level_size; i++)
        {
            Node *current = nullptr;

#pragma omp critical
            {
                if (!q.empty())
                {
                    current = q.front();
                    q.pop();
                    cout << current->data << "\t";
                }
            }

#pragma omp critical
            {
                if (current)
                {
                    if (current->left)
                        q.push(current->left);
                    if (current->right)
                        q.push(current->right);
                }
            }
        }
    }
}

int main()
{
    BreadthFS bfsObj;
    Node *root = nullptr;
    char data;
    char choice;

    do
    {
        cout << "Enter data: ";
        cin >> data;
        root = bfsObj.insert(root, data);
        cout << "Insert another node? (y/n): ";
        cin >> choice;
    } while (choice == 'y' || choice == 'Y');

    cout << "\nSequential BFS Traversal:\n";
    double start_seq = omp_get_wtime();
    bfsObj.bfsSequential(root);
    double end_seq = omp_get_wtime();
    cout << "\nSequential Execution Time: " << (end_seq - start_seq) << " seconds\n";

    cout << "\nParallel BFS Traversal:\n";
    double start_par = omp_get_wtime();
    bfsObj.bfs(root);
    double end_par = omp_get_wtime();
    cout << "\nParallel Execution Time: " << (end_par - start_par) << " seconds\n";

    return 0;
}

