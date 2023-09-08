#include <vector>
#include <omp.h>
#include <climits>
using namespace std;
double findMax(vector<vector<double>> &mat, int col)
{
    double res = INT_MIN;
    int m = mat.size();
    for (int i = 0; i < m; i++)
    {
        res = max(res, mat[i][col]);
    }
    return res;
}
double findMin(vector<vector<double>> &mat, int col)
{
    double res = INT_MAX;
    int m = mat.size();
    for (int i = 0; i < m; i++)
    {
        res = min(res, mat[i][col]);
    }
    return res;
}
vector<vector<double>> dot(vector<vector<double>> a, vector<vector<double>> b, int isParallel)
{
    int m1 = a.size();
    int n1 = a[0].size();
    int m2 = b.size();
    int n2 = b[0].size();
    if (n1 != m2)
    {
        cout << "Error:Input not compatible for multiplication\n";
        return {};
    }
    vector<vector<double>> res(m1, vector<double>(n2, 0));
#pragma omp parallel for if (isParallel == 1) num_threads(8) default(private) shared(m1, n1, n2, a, b, res) schedule(static)
    for (int i = 0; i < m1; i++)
    {
        for (int j = 0; j < n2; j++)
        {
            double sum = 0;
            for (int k = 0; k < n1; k++)
            {
                sum += a[i][k] * b[k][j];
            }
            res[i][j] = sum;
            // cout << res[i][j] << " " << Y[i][j] << endl;
        }
    }
    // cout << res.size() << " " << res[0].size() << endl;
    return res;
}
vector<vector<double>> transpose(vector<vector<double>> mat)
{
    int m = mat.size();
    int n = mat[0].size();
    vector<vector<double>> res(n, vector<double>(m));
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            res[i][j] = mat[j][i];
        }
    }
    return res;
}
vector<vector<double>> add(vector<vector<double>> a, vector<vector<double>> b, double minus)
{
    int m = a.size();
    int n = a[0].size();
    vector<vector<double>> res(m, vector<double>(n));
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
            res[i][j] = a[i][j] + (minus * b[i][j]);
    }
    return res;
}