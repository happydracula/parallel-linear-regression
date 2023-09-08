#include <iostream>
#include <omp.h>
#include <math.h>
#include <vector>
#include <fstream>
#include <string.h>
#include <sstream>
#include "matrix.h"
#include <chrono>
#include <algorithm>
using namespace std::chrono;
using namespace std;
class LinearRegressor
{
private:
    int n;
    int m;
    double yMinval, yMaxVal;
    double xMinval, xMaxVal;
    vector<vector<double>> Y;
    vector<vector<double>> X;
    vector<vector<double>> theta;
    int parallel;

public:
    LinearRegressor(int m, int n, vector<vector<double>> &x, vector<vector<double>> &y, int parallel)
    {
        this->parallel = parallel;
        this->m = m;
        this->n = n;
        X = x;
        Y = y;

        for (int i = 0; i < n; i++)
            theta.push_back(vector<double>{((double)rand()) / RAND_MAX});

        scaleValues();
    }

    void scaleValues()
    {
        for (int i = 1; i < n; i++)
        {
            xMinval = findMin(X, i);
            xMaxVal = findMax(X, i);
            for (int j = 0; j < m; j++)
            {
                X[j][i] = (X[j][i] - xMinval) / (xMaxVal - xMinval);
            }
        }
        yMinval = findMin(Y, 0);
        yMaxVal = findMax(Y, 0);
        for (int i = 0; i < m; i++)
        {
            Y[i][0] = (Y[i][0] - yMinval) / (yMaxVal - yMinval);
        }
    }
    vector<vector<double>> generateHypothesis()
    {
        return dot(X, theta, parallel);
    }

    double getRSS()
    {
        vector<vector<double>> temp = add(dot(X, theta, parallel), Y, -1);
        return dot(transpose(temp), temp, parallel)[0][0];
    }
    double getCost()
    {
        return getRSS() / (2 * m);
    }
    vector<vector<double>> differential()
    {
        return dot(transpose(X), add(dot(X, theta, parallel), Y, -1), parallel);
    }
    double getTSS()
    {
        double TSS = 0;
        double mean = 0;
        for (int i = 0; i < m; i++)
        {
            mean += Y[i][0];
        }
        mean /= m;
        for (int i = 0; i < m; i++)
        {
            TSS += pow(Y[i][0] - mean, 2);
        }
        return TSS;
    }
    double getAccuracy()
    {
        return (1 - (getRSS() / getTSS())) * 100;
    }
    void gradientDescent(double alfa, int epochs)
    {
        for (int i = 0; i < epochs; i++)
        {
            // cout << getAccuracy() << endl;
            theta = add(theta, differential(), -(alfa / m));
        }
    }
    void predict(vector<vector<double>> testX)
    {
        int test_m = testX.size();

        for (int i = 0; i < test_m; i++)
        {
            testX[i].insert(testX[i].begin(), 1);
        }

        int test_n = testX[0].size();
        for (int i = 1; i < test_n; i++)
        {

            for (int j = 0; j < test_m; j++)
            {
                testX[j][i] = (testX[j][i] - xMinval) / (xMaxVal - xMinval);
            }
        }

        vector<vector<double>> res = dot(testX, theta, parallel);
        for (int i = 0; i < test_m; i++)
        {
            res[i][0] = (res[i][0] * (yMaxVal - yMinval)) + yMinval;
            cout << res[i][0] << endl;
        }
    }
};
void readInput(string inputPath, vector<vector<double>> &X, vector<vector<double>> &Y)
{
    int m = 0;
    fstream file(inputPath, ios::in);
    string line, word;
    int line_num = 0;
    if (file.is_open())
    {
        while (getline(file, line))
        {
            line_num++;
            if (line_num == 1)
                continue;
            stringstream str(line);
            vector<double> temp;
            while (getline(str, word, ','))
            {
                // cout << word.size() << " ";
                word.erase(remove(word.begin(), word.end(), ' '), word.end());
                temp.push_back(stod(word));
            }
            X.push_back(vector<double>());
            X[m].push_back(1);
            for (int i = 0; i < temp.size() - 1; i++)
            {
                X[m].push_back(temp[i]);
            }

            m++;
            Y.push_back(vector<double>{temp[temp.size() - 1]});
        }
    }
    else
        cout << "Could not open the file\n";
}
int main()
{

    vector<vector<double>> X;
    vector<vector<double>> Y;
    cout << "Enter the path of the dataset:";
    string path;
    cin >> path;
    readInput(path, X, Y);
    int m = X.size();
    int n = X[0].size();
    double alfa;
    int epochs;
    cout << endl
         << "Enter the alfa value:";
    cin >> alfa;
    cout << endl
         << "Enter the number of epochs:";
    cin >> epochs;
    cout << endl;
    LinearRegressor lr_normal(m, n, X, Y, 0);
    LinearRegressor lr_parallel(m, n, X, Y, 1);

    auto start = high_resolution_clock::now();
    lr_normal.gradientDescent(alfa, epochs);
    cout << lr_normal.getAccuracy() << endl;
    // lr.predict(vector<vector<double>>{{1.1}, {1.5}});
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    cout << "Time taken to train without parallelisation: "
         << duration.count() << " microseconds" << endl;
    start = high_resolution_clock::now();
    lr_parallel.gradientDescent(alfa, epochs);
    cout << lr_parallel.getAccuracy() << endl;
    // lr.predict(vector<vector<double>>{{1.1}, {1.5}});
    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
    cout << "Time taken to train with parallelisation: "
         << duration.count() << " microseconds" << endl;
}