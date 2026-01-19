#include <mpi.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <algorithm>

#include <Eigen/Dense>
#include <Eigen/LU>

using namespace std;
using namespace Eigen;


class Partition {
    public:
    static void compute_1d_partition(int N, int nb, int pid, int &start, int &len) {
        int base = N / nb;                          // divide N point equally, the first "rest" processes get base+1 elements
        int rest = N % nb;
        start = pid * base + std::min(pid, rest);
        len = base + (pid < rest ? 1 : 0);
    }

};


class LocalProblem {

};


class SchwarzSolver {

};


int main(int argc, char** argv) {






    return 0;
}



