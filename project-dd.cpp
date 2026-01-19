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
    public:
    LocalProblem() : ex_i0(0), ex_i1(-1), ex_j0(0), ex_j1(-1), ex_nx(0), ex_ny(0), ex_n(0),
                     ci_s(0), ci_e(-1), cj_s(0), cj_e(-1), core_nx(0), core_ny(0), core_n(0),
                     Nx(0), Ny(0), hx(0.0), hy(0.0), mu(0.0), c(0.0), lu_ok(false) {}

    // init: exactly same inputs as original assembly code
    void init(int _ex_i0, int _ex_i1, int _ex_j0, int _ex_j1,
              int _ci_s, int _ci_e, int _cj_s, int _cj_e,
              int _Nx, int _Ny, double _hx, double _hy,
              double _mu, double _c)
    {


    }

    private:


    

};


class SchwarzSolver {

};


int main(int argc, char** argv) {






    return 0;
}



