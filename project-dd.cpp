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

// ---------- Partition helper (stile: class con metodo statico) ----------
class Partition {
    public:
    static void compute_1d_partition(int N, int nb, int pid, int &start, int &len) {
        int base = N / nb;                          // divide N point equally, the first "rest" processes get base+1 elements
        int rest = N % nb;
        start = pid * base + std::min(pid, rest);
        len = base + (pid < rest ? 1 : 0);
    }

};

// ---------- LocalProblem (2D) -----
// Incapsula la costruzione della matrice locale estesa, la fattorizzazione con Eigen
// e l'applicazione del precondizionatore RAS (solve locale + restrizione).

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


// ---------- SchwarzSolver (2D) -----
// Incapsula lo scambio halo, il matvec, l'applicazione del precondizionatore locale e
// l'algoritmo GMRES restartable. Ha interfaccia simile all'esempio 1D (run()).
class SchwarzSolver {
    public:
    SchwarzSolver(MPI_Comm cart_comm, int rank, int size,
                  int Nx, int Ny,
                  int ci_s, int core_nx, int cj_s, int core_ny,
                  double hx, double hy, double mu, double c,
                  int left, int right, int down, int up,
                  LocalProblem *localProb)
        : cart(cart_comm), rank(rank), size(size), Nx(Nx), Ny(Ny),
          ci_s(ci_s), cj_s(cj_s), core_nx(core_nx), core_ny(core_ny),
          hx(hx), hy(hy), mu(mu), c(c), left(left), right(right), down(down), up(up),
          local(localProb)
    {



    }

     ~SchwarzSolver() {
    }


    private:


};


int main(int argc, char** argv) {
 MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // default parameters
    int Nx = 51, Ny = 51;
    int Px = 0, Py = 0;
    int overlap = 2;
    double mu = 0.01, c = 5.0;
    int max_it = 1000;
    double tol = 1e-6;

    if (argc >= 3) { Nx = stoi(argv[1]); Ny = stoi(argv[2]); }
    if (argc >= 5) { Px = stoi(argv[3]); Py = stoi(argv[4]); }
    if (argc >= 6) overlap = stoi(argv[5]);
    if (argc >= 7) mu = stod(argv[6]);
    if (argc >= 8) c = stod(argv[7]);
    if (argc >= 9) max_it = stoi(argv[8]);
    if (argc >= 10) tol = stod(argv[9]);





    MPI_Finalize();

    return 0;
}



