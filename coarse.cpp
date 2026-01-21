// 2D Parallel Two-Level Schwarz Solver using MPI and Eigen library
// to solve a PDE (Reaction-Diffusion equation) on 2D unit square grid
// using Two-Level Restricted Additive Schwarz as a preconditioner + GMRES

// coarse solver running on all processes (not only on rank 0)
// - CoarseSolver Constructor: every rank builds and factorizes the small coarse matrix
    // since the coarse grid is typically very small, the redundant memory and computation cost is negligible compared to the communication cost of gathering everything to rank 0
// - apply_TwoLevel: instead of gather_solution, it uses MPI_Allgather  (allows every process to have the full residual vector)
// - solve: every process performs the restriction, the LU solve and the prolongation locally (eliminates the need for MPI_Bcast and reduces synchronization bottlenecks)


#include <mpi.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <algorithm>

#include <Eigen/Sparse>
#include <Eigen/SparseLU>

using namespace std;
using namespace Eigen;

// ---------- utility free inline ----------
inline int idlocal(int i, int j, int nx) { return i + j * nx; }
    // converts 2D grid coordinates into a single 1D index for memory storage (row-major)

// ---------- Partition helper ----------
class Partition {
    public:
    static void compute_1d_partition(int N, int nb, int pid, int &start, int &len) {
        int base = N / nb;                          // divide N point equally, the first "rest" processes get base+1 elements
        int rest = N % nb;
        start = pid * base + std::min(pid, rest);
        len = base + (pid < rest ? 1 : 0);
    }
};     // splits the N grid points into nb processes


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

        ex_i0 = _ex_i0; ex_i1 = _ex_i1; ex_j0 = _ex_j0; ex_j1 = _ex_j1;
        ex_nx = (ex_i1 >= ex_i0) ? (ex_i1 - ex_i0 + 1) : 0;
        ex_ny = (ex_j1 >= ex_j0) ? (ex_j1 - ex_j0 + 1) : 0;
        ex_n = ex_nx * ex_ny;

        ci_s = _ci_s; ci_e = _ci_e; cj_s = _cj_s; cj_e = _cj_e;
        core_nx = (ci_e >= ci_s) ? (ci_e - ci_s + 1) : 0;
        core_ny = (cj_e >= cj_s) ? (cj_e - cj_s + 1) : 0;
        core_n = core_nx * core_ny;

        Nx = _Nx; Ny = _Ny; hx = _hx; hy = _hy; mu = _mu; c = _c;

        assemble_and_factorize();

    }

     bool is_lu_ok() const { return lu_ok; }

     // apply RAS: r_core (size core_n) -> z_core (size core_n)
     void apply_RAS(const vector<double> &r_core, vector<double> &z_core) const {
        z_core.assign(core_n, 0.0);
        if (ex_n == 0 || !lu_ok) return;

        VectorXd r_loc(ex_n);
        r_loc.setZero();

        for (int j = 0; j < ex_ny; ++j) {
            for (int i = 0; i < ex_nx; ++i) {
                int gi = ex_i0 + i;
                int gj = ex_j0 + j;
                int loc = idlocal(i, j, ex_nx);
                if (gi == 0 || gi == Nx-1 || gj == 0 || gj == Ny-1) {
                    r_loc[loc] = 0.0;
                    continue;
                }
                if (gi >= ci_s && gi <= ci_e && gj >= cj_s && gj <= cj_e) {
                    int ii = gi - ci_s;
                    int jj = gj - cj_s;
                    int cid = idlocal(ii, jj, core_nx);
                    r_loc[loc] = r_core[cid];
                } else {
                    r_loc[loc] = 0.0;
                }
            }
        }

        VectorXd xloc = eig_lu.solve(r_loc);

        for (int j = 0; j < core_ny; ++j)
            for (int i = 0; i < core_nx; ++i) {
                int gi = ci_s + i;
                int gj = cj_s + j;
                if (gi == 0 || gi == Nx-1 || gj == 0 || gj == Ny-1) continue;
                int ei = gi - ex_i0;
                int ej = gj - ex_j0;
                int loc = idlocal(ei, ej, ex_nx);
                int cid = idlocal(i, j, core_nx);
                z_core[cid] = xloc[loc];
            }
    }

     // getters used later for gathering
    int ext_start() const { return ex_i0; } // WARNING: these are 2D extents in original; keep semantics used by gather
    int ext_end()   const { return ex_i1; }
    int ext_nx()    const { return ex_nx; }
    int ext_ny()    const { return ex_ny; }
    int core_nx_get() const { return core_nx; }
    int core_ny_get() const { return core_ny; }
    int core_start_x() const { return ci_s; }
    int core_start_y() const { return cj_s; }

    private:
        // geometry
        int ex_i0, ex_i1, ex_j0, ex_j1;
        int ex_nx, ex_ny, ex_n;
        int ci_s, ci_e, cj_s, cj_e;
        int core_nx, core_ny, core_n;
        int Nx, Ny;

        // PDE / grid
        double hx, hy, mu, c;

        // eigen LU
        MatrixXd A_loc;
        PartialPivLU<MatrixXd> eig_lu;
        bool lu_ok;


        void assemble_and_factorize() {
        lu_ok = true;
        if (ex_n <= 0) { lu_ok = false; return; }

        A_loc = MatrixXd::Zero(ex_n, ex_n);
        double idx2 = 1.0/(hx*hx);
        double idy2 = 1.0/(hy*hy);
        double diag_center = mu * (2.0*idx2 + 2.0*idy2) + c;
        double diag_off_x = -mu * idx2;
        double diag_off_y = -mu * idy2;

        for (int j = 0; j < ex_ny; ++j) {
            for (int i = 0; i < ex_nx; ++i) {
                int row = idlocal(i,j,ex_nx);
                int gi = ex_i0 + i;
                int gj = ex_j0 + j;
                if (gi == 0 || gi == Nx-1 || gj == 0 || gj == Ny-1) {
                    A_loc(row,row) = 1.0;
                    continue;
                }
                A_loc(row,row) = diag_center;
                if (gi - 1 >= ex_i0) A_loc(row, idlocal(i-1,j,ex_nx)) = diag_off_x;
                if (gi + 1 <= ex_i1) A_loc(row, idlocal(i+1,j,ex_nx)) = diag_off_x;
                if (gj - 1 >= ex_j0) A_loc(row, idlocal(i,j-1,ex_nx)) = diag_off_y;
                if (gj + 1 <= ex_j1) A_loc(row, idlocal(i,j+1,ex_nx)) = diag_off_y;
            }
        }

        eig_lu.compute(A_loc);
    }
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
         core_n = core_nx * core_ny;
        halo_nx = core_nx + 2; halo_ny = core_ny + 2;
        x_halo.assign(halo_nx * halo_ny, 0.0);
        send_left.assign(core_ny, 0.0); recv_left.assign(core_ny, 0.0);
        send_right.assign(core_ny, 0.0); recv_right.assign(core_ny, 0.0);
        send_bottom.assign(core_nx, 0.0); recv_bottom.assign(core_nx, 0.0);
        send_top.assign(core_nx, 0.0); recv_top.assign(core_nx, 0.0);

        // initialize RHS exactly as original
        b_local.assign(core_n, 0.0);
        for (int j = 0; j < core_ny; ++j) {
            for (int i = 0; i < core_nx; ++i) {
                int gi = ci_s + i;
                int gj = cj_s + j;
                int lid = idlocal(i, j, core_nx);
                if (gi == 0 || gi == Nx-1 || gj == 0 || gj == Ny-1)
                    b_local[lid] = 0.0;
                else
                    b_local[lid] = 1.0;
            }
        }

    }

     ~SchwarzSolver() {
    }


     void run(int max_it, double tol, int m_restart) {
        // rhs_pre = M^{-1} rhs
        vector<double> rhs_pre(core_n, 0.0);
        apply_RAS(b_local, rhs_pre);

        double rhs_norm = sqrt(dot_local(rhs_pre, rhs_pre));
        if (rhs_norm == 0.0) rhs_norm = 1.0;

        // inizial residual r = rhs_pre - M^{-1} A x
        vector<double> x(core_n, 0.0);
        vector<double> Ax(core_n, 0.0), MInvAx(core_n, 0.0);
        matvec(x, Ax);
        apply_RAS(Ax, MInvAx);

        vector<double> r(core_n, 0.0);
        for (int i = 0; i < core_n; ++i) r[i] = rhs_pre[i] - MInvAx[i];


        //GMRES METHOD ...
        

        // gather and save same as original
        gather_solution(x);
    }


    private:

     MPI_Comm cart;
    int rank, size;
    int Nx, Ny;
    int ci_s, cj_s;
    int core_nx, core_ny, core_n;
    double hx, hy, mu, c;
    int left, right, down, up;

    LocalProblem *local; // non-owning pointer: ownership in main

    // halos and buffers
    int halo_nx, halo_ny;
    vector<double> x_halo;
    vector<double> send_left, recv_left, send_right, recv_right;
    vector<double> send_bottom, recv_bottom, send_top, recv_top;

    vector<double> b_local;


    double dot_local(const vector<double> &a, const vector<double> &b) const {
        double s = 0.0;
        int n = (int)a.size();
        for (int i = 0; i < n; ++i) s += a[i] * b[i];
        double sglob = 0.0;
        MPI_Allreduce(&s, &sglob, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        return sglob;
    }

    void matvec(const vector<double> &p, vector<double> &Ap) {
        double idx2 = 1.0/(hx*hx);
        double idy2 = 1.0/(hy*hy);
        double diag_center = mu * (2.0*idx2 + 2.0*idy2) + c;
        double diag_off_x = -mu * idx2;
        double diag_off_y = -mu * idy2;

        for (int j = 0; j < core_ny; ++j)
            for (int i = 0; i < core_nx; ++i)
                x_halo[idlocal(i+1, j+1, halo_nx)] = p[idlocal(i,j,core_nx)];

        for (int j = 0; j < core_ny; ++j) {
            send_left[j]  = p[idlocal(0, j, core_nx)];
            send_right[j] = p[idlocal(core_nx-1, j, core_nx)];
        }
        for (int i = 0; i < core_nx; ++i) {
            send_bottom[i] = p[idlocal(i, 0, core_nx)];
            send_top[i]    = p[idlocal(i, core_ny-1, core_nx)];
        }

        MPI_Status st;

        MPI_Sendrecv(send_right.data(), core_ny, MPI_DOUBLE, right, 0,
                     recv_left.data(),  core_ny, MPI_DOUBLE, left,  0,
                     cart, &st);

        MPI_Sendrecv(send_left.data(), core_ny, MPI_DOUBLE, left,  1,
                     recv_right.data(), core_ny, MPI_DOUBLE, right, 1,
                     cart, &st);

        MPI_Sendrecv(send_top.data(), core_nx, MPI_DOUBLE, up, 2,
                     recv_bottom.data(), core_nx, MPI_DOUBLE, down, 2,
                     cart, &st);

        MPI_Sendrecv(send_bottom.data(), core_nx, MPI_DOUBLE, down, 3,
                     recv_top.data(), core_nx, MPI_DOUBLE, up, 3,
                     cart, &st);

        for (int j = 0; j < core_ny; ++j) {
            x_halo[idlocal(0, j+1, halo_nx)]         = recv_left[j];
            x_halo[idlocal(halo_nx-1, j+1, halo_nx)] = recv_right[j];
        }
        for (int i = 0; i < core_nx; ++i) {
            x_halo[idlocal(i+1, 0, halo_nx)]           = recv_bottom[i];
            x_halo[idlocal(i+1, halo_ny-1, halo_nx)]   = recv_top[i];
        }

        for (int j = 0; j < core_ny; ++j) {
            for (int i = 0; i < core_nx; ++i) {
                int gi = ci_s + i;
                int gj = cj_s + j;
                int lid = idlocal(i, j, core_nx);
                if (gi == 0 || gi == Nx-1 || gj == 0 || gj == Ny-1) {
                    Ap[lid] = p[lid];
                    continue;
                }
                double uc = x_halo[idlocal(i+1, j+1, halo_nx)];
                double ul = x_halo[idlocal(i,   j+1, halo_nx)];
                double ur = x_halo[idlocal(i+2, j+1, halo_nx)];
                double ud = x_halo[idlocal(i+1, j,   halo_nx)];
                double uu = x_halo[idlocal(i+1, j+2, halo_nx)];
                Ap[lid] = diag_center * uc + diag_off_x * (ul + ur) + diag_off_y * (ud + uu);
            }
        }
    }

    void apply_RAS(const vector<double> &r, vector<double> &z) const {
        // delegate to local problem (same semantics)
        local->apply_RAS(r, z);
    }

     // gather solution (identical semantic to original: rank0 collects core pieces and writes solution.csv)
    void gather_solution(const vector<double> &x_local) {
        if (rank == 0) {
            vector<double> U(Nx * Ny, 0.0);
            for (int j = 0; j < core_ny; ++j)
                for (int i = 0; i < core_nx; ++i) {
                    int gi = ci_s + i;
                    int gj = cj_s + j;
                    U[idlocal(gi, gj, Nx)] = x_local[idlocal(i,j,core_nx)];
                }
            for (int p = 1; p < size; ++p) {
                MPI_Status st;
                int meta[4];
                MPI_Recv(meta, 4, MPI_INT, p, 300, MPI_COMM_WORLD, &st);
                int rs = meta[0], cs = meta[1], rn = meta[2], cn = meta[3];
                vector<double> buf(rn*cn);
                MPI_Recv(buf.data(), rn*cn, MPI_DOUBLE, p, 301, MPI_COMM_WORLD, &st);
                for (int jj=0; jj<cn; ++jj)
                    for (int ii=0; ii<rn; ++ii) {
                        int gi = rs + ii;
                        int gj = cs + jj;
                        U[idlocal(gi, gj, Nx)] = buf[idlocal(ii,jj,rn)];
                    }
            }
            ofstream ofs("solution.csv");
            ofs << "x,y,u\n";
            for (int j=0;j<Ny;++j) {
                for (int i=0;i<Nx;++i) {
                    double xg = i * (1.0/(Nx - 1));
                    double yg = j * (1.0/(Ny - 1));
                    ofs << xg << "," << yg << "," << U[idlocal(i,j,Nx)] << "\n";
                }
            }
            ofs.close();
            cout << "solution.csv written by rank 0";
        } else {
            int meta[4] = { ci_s, cj_s, core_nx, core_ny };
            MPI_Send(meta, 4, MPI_INT, 0, 300, MPI_COMM_WORLD);
            MPI_Send(x_local.data(), core_n, MPI_DOUBLE, 0, 301, MPI_COMM_WORLD);
        }
    }


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


    int dims[2] = {Px, Py};
    if (dims[0] == 0 || dims[1] == 0) {
        dims[0] = dims[1] = 0;
        MPI_Dims_create(size, 2, dims);
    }
    Px = dims[0]; Py = dims[1];
    int periods[2] = {0,0};
    MPI_Comm cart;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart);

    int coords[2];
    MPI_Cart_coords(cart, rank, 2, coords);
    int px = coords[0], py = coords[1];

    int ci_s, cnx, cj_s, cny;
    Partition::compute_1d_partition(Nx, Px, px, ci_s, cnx);
    Partition::compute_1d_partition(Ny, Py, py, cj_s, cny);
    int ci_e = ci_s + cnx - 1;
    int cj_e = cj_s + cny - 1;

    int left, right, down, up;
    MPI_Cart_shift(cart, 0, 1, &left, &right);
    MPI_Cart_shift(cart, 1, 1, &down, &up);

    double hx = 1.0 / (Nx - 1);
    double hy = 1.0 / (Ny - 1);

    // extended domain for RAS
    int ex_i0 = max(0, ci_s - overlap);
    int ex_i1 = min(Nx-1, ci_e + overlap);
    int ex_j0 = max(0, cj_s - overlap);
    int ex_j1 = min(Ny-1, cj_e + overlap);

    // create LocalProblem (si occupa di assemblare e fattorizzare la matrice locale)
    LocalProblem *local = new LocalProblem();
    local->init(ex_i0, ex_i1, ex_j0, ex_j1,
                ci_s, ci_e, cj_s, cj_e,
                Nx, Ny, hx, hy, mu, c);

    // create SchwarzSolver object (incapsula matvec, precond e GMRES)
    SchwarzSolver solver(cart, rank, size,
                         Nx, Ny,
                         ci_s, cnx, cj_s, cny,
                         hx, hy, mu, c,
                         left, right, down, up,
                         local);

    int m = min(30, max_it);
    // run the solver (metodo run simile all'esempio 1D)
    solver.run(max_it, tol, m);

    delete local;

    MPI_Finalize();
    

    return 0;
}
