// 2D Parallel Two-Level Schwarz Solver using MPI and Eigen library
// to solve a PDE (Reaction-Diffusion equation) on 2D unit square grid
// using Two-Level Restricted Additive Schwarz as a preconditioner + GMRES

// coarse solver running on all processes (not only on rank 0)
// - CoarseSolver Constructor: every rank builds and factorizes the small coarse matrix
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


// ---------- CoarseSolver ----------
// handles the second level of the Schwarz method: 
// global coarse grid that allows information to propagate across the entire domain in one step
class CoarseSolver {
    int Ncx, Ncy;  // coarse grid dimensions
    int Nx, Ny;    // fine grid dimensions
    SparseMatrix<double> Ac;   // sparse matrix
    SparseLU<SparseMatrix<double>> lu_coarse;
    int rank;

public:
    CoarseSolver(int _Nx, int _Ny, int _Ncx, int _Ncy, double mu, double c, int _rank) 
        : Nx(_Nx), Ny(_Ny), Ncx(_Ncx), Ncy(_Ncy), rank(_rank) {
        
        // every rank stores and solves the coarse system 
        int n_coarse = Ncx * Ncy;
        Ac.resize(n_coarse, n_coarse);
        vector<Triplet<double>> triplets;  // global matrix representing the same PDE but on the coarse grid
        
        double hcx = 1.0 / (Ncx - 1);
        double hcy = 1.0 / (Ncy - 1);
        double idx2 = 1.0 / (hcx*hcx);
        double idy2 = 1.0 / (hcy*hcy);

        // assemble coarse system matrix (identical on all processes)
        for (int j = 0; j < Ncy; ++j) {
            for (int i = 0; i < Ncx; ++i) {
                int row = idlocal(i, j, Ncx);  // 2D index to a 1D row index
                if (i == 0 || i == Ncx-1 || j == 0 || j == Ncy-1) {
                    triplets.push_back(Triplet<double>(row, row, 1.0));
                    // Dirichlet boundary condition (fix the value at the boundary)
                    continue;
                }
                triplets.push_back(Triplet<double>(row, row, mu * (2*idx2 + 2*idy2) + c));
                triplets.push_back(Triplet<double>(row, idlocal(i-1, j, Ncx), -mu * idx2));
                triplets.push_back(Triplet<double>(row, idlocal(i+1, j, Ncx), -mu * idx2));
                triplets.push_back(Triplet<double>(row, idlocal(i, j-1, Ncx), -mu * idy2));
                triplets.push_back(Triplet<double>(row, idlocal(i, j+1, Ncx), -mu * idy2));
                // build the 5-point finite difference stencil (consider interior nodes)
            }
        }
        Ac.setFromTriplets(triplets.begin(), triplets.end());  // convert the triplet list into a sparse matrix Ac
        lu_coarse.compute(Ac);   // factorize the matrix with a sparse LU solver
    }

    // solve the coarse problem: restriction -> solve -> prolongation
    void solve(const vector<double>& r_glob, vector<double>& e_local, int ci_s, int cj_s, int core_nx, int core_ny) {
        int n_coarse = Ncx * Ncy;
        VectorXd rc = VectorXd::Zero(n_coarse);
        
        // restriction: (fine grid points sampled for coarse grid)
        // take the global fine residual and samples it at the coarse grid locations
        double rx = (double)(Nx - 1) / (Ncx - 1);
        double ry = (double)(Ny - 1) / (Ncy - 1);

        for (int jc = 0; jc < Ncy; ++jc) {
            for (int ic = 0; ic < Ncx; ++ic) {
                rc[idlocal(ic, jc, Ncx)] = r_glob[idlocal(round(ic*rx), round(jc*ry), Nx)];
            }
        }

        VectorXd ec = lu_coarse.solve(rc);   // solve the global system on the coarse grid

        // prolongation: (back to fine grid)
        // every process calculates only its local piece of the global error correction
        e_local.assign(core_nx * core_ny, 0.0);
        for (int j = 0; j < core_ny; ++j) {
            for (int i = 0; i < core_nx; ++i) {
                int gi = ci_s + i;    // global coarse grid coordinates corresponding to the local (i,j) point
                int gj = cj_s + j;

                double xc = gi / rx;  // map the local coordinates to coarse-grid 'real' coordinates
                double yc = gj / ry;
                int i0 = (int)xc; 
                int i1 = min(i0 + 1, Ncx - 1);
                int j0 = (int)yc; 
                int j1 = min(j0 + 1, Ncy - 1);
                double dx = xc - i0; 
                double dy = yc - j0;
                
                e_local[idlocal(i, j, core_nx)] =                // bilinear interpolation
                    (1-dx)*(1-dy)*ec[idlocal(i0, j0, Ncx)] +
                    dx*(1-dy)*ec[idlocal(i1, j0, Ncx)] +
                    (1-dx)*dy*ec[idlocal(i0, j1, Ncx)] +
                    dx*dy*ec[idlocal(i1, j1, Ncx)];
            }
        }
    }
};


// ---------- LocalProblem (2D) -----
// Incapsula la costruzione della matrice locale estesa, la fattorizzazione con Eigen
// e l'applicazione del precondizionatore RAS (solve locale + restrizione).

// class to manage the overlapping subdomains
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
        // extended
        ex_i0 = _ex_i0; ex_i1 = _ex_i1; ex_j0 = _ex_j0; ex_j1 = _ex_j1;
        ex_nx = (ex_i1 >= ex_i0) ? (ex_i1 - ex_i0 + 1) : 0;
        ex_ny = (ex_j1 >= ex_j0) ? (ex_j1 - ex_j0 + 1) : 0;
        ex_n = ex_nx * ex_ny;

        // core
        ci_s = _ci_s; ci_e = _ci_e; cj_s = _cj_s; cj_e = _cj_e;   // core index start (global coordinates)
        core_nx = (ci_e >= ci_s) ? (ci_e - ci_s + 1) : 0;         // number of points in the core area
        core_ny = (cj_e >= cj_s) ? (cj_e - cj_s + 1) : 0;
        core_n = core_nx * core_ny;                               // total points in the core

        Nx = _Nx; Ny = _Ny; hx = _hx; hy = _hy; mu = _mu; c = _c;
        // Nx and Ny are the total number of points in the global grid along the X and Y axes
        // hx, hy is the mesh size (distance between points)
        // mu is the diffusion coefficient
        // c is the reaction coefficient

        assemble_and_factorize();

    }

     // bool is_lu_ok() const { return lu_ok; }

     // take the global residual and map it to the local extended subdomain
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

        VectorXd xloc = eig_lu.solve(r_loc);  // solve A_loc x = r using the pre-calculated LU factorization

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
        SparseMatrix<double> A_loc;
        SparseLU<SparseMatrix<double>> eig_lu; // stores the LU factorization
        bool lu_ok;


        void assemble_and_factorize() {
        lu_ok = true;
        if (ex_n <= 0) { lu_ok = false; return; }

        A_loc.resize(ex_n, ex_n);
        vector<Triplet<double>> triplets;  // build a local matrix representing the 5-point stencil Finite Difference discretization
                                           // stores the coefficients of the local system including the overlap
        double idx2 = 1.0 /(hx*hx);
        double idy2 = 1.0 /(hy*hy);
        double diag_center = mu * (2.0*idx2 + 2.0*idy2) + c;  // main diagonal value
        double diag_off_x = -mu * idx2;
        double diag_off_y = -mu * idy2;

        for (int j = 0; j < ex_ny; ++j) {
            for (int i = 0; i < ex_nx; ++i) {
                int row = idlocal(i,j,ex_nx);
                int gi = ex_i0 + i;
                int gj = ex_j0 + j;
                if (gi == 0 || gi == Nx-1 || gj == 0 || gj == Ny-1) {
                    triplets.push_back(Triplet<double>(row,row,1.0));
                    continue;
                }
                triplets.push_back(Triplet<double>(row,row, mu * (2.0*idx2 + 2.0*idy2) + c));  // main diagonal value
                if (gi - 1 >= ex_i0) triplets.push_back(Triplet<double>(row, idlocal(i-1,j,ex_nx), diag_off_x));
                if (gi + 1 <= ex_i1) triplets.push_back(Triplet<double>(row, idlocal(i+1,j,ex_nx), diag_off_x));
                if (gj - 1 >= ex_j0) triplets.push_back(Triplet<double>(row, idlocal(i,j-1,ex_nx), diag_off_y));
                if (gj + 1 <= ex_j1) triplets.push_back(Triplet<double>(row, idlocal(i,j+1,ex_nx), diag_off_y));
            }
        }
        A_loc.setFromTriplets(triplets.begin(), triplets.end());
        eig_lu.compute(A_loc);
    }
};


// ---------- SchwarzSolver (2D) -----
// Incapsula lo scambio halo, il matvec, l'applicazione del precondizionatore locale e
// l'algoritmo GMRES restartable. Ha interfaccia simile all'esempio 1D (run()).

// this class handles MPI communication + global iterative algorithm (GMRES)
class SchwarzSolver {
    public:
    SchwarzSolver(MPI_Comm cart_comm, int rank, int size,
                  int Nx, int Ny,
                  int ci_s, int core_nx, int cj_s, int core_ny,
                  double hx, double hy, double mu, double c,
                  int left, int right, int down, int up,
                  LocalProblem *localProb, CoarseSolver *coarseProb)
        : cart(cart_comm), rank(rank), size(size), Nx(Nx), Ny(Ny),
          ci_s(ci_s), cj_s(cj_s), core_nx(core_nx), core_ny(core_ny),
          hx(hx), hy(hy), mu(mu), c(c), left(left), right(right), down(down), up(up),
          local(localProb), coarse(coarseProb)
    {
        core_n = core_nx * core_ny;
        halo_nx = core_nx + 2; halo_ny = core_ny + 2;
        x_halo.assign(halo_nx * halo_ny, 0.0);
        send_left.assign(core_ny, 0.0); recv_left.assign(core_ny, 0.0);       // buffer vectors used to package edge data before sending across the network
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


     // iterative method for solving non-symmetric systems
     void run(int max_it, double tol, int m_restart) {
        // rhs_pre = M^{-1} rhs
        vector<double> x(core_n, 0.0), rhs_pre(core_n, 0.0);
        apply_TwoLevel(b_local, rhs_pre);

        double rhs_norm = sqrt(dot_local(rhs_pre, rhs_pre));
        if (rhs_norm == 0.0) rhs_norm = 1.0;

        // inizial residual r = rhs_pre - M^{-1} A x
        vector<double> Ax(core_n, 0.0), MInvAx(core_n, 0.0);
        matvec(x, Ax);
        apply_TwoLevel(Ax, MInvAx);

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
    CoarseSolver *coarse;

    // halos and buffers
    int halo_nx, halo_ny;
    vector<double> x_halo;
    vector<double> send_left, recv_left, send_right, recv_right;
    vector<double> send_bottom, recv_bottom, send_top, recv_top;

    vector<double> b_local;

    // --- TWO-LEVEL PRECONDITIONER ---
    void apply_TwoLevel(const vector<double> &r_local, vector<double> &z_local) {
        // 1. level 1: RAS Fine Correction  (fine level)
        local->apply_RAS(r_local, z_local);
            // each processor solves its local system (with A_loc) using the pre-calculated LU
            // this fixes "high-frequency" (local) errors

        // 2. level 2: Coarse Grid Correction
        vector<double> r_glob(Nx * Ny, 0.0);
        allgather_global_vector(r_local, r_glob);
            // all processes now have the global residual map

        vector<double> e_local_coarse(core_n, 0.0);
        coarse->solve(r_glob, e_local_coarse, ci_s, cj_s, core_nx, core_ny);
            // every process solves the global coarse system and prolongs only to its local core points

        // Add coarse correction to RAS result (Additive Schwarz)
        // the local RAS result and the global coarse result are added together
        for (int i = 0; i < core_n; ++i) {
            z_local[i] += e_local_coarse[i];
        }
    }

    // give to every process the full global vector (needed for global coarse restriction)
    void allgather_global_vector(const vector<double> &loc, vector<double> &glob) {
        vector<int> recvcounts(size);
        vector<int> displs(size);
        
        int my_n = core_n;    // number of element each process has
        MPI_Allgather(&my_n, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, MPI_COMM_WORLD);
        
        displs[0] = 0;
        for (int i = 1; i < size; ++i) 
            displs[i] = displs[i-1] + recvcounts[i-1];
        
        vector<double> flat_glob(Nx * Ny, 0.0);
        MPI_Allgatherv(loc.data(), core_n, MPI_DOUBLE, flat_glob.data(), recvcounts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        
        // Allgatherv gives us a list of core-blocks. We must map them back to (i,j) coordinates
        // We need to know the ci_s, cj_s, cnx, cny of every process (share the coordinates)
        struct ProcInfo { int cs, rs, cn, rn; };
        vector<ProcInfo> all_info(size);
        ProcInfo my_info = {ci_s, cj_s, core_nx, core_ny};
        MPI_Allgather(&my_info, 4, MPI_INT, all_info.data(), 4, MPI_INT, MPI_COMM_WORLD);

        for (int p = 0; p < size; ++p) {
            int offset = displs[p];
            for (int jj = 0; jj < all_info[p].rn; ++jj) {
                for (int ii = 0; ii < all_info[p].cn; ++ii) {
                    glob[idlocal(all_info[p].cs + ii, all_info[p].rs + jj, Nx)] = flat_glob[offset + idlocal(ii, jj, all_info[p].cn)];
                }
            }
        }
    }


    double dot_local(const vector<double> &a, const vector<double> &b) const {
        double s = 0.0;
        int n = (int)a.size();
        for (int i = 0; i < n; ++i) 
            s += a[i] * b[i];
        double sglob = 0.0;
        MPI_Allreduce(&s, &sglob, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        return sglob;
    }

    void matvec(const vector<double> &p, vector<double> &Ap) {   // to compute matrix-vector product
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

    /*
    void apply_RAS(const vector<double> &r, vector<double> &z) const {
        // delegate to local problem (same semantics)
        local->apply_RAS(r, z);
    }
    */

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
    int Ncx = 9, Ncy = 9; // coarse grid size

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
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart);  // create a 2D cartesian topology

    int coords[2];
    MPI_Cart_coords(cart, rank, 2, coords);
    int px = coords[0], py = coords[1];

    int ci_s, cnx, cj_s, cny;
    Partition::compute_1d_partition(Nx, Px, px, ci_s, cnx);   // calculates the local 'core' indices
    Partition::compute_1d_partition(Ny, Py, py, cj_s, cny);
    int ci_e = ci_s + cnx - 1;
    int cj_e = cj_s + cny - 1;

    int left, right, down, up;
    MPI_Cart_shift(cart, 0, 1, &left, &right);
    MPI_Cart_shift(cart, 1, 1, &down, &up);

    double hx = 1.0 / (Nx - 1);
    double hy = 1.0 / (Ny - 1);

    // extended domain for RAS
    int ex_i0 = max(0, ci_s - overlap);     // extending the indices by adding the overlap
    int ex_i1 = min(Nx-1, ci_e + overlap);
    int ex_j0 = max(0, cj_s - overlap);
    int ex_j1 = min(Ny-1, cj_e + overlap);

    // create LocalProblem (si occupa di assemblare e fattorizzare la matrice locale)
    LocalProblem *local = new LocalProblem();
    local->init(ex_i0, ex_i1, ex_j0, ex_j1,
                ci_s, ci_e, cj_s, cj_e,
                Nx, Ny, hx, hy, mu, c);

    // CoarseSolver object
    CoarseSolver *coarse = new CoarseSolver(Nx, Ny, Ncx, Ncy, mu, c, rank);

    // create SchwarzSolver object (incapsula matvec, precond e GMRES)
    SchwarzSolver solver(cart, rank, size,
                         Nx, Ny,
                         ci_s, cnx, cj_s, cny,
                         hx, hy, mu, c,
                         left, right, down, up,
                         local, coarse);

    int m = min(30, max_it);
    // run the solver (metodo run simile all'esempio 1D)
    solver.run(max_it, tol, m);

    delete local;
    delete coarse;

    MPI_Finalize();

    return 0;
}
