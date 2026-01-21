#include "ras_2d.hpp"
#include "gmres.hpp"
#include "bicgstab.hpp"


// ============================================================
// PARTITIONING
// ============================================================

void Partition::compute_1d_partition(int N, int nb, int proc_id, int& start, int& len) {

    // Divide N nodes into nb blocks as evenly as possible:
    //   -  base = N/nb
    //   -  first "rest" processes get base+1 nodes

    const int base = N / nb;
    const int rest = N % nb;

    start = proc_id * base + std::min(proc_id, rest);
    len = base + (proc_id < rest ? 1 : 0);
}


// ============================================================
// COARSE SOLVER
// ============================================================

CoarseSolver::CoarseSolver(int Nx_, int Ny_, int Ncx_, int Ncy_, double mu_, double c_, int rank_)
    : Nx(_Nx), Ny(_Ny), 
      Ncx(_Ncx), Ncy(_Ncy), 
      rank(_rank)
{
    // Every rank now stores and solves the coarse system 
    int n_coarse = Ncx * Ncy;
    Ac.resize(n_coarse, n_coarse);
    std::vector<Eigen::Triplet<double>> trips;  // Global matrix representing the PDE on the coarse grid

    // Lambda function to add a triplet
    auto add_trip = [&](int r, int cid, double val) {
       trips.emplace_back(static_cast<Eigen::Index>(r), static_cast<Eigen::Index>(cid), val);
    };
    
    double hcx = 1.0 / (Ncx - 1);
    double hcy = 1.0 / (Ncy - 1);
    double idx2 = 1.0 / (hcx * hcx);
    double idy2 = 1.0 / (hcy * hcy);

    // Assemble coarse system matrix 
    for (int j = 0; j < Ncy; ++j) {
            for (int i = 0; i < Ncx; ++i) {
                int row = idlocal(i, j, Ncx);   // 2D index to a 1D row index
                if (i == 0 || i == Ncx-1 || j == 0 || j == Ncy-1) {
                    add_trip(row, row, 1.0);
                    // Dirichlet boundary condition (fix the value at the boundary)
                    continue;
                }
                
                // Diagonal Entry
                add_trip(row, row, mu * (2 * idx2 + 2 * idy2) + c);

                if (i > 0)       add_trip(row, idlocal(i - 1, j, Ncx), -mu * idx2);
                if (i < Ncx - 1) add_trip(row, idlocal(i + 1, j, Ncx), -mu * idx2);
                if (j > 0)       add_trip(row, idlocal(i, j - 1, Ncx), -mu * idy2);
                if (j < Ncy - 1) add_trip(row, idlocal(i, j + 1, Ncx), -mu * idy2);
            }
        }
        Ac.setFromTriplets(triplets.begin(), triplets.end());
        lu_coarse.compute(Ac); 
}

// Solve the coarse problem: Sparse Restriction -> Solve -> Prolongation
void CoarseSolver::solve(const Eigen::VectorXd& r_local, Eigen::VectorXd& e_local, 
           int ci_s, int cj_s, int core_nx, int core_ny, MPI_Comm comm_to_use) 
{
        int n_coarse = Ncx * Ncy;
        // Each process creates a small local contribution vector
        Eigen::VectorXd rc_local_contrib = Eigen::VectorXd::Zero(n_coarse);  
        Eigen::VectorXd rc = Eigen::VectorXd::Zero(n_coarse);
        
        // Sparse Restriction: Injection 
        // Each process only samples points that fall within its local core domain
        double rx = (double)(Nx - 1) / (Ncx - 1);
        double ry = (double)(Ny - 1) / (Ncy - 1);

        for (int jc = 0; jc < Ncy; ++jc) {
            int fine_gj = (int)round(jc * ry);    // Global fine Y index
            if (fine_gj >= cj_s && fine_gj < cj_s + core_ny) {   
                for (int ic = 0; ic < Ncx; ++ic) {
                    int fine_gi = (int)round(ic * rx);   // Global fine X index
                    if (fine_gi >= ci_s && fine_gi < ci_s + core_nx) {
                        int local_idx = idlocal(fine_gi - ci_s, fine_gj - cj_s, core_nx);
                        rc_local_contrib[idlocal(ic, jc, Ncx)] = r_local[local_idx];
                    }
                }
            }
        }

        // Communicate only the coarse grid data 
        MPI_Allreduce(rc_local_contrib.data(), rc.data(), n_coarse, MPI_DOUBLE, MPI_SUM, comm_to_use);

        // Solve the global system on the coarse grid
        Eigen::VectorXd ec = lu_coarse.solve(rc);  

        // Prolongation: Bi-linear Interpolation back to fine grid
        // Every process calculates only its local piece of the global error correction
        e_local = Eigen::VectorXd::Zero(core_nx * core_ny);
        for (int j = 0; j < core_ny; ++j) {
            for (int i = 0; i < core_nx; ++i) {
                int gi = ci_s + i;
                int gj = cj_s + j;

                double xc = gi / rx; 
                double yc = gj / ry;
                int i0 = (int)xc; 
                int i1 = std::min(i0 + 1, Ncx - 1);
                int j0 = (int)yc; 
                int j1 = std::min(j0 + 1, Ncy - 1);
                double dx = xc - i0; 
                double dy = yc - j0;
                
                e_local[idlocal(i, j, core_nx)] =            // Bilinear interpolation
                    (1 - dx) * (1 - dy) * ec[idlocal(i0, j0, Ncx)] +
                    dx * (1 - dy) * ec[idlocal(i1, j0, Ncx)] +
                    (1 - dx) * dy * ec[idlocal(i0, j1, Ncx)] +
                    dx * dy * ec[idlocal(i1, j1, Ncx)];
            }
        }
    }


// ============================================================
// LOCAL PROBLEM
// ============================================================

LocalProblem::LocalProblem(int ci_s_, int cj_s_, int c_nx_, int c_ny_, int overlap_,
                           int Nx_, int Ny_, double hx_, double hy_,
                           double mu_, double c_)

  : core_i0(ci_s_), core_j0(cj_s_), core_nx(c_nx_), core_ny(c_ny_), 
    overlap(overlap_),
    Nx(Nx_), Ny(Ny_),
    hx(hx_), hy(hy_), mu(mu_), c(c_),
    lu_ok(false)
{
    core_i1 = core_i0 + core_nx - 1;  // End index along x (core)
    core_j1 = core_j0 + core_ny - 1;  // End index along y (core)

    core_n  = core_nx * core_ny;

    // DOMAIN DECOMPOSITION: compute extended region
    ext_i0 = std::max(0, core_i0 - overlap);
    ext_i1 = std::min(Nx - 1, core_i1 + overlap);
    ext_j0 = std::max(0, core_j0 - overlap);
    ext_j1 = std::min(Ny - 1, core_j1 + overlap);

    ext_nx = ext_i1 - ext_i0 + 1;
    ext_ny = ext_j1 - ext_j0 + 1;
    ext_n  = ext_nx * ext_ny;

    // Construct and factorize local matrix
    assemble_and_factorize();
}


bool LocalProblem::is_lu_ok() const {
    return lu_ok;
}


void LocalProblem::apply_RAS(const Eigen::VectorXd& r_core,
                            Eigen::VectorXd& z_core) const
{
    assert(r_core.size() == core_n);

    // Output has size core_n
    z_core.resize(core_n);
    z_core.setZero();

    // If LU factorization failed or empty problem, return
    if (ext_n == 0 || !lu_ok) return;

    // r_loc is defined on extended domain but contains r_core only (0 elsewhere)
    Eigen::VectorXd r_loc = Eigen::VectorXd::Zero(ext_n);

    for (int j = 0; j < ext_ny; ++j) {
        for (int i = 0; i < ext_nx; ++i) {
            int gi = ext_i0 + i;  // Global x index
            int gj = ext_j0 + j;  // Global y index

            // Global Dirichlet BCs: at boundaries, impose r=0
            if (gi == 0 || gi == Nx-1 || gj == 0 || gj == Ny-1) continue;

            // If this node falls into the core, then copy r_core element; otherwise write 0.            
            if (gi >= core_i0 && gi <= core_i1 && gj >= core_j0 && gj <= core_j1) {
                int ii = gi - core_i0;
                int jj = gj - core_j0;
                int cid = idlocal(ii, jj, core_nx);
                int eid = idlocal(i, j, ext_nx);
                
                r_loc[eid] = r_core[cid];
            }
        }
    }

    // Local solve: z_loc = A_loc^{-1} r_loc (sparse LU)
    Eigen::VectorXd z_loc = lu.solve(r_loc);

    // Restriction RAS: take only values on core
    for (int j = 0; j < core_ny; ++j)
        for (int i = 0; i < core_nx; ++i) {
            int gi = core_i0 + i;  // Global x index (core)
            int gj = core_j0 + j;  // Global y index (core)

            // Global Dirichlet BCs: skip update of boundary nodes
            if (gi == 0 || gi == Nx-1 || gj == 0 || gj == Ny-1) continue;
            
            int ei = gi - ext_i0;  // Global x index (extended)
            int ej = gj - ext_j0;  // Global y index (extended)
            
            int eid = idlocal(ei, ej, ext_nx);
            int cid = idlocal(i, j, core_nx);
            
            z_core[cid] = z_loc[eid];
        }
    }
}


void LocalProblem::assemble_and_factorize() {
    lu_ok = true;
    
    // Empty local problem
    if (ext_n <= 0) {
    lu_ok = false;
    return;
    }

    // Local sparse matrix assembly
    A_loc.resize(ext_n, ext_n);
    
    const double hx2 = hx * hx;
    const double hy2 = hy * hy;
    double diag_center = mu * (2.0 / hx2 + 2.0 / hy2) + c;
    double diag_off_x = -mu / hx2;
    double diag_off_y = -mu / hy2;

    // Triplet list for sparse matrix construction
    std::vector<Eigen::Triplet<double>> trips;
    trips.reserve(static_cast<std::size_t>(ext_n) * 5);  // 5-point FD stencil

    // Lambda function to add a triplet
    auto add_trip = [&](int r, int cid, double val) {
      trips.emplace_back(static_cast<Eigen::Index>(r), static_cast<Eigen::Index>(cid), val);
    };

    for (int j = 0; j < ext_ny; ++j) {
        for (int i = 0; i < ext_nx; ++i) {
            const int row = idlocal(i, j, ext_nx);
            const int gi = ext_i0 + i;  // Global x index (extended)
            const int gj = ext_j0 + j;  // Global y index (extended)

            // If on global boundary, Dirichlet BC -> A_loc(row,row) = 1.0
            if (gi == 0 || gi == Nx - 1 || gj == 0 || gj == Ny - 1) {
                add_trip(row, row, 1.0);
                continue;
            }

            // Diagonal entry
            add_trip(row, row, diag_center);

            // Neighboring entries (if within extended domain)
            if (gi - 1 >= ext_i0) add_trip(row, idlocal(i - 1, j,     ext_nx), diag_off_x);
            if (gi + 1 <= ext_i1) add_trip(row, idlocal(i + 1, j,     ext_nx), diag_off_x);
            if (gj - 1 >= ext_j0) add_trip(row, idlocal(i,     j - 1, ext_nx), diag_off_y);
            if (gj + 1 <= ext_j1) add_trip(row, idlocal(i,     j + 1, ext_nx), diag_off_y);
        }
    }

    // Build sparse matrix from triplets
    A_loc.resize(static_cast<Eigen::Index>(ext_n), static_cast<Eigen::Index>(ext_n));
    A_loc.setFromTriplets(trips.begin(), trips.end());

    // CSR compression for efficiency
    A_loc.makeCompressed();

    // Sparse LU factorization
    eig_lu.compute(A_loc);

    if (lu.info() != Eigen::Success) {
        lu_ok = false;
    }
}


// ============================================================
// WRAPPER CLASSES FOR GMRES 
// ============================================================
//
// Adapt GMRES requirements to our code

class MatrixWrapper
{
private:
  Solver* solver;
  
public:
  MatrixWrapper(Solver* s) : solver(s) {}
  
  // Operator required by GMRES: A * v
  Eigen::VectorXd operator*(const Eigen::VectorXd& v) const
  {
    assert(v.size() == solver->get_core_n() && "Error: mismatch in sizes in A*v");
    Eigen::VectorXd result(v.size());
    solver->matvec(v, result);
    return result;
  }
};

// Preconditioner wrapper for M^{-1}
class PreconditionerWrapper
{
private:
  Solver* solver;
  
public:
  PreconditionerWrapper(Solver* s) : solver(s) {}
  
  // Method required by GMRES: z = M^{-1} * r
  Eigen::VectorXd solve(const Eigen::VectorXd& r) const
  {
    Eigen::VectorXd z(r.size());
    solver->apply_RAS(r, z);
    return z;
  }
};


// ============================================================
// SOLVER
// ============================================================

Solver::Solver(MPI_Comm cart_comm, int rank_, int size_,
                             int Nx_, int Ny_,
                             int ci_s_, int core_nx_,
                             int cj_s_, int core_ny_,
                             double hx_, double hy_, double mu_, double c_,
                             int left_, int right_, int down_, int up_,
                             LocalProblem* localProb, CoarseSolver *coarseProb)
  : cart(cart_comm),
    rank(rank_), size(size_),
    Nx(Nx_), Ny(Ny_),
    core_i0(ci_s_), core_j0(cj_s_),
    core_nx(core_nx_), core_ny(core_ny_),
    core_n(core_nx_ * core_ny_),
    hx(hx_), hy(hy_), mu(mu_), c(c_),
    left(left_), right(right_), down(down_), up(up_),
    local(localProb),
    coarse(coarseProb)
{
    // Halo = core + 1 layer ghost cells on each side
    halo_nx = core_nx + 2;
    halo_ny = core_ny + 2;
    x_halo = Eigen::VectorXd::Zero(halo_nx * halo_ny);

    // Initialize exchange buffers (core size)
    send_left   = Eigen::VectorXd::Zero(core_ny);
    recv_left   = Eigen::VectorXd::Zero(core_ny);
    send_right  = Eigen::VectorXd::Zero(core_ny);
    recv_right  = Eigen::VectorXd::Zero(core_ny);
    send_bottom = Eigen::VectorXd::Zero(core_nx);
    recv_bottom = Eigen::VectorXd::Zero(core_nx);
    send_top    = Eigen::VectorXd::Zero(core_nx);
    recv_top    = Eigen::VectorXd::Zero(core_nx);

    // Local RHS 
    b_local = Eigen::VectorXd::Zero(core_n);
    
    for (int j = 0; j < core_ny; ++j) {
        for (int i = 0; i < core_nx; ++i) {
            int gi = core_i0 + i;
            int gj = core_j0 + j;
            int lid = idlocal(i, j, core_nx);

            // 0.0 at global Dirichlet boundaries, 1.0 elsewhere
            if (gi == 0 || gi == Nx-1 || gj == 0 || gj == Ny-1)
                b_local[lid] = 0.0;
            else
                b_local[lid] = 1.0;
        }
    }
}


void Solver::apply_RAS(const Eigen::VectorXd& r,Eigen::VectorXd& z) const
{
    // Delegate to LocalProblem
    local->apply_RAS(r, z);
}


void Solver::apply_TwoLevel(const Eigen::VectorXd& r_local, Eigen::VectorXd& z_local)
{
    // 1. Level 1: RAS Fine Correction  (fine level)
    local->apply_RAS(r_local, z_local);

    // 2. Level 2: Coarse Grid Correction
    Eigen::VectorXd e_local_coarse = Eigen::VectorXd::Zero(core_n);
    coarse->solve(r_local, e_local_coarse, core_i0, c0re_j0, core_nx, core_ny, cart);

    // Add coarse correction to RAS result 
    for (int i = 0; i < core_n; ++i) {
            z_local[i] += e_local_coarse[i];
    }
}


double Solver::dot_global(const Eigen::VectorXd& a,
                          const Eigen::VectorXd& b) const
{
    // Optimized local dot product
    double local_dot = a.dot(b);
    
    double global_dot = 0.0;
    MPI_Allreduce(&local_dot, &global_dot, 1, MPI_DOUBLE, MPI_SUM, cart);
    return global_dot;
}


void Solver::matvec(const Eigen::VectorXd& p,
                    Eigen::VectorXd& Ap)
{
    assert(p.size() == core_n && "Error: input vector to matvec has wrong size!");

    Ap.resize(core_n);
    Ap.setZero();
  
    // Implicit Dirichlet BCs, ghosts on MPI_PROC_NULL
    x_halo.setZero();

    // Stencil coefficients to approximate -μ∇²u + cu = f
    const double hx2 = hx * hx;
    const double hy2 = hy * hy;
    const double diag_center = mu * (2.0 / hx2 + 2.0 / hy2) + c;
    const double diag_off_x  = -mu / hx2;
    const double diag_off_y  = -mu / hy2;

    // 1. Copy local vector p into halo center (a larger vector containing a ghost layer)
    for (int j = 0; j < core_ny; ++j)
        for (int i = 0; i < core_nx; ++i)
            x_halo[idlocal(i + 1, j + 1, halo_nx)] = p[idlocal(i, j, core_nx)];

  // 2. Prepare send buffers: extract boundaries of core region to send to neighbors
    for (int j = 0; j < core_ny; ++j) {
        send_left[j]  = p[idlocal(0,         j, core_nx)];    // Left column
        send_right[j] = p[idlocal(core_nx-1, j, core_nx)];    // Right column
    }
    for (int i = 0; i < core_nx; ++i) {
        send_bottom[i] = p[idlocal(i, 0,          core_nx)]; // Bottom row
        send_top[i]    = p[idlocal(i, core_ny-1,  core_nx)];  // Top row
    }

    // 3. Exchange ghost cells with neighbors using MPI_Sendrecv
    MPI_Status status;

    // Simultaneous exchanges:
    //   -  Right send / left recv  -> tag 0
    //   -  Left send  / right recv -> tag 1
    //   -  Up send    / down recv  -> tag 2
    //   -  Down send  / up recv    -> tag 3

    MPI_Sendrecv(send_right.data(), core_ny, MPI_DOUBLE, right, 0,
               recv_left.data(),  core_ny, MPI_DOUBLE, left,  0,
               cart, &status);
    
    MPI_Sendrecv(send_left.data(),  core_ny, MPI_DOUBLE, left,  1,
               recv_right.data(), core_ny, MPI_DOUBLE, right, 1,
               cart, &status);
    
    MPI_Sendrecv(send_top.data(),    core_nx, MPI_DOUBLE, up,   2,
               recv_bottom.data(), core_nx, MPI_DOUBLE, down, 2,
               cart, &status);
    
    MPI_Sendrecv(send_bottom.data(), core_nx, MPI_DOUBLE, down, 3,
               recv_top.data(),    core_nx, MPI_DOUBLE, up,   3,
               cart, &status);

    // 4. Unpack received ghost cells into halo region
    for (int j = 0; j < core_ny; ++j) {
        x_halo[idlocal(0,         j + 1, halo_nx)] = recv_left[j];
        x_halo[idlocal(halo_nx-1, j + 1, halo_nx)] = recv_right[j];
    }
    for (int i = 0; i < core_nx; ++i) {
        x_halo[idlocal(i + 1, 0,         halo_nx)] = recv_bottom[i];
        x_halo[idlocal(i + 1, halo_ny-1, halo_nx)] = recv_top[i];
    }
    
    // 5. Apply stencil to compute Ap = A * p
    for (int j = 0; j < core_ny; ++j) {
        for (int i = 0; i < core_nx; ++i) {
            const int gi = core_i0 + i;
            const int gj = core_j0 + j;
            const int lid = idlocal(i, j, core_nx);
    
            // Global Dirichlet BCs: Ap = p at boundaries
            if (gi == 0 || gi == Nx - 1 || gj == 0 || gj == Ny - 1) {
                Ap[lid] = p[lid];
                continue;
            }

          /*
                   uu
                    |
              ul — uc — ur
                    |
                   ud 
          */
    
          const double uc = x_halo[idlocal(i + 1, j + 1, halo_nx)];
          const double ul = x_halo[idlocal(i,     j + 1, halo_nx)];
          const double ur = x_halo[idlocal(i + 2, j + 1, halo_nx)];
          const double ud = x_halo[idlocal(i + 1, j,     halo_nx)];
          const double uu = x_halo[idlocal(i + 1, j + 2, halo_nx)];
    
          Ap[lid] = diag_center * uc
                  + diag_off_x  * (ul + ur)
                  + diag_off_y  * (ud + uu);
          }
     }
}


void Solver::gather_and_save(const Eigen::VectorXd& x_local) {
    // Rank 0 gathers info + buffer from all ranks and writes solution.csv
    if (rank == 0) {
        // Create global solution vector to contain all local solutions
        std::vector<double> u(static_cast<std::size_t>(Nx * Ny), 0.0);

        // Copy local solution from rank 0
        for (int j = 0; j < core_ny; ++j) 
            for (int i = 0; i < core_nx; ++i) {
                int gi = core_i0 + i;
                int gj = core_j0 + j;
                u[idlocal(gi, gj, Nx)] = x_local[idlocal(i, j, core_nx)];
            }

        // Receive from ranks > 0
        for (int p = 1; p < size; ++p) {
            MPI_Status status;

            // info is a vector containing information about the sending rank's core region:
            //   {core_i0, core_j0, core_nx, core_ny}
            int info[4];
            
            MPI_Recv(info, 4, MPI_INT, p, 300, MPI_COMM_WORLD, &status);
            
            const int cs = info[0];  // Column start (core_i0 of process p)
            const int rs = info[1];  // Row start (core_j0 of process p)
            const int cn = info[2];  // Column size (core_nx of process p)
            const int rn = info[3];  // Row size (core_ny of process p)

            std::vector<double> buf(static_cast<std::size_t>(rn * cn));
            
            MPI_Recv(buf.data(), rn*cn, MPI_DOUBLE, p, 301, MPI_COMM_WORLD, &status);

            // Copy into global solution u
            for (int jj = 0; jj < rn; ++jj)
                for (int ii = 0; ii < cn; ++ii) {
                    int gi = cs + ii;  // Global x index
                    int gj = rs + jj;  // global y index
                    u[idlocal(gi, gj, Nx)] = buf[idlocal(ii,jj,cn)];
                }
            }

        // Rank 0 writes solution.csv
        ofstream ofs("solution.csv");
        ofs << "x,y,u\n";
        for (int j = 0; j < Ny; ++j) {
            for (int i = 0; i < Nx; ++i) {
                double xg = i * (1.0/(Nx - 1));
                double yg = j * (1.0/(Ny - 1));
                ofs << xg << "," << yg << "," << u[idlocal(i,j,Nx)] << "\n";
            }
        }
        ofs.close();
        std::cout << "\nFile solution.csv written by rank 0." << std::endl;
        
    } else {
        // Rank non-zero sends info + buffer to rank0
        int info[4] = { core_i0, core_j0, core_nx, core_ny };
        MPI_Send(info, 4, MPI_INT, 0, 300, MPI_COMM_WORLD);
        MPI_Send(x_local.data(), core_n, MPI_DOUBLE, 0, 301, MPI_COMM_WORLD);
    }
}


void Solver::run(int max_it, double tol, int m_restart)
{
  // Initial guess
  Eigen::VectorXd u = Eigen::VectorXd::Zero(core_n);
  
  // Create wrappers for matrix and preconditioner
  MatrixWrapper A_wrap(this);
  PreconditionerWrapper M_wrap(this);
  int stat = -1;
  
  // GMRES parameters (are modified by GMRES)
  int iterations = max_it;
  double final_tol = tol;
  
  /*
  if (rank == 0) {
    std::cout << "Starting GMRES with:" << std::endl;
    std::cout << "  Max iterations: " << max_it << std::endl;
    std::cout << "  Tolerance: " << tol << std::endl;
    std::cout << "  Restart: " << m_restart << std::endl;
  }
  
  // Call GMRES
  stat = LinearAlgebra::GMRES(
    A_wrap,      // Matrix A, with operator *
    u,           // Solution vector 
    b_local,     // Right-hand side vector
    M_wrap,      // Preconditioner M, with method solve()
    m_restart,   // Restart level
    iterations,  // Maximum number of iterations
    final_tol    // Tolerance
  );
  
  // Print results
  if (rank == 0) {
    if (stat == 0) {
      std::cout << "\nGMRES CONVERGED successfully!" << std::endl;
      std::cout << "  Iterations: " << iterations << std::endl;
      std::cout << "  Final residual: " << final_tol << std::endl;
    } else if (stat == 1) {
      std::cout << "\nGMRES did NOT converge within max iterations" << std::endl;
      std::cout << "  Iterations performed: " << iterations << std::endl;
      std::cout << "  Final residual: " << final_tol << std::endl;
    } else {
      std::cout << "\nGMRES returned with unexpected status: " << stat << std::endl;
    }
  }
  */
  
  if (rank == 0) {
    std::cout << "Starting BiCGSTAB with:" << std::endl;
    std::cout << "  Max iterations: " << max_it << std::endl;
    std::cout << "  Tolerance: " << tol << std::endl;
  }
  
  // Call BiCGSTAB
  stat = LinearAlgebra::BiCGSTAB(
    A_wrap,      // Matrix A, with operator *
    u,           // Solution vector 
    b_local,     // Right-hand side vector
    M_wrap,      // Preconditioner M, with method solve()
    iterations,  // Maximum number of iterations
    final_tol    // Tolerance
  );
  
  // Print results
  if (rank == 0) {
    if (stat == 0) {
      std::cout << "\nBiCGSTAB CONVERGED successfully!" << std::endl;
      std::cout << "  Iterations: " << iterations << std::endl;
      std::cout << "  Final residual: " << final_tol << std::endl;
    } else if (stat == 1) {
      std::cout << "\nBiCGSTAB did NOT converge within max iterations" << std::endl;
      std::cout << "  Iterations performed: " << iterations << std::endl;
      std::cout << "  Final residual: " << final_tol << std::endl;
    } else {
      std::cout << "\nBiCGSTAB returned with unexpected status: " << stat << std::endl;
    }
  }
  

  // Gather and save solution
  gather_and_save(u);

}

     
    
