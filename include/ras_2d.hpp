#pragma once

#include <mpi.h>

#include <Eigen/Sparse>
#include <Eigen/SparseLU>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// Utility function to compute local linear index in a 2D array stored in row-major order
inline int idlocal(int i, int j, int nx) { return i + j * nx; }


// Partitioning utility
class Partition {
public:
  
  // 1D balanced partitioning:
  //   -  N:       total number of nodes
  //   -  nb:      number of blocks/processes
  //   -  proc_id: id of the process [0..nb-1]
  //
  // Outputs:
  //   -  start: starting index of the block
  //   -  len  : length of the block
   
  static void compute_1d_partition(int N, int nb, int proc_id, int& start, int& len);

  static int find_best_coarse_grid(int Nf, int target_ratio);
};


// CoarseSolver
class CoarseSolver {

    int Nx, Ny;    // Fine grid dimensions

    double Lx, Ly;

    int Ncx, Ncy;  // Coarse grid dimensions

    Eigen::SparseMatrix<double> Ac;
    Eigen::SparseLU<Eigen::SparseMatrix<double>> lu_coarse;

    double mu, c;
    int rank;

  public:

    CoarseSolver(int Nx_, int Ny_, double Lx_, double Ly_, int Ncx_, int Ncy_, double mu_, double c_, int rank_);

    void solve(const Eigen::VectorXd& r_local, Eigen::VectorXd& e_local, int ci_s, int cj_s, int core_nx, int core_ny, MPI_Comm cart);
  
  private:

    Eigen::VectorXd rc_local;
    Eigen::VectorXd rc_global;
};


// Assemble and solve local problems on extended subdomains
class LocalProblem {
public:

  // Initialize the local problem:
  //   -  (ext_i0..ext_i1, ext_j0..ext_j1): subdomain extension (inclusive)
  //   -  (ci_s..ci_e, cj_s..cj_e):         core assigned to the rank (inclusive)
  //   -  Nx, Ny:                           global grid sizes
  //   -  hx, hy:                           grid spacings
  //   -  mu, c:                            PDE parameters

  LocalProblem(int ci_s_, int cj_s_, int c_nx_, int c_ny_, int overlap_,
               int Nx_, int Ny_, double hx_, double hy_,
               double mu_, double c_);

  bool is_lu_ok() const;

  // Apply Restricted Additive Schwarz preconditioner to r_core (residual on core region),
  // producing z_core (preconditioned residual on core region).
  //
  // Steps:
  //   1.  Create r_loc on extended domain:
  //         - Dirichlet on global boundaries -> r_loc = 0
  //         - Copy only core values into r_loc, zero elsewhere (Restriction "R" and Prolongation "P")
  //   2.  Solve local problem on extended domain:
  //         - Factorize using LU: x_loc = A_loc^{-1} r_loc
  //         - Restrict solution to core region: z_core = R * x_loc
  
  void apply_RAS(const Eigen::VectorXd& r_core,
                 Eigen::VectorXd& z_core) const;

  // Getters 
  int get_ext_i0() const { return ext_i0; }
  int get_ext_i1() const { return ext_i1; }
  int get_ext_j0() const { return ext_j0; }
  int get_ext_j1() const { return ext_j1; }
  int get_ext_nx() const { return ext_nx; }
  int get_ext_ny() const { return ext_ny; }
  int get_ext_n()  const { return ext_n;  }

  int get_core_i0() const { return core_i0; }
  int get_core_i1() const { return core_i1; }
  int get_core_j0() const { return core_j0; }
  int get_core_j1() const { return core_j1; }
  int get_core_nx() const { return core_nx; }
  int get_core_ny() const { return core_ny; }
  int get_core_n()  const { return core_n;  }

private:

  // Assemble and factorize the local extended matrix
  void assemble_and_factorize();

private:

  // Geometry: extended box subdomain (inclusive)
  int ext_i0, ext_i1, ext_j0, ext_j1;
  int ext_nx, ext_ny, ext_n;

  // Geometry: core box subdomain (inclusive)
  int core_i0, core_i1, core_j0, core_j1;
  int core_nx, core_ny, core_n;

  int overlap;

  // Global sized
  int Nx, Ny;

  // PDE diffusion-reaction parameters and grid spacings
  double hx, hy, mu, c;

  // Sparse local operator and its LU factorization
  Eigen::SparseMatrix<double> A_loc;
  Eigen::SparseLU<Eigen::SparseMatrix<double>> lu;
  bool lu_ok;
};


// Solver implementing GMRES with RAS preconditioning

class Solver {

  // Declare wrappers as friends so that they can access to private methods
  friend class MatrixWrapper;
  friend class PreconditionerWrapper;

public:
  Solver(MPI_Comm cart_comm, int rank_, int size_,
                int Nx_, int Ny_,
                int ci_s_, int core_nx_,
                int cj_s_, int core_ny_,
                double hx_, double hy_, double mu_, double c_,
                int left_, int right_, int down_, int up_,
                LocalProblem* localProb,
                CoarseSolver *coarseProb );

  // Ownership of LocalProblem is not transferred
  ~Solver() = default;

  // Restartable GMRES (m_restart) with left preconditioning: M^{-1} A x = M^{-1} b
  // 
  // Steps:
  //   1.  Construct rhs_pre = M^{-1} b
  //   2.  Initialize x=0 and residual r = rhs_pre - M^{-1} A x
  //   3.  Restart loop until relres <= tol, or total_iters >= max_it
  //   4.  At the end call gather_and_save(x)
 
  void run(int max_it, double tol, int m_restart, const double hx, const double hy);

  // Getter
  int get_core_n() const { return core_n; }

private:

  // Global dot product with MPI_Allreduce
  double dot_global(const Eigen::VectorXd& a,
                    const Eigen::VectorXd& b) const;

  // Distributed MatVec product: Ap = A * p
  void matvec(const Eigen::VectorXd& p,
              Eigen::VectorXd& Ap);

  // Apply the preconditioner (delegated to LocalProblem)
  void apply_RAS(const Eigen::VectorXd& r, Eigen::VectorXd& z) const;

  // Gather final solution: rank 0 receives (info + buffer) and writes solution.csv
  void gather_and_save(const Eigen::VectorXd& x_local, const double hx, const double hy);

  // Two-Level preconditioner
  void apply_TwoLevel(const Eigen::VectorXd& r_local, Eigen::VectorXd& z_local);

private:
  MPI_Comm cart;
  int rank, size;

  int Nx, Ny;
  int core_i0, core_j0;
  int core_nx, core_ny, core_n;

  double hx, hy, mu, c;

  // Neighbors
  int left, right, down, up;

  // Non-owning
  LocalProblem* local;
  CoarseSolver* coarse;

  // Halo storage (core + ghost layer 1 cell)
  int halo_nx, halo_ny;
  Eigen::VectorXd x_halo;

  // Send/receive buffers
  Eigen::VectorXd send_left, recv_left;
  Eigen::VectorXd send_right, recv_right;
  Eigen::VectorXd send_bottom, recv_bottom;
  Eigen::VectorXd send_top, recv_top;

  // Local RHS (core)
  Eigen::VectorXd b_local;
};

