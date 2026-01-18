#pragma once

#include <mpi.h>

#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>

#include <array>
#include <vector>
#include <cmath>
#include <iostream>
#include <string>
#include <fstream>
#include <iomanip>
#include <cassert>
#include <functional>

namespace schwarz {

inline std::size_t my_pow(std::size_t a, int p) {
    std::size_t r = 1;
    for (int i = 0; i < p; ++i) r *= a;
    return r;
}

// ======================================================
// ==================== LOCAL PROBLEM ===================
// ======================================================
//
// Local solver used by each MPI process to compute the 
// solution in its own subdomain, coordinating with others
// through overlap value exchanges.
// 
// Parameters:
//   -  N_global:     number of nodes per dimension (global)
//   -  core_start:   initial indices of the core (non-overlapped) region for this process
//   -  core_end:     final indices of the core (non-overlapped) region for this process
//   -  overlap_l:    number of overlapping nodes with neighboring subdomains
//   -  mu_:          diffusion coefficient in the equation
//   -  c_:           reaction coefficient in the equation
//   -  a_:           left boundary of the physical domain
//   -  b_:           right boundary of the physical domain
//   -  ua_:          Dirichlet value at left boundary (x=a)
//   -  ub_:          Dirichlet value at right boundary (x=b)
//   -  forcing_func: forcing function f(x,y,z)

template <int Dim>
class LocalProblem {
public:
    static_assert(Dim == 1 || Dim == 2 || Dim == 3, "Dim must be 1, 2 or 3");

    using Coord = std::array<double, Dim>;
    using Forcing = std::function<double(const Coord&)>;
    using Index = std::array<int, Dim>;

    LocalProblem(int N_global,
                 const std::array<int, Dim>& core_s, 
                 const std::array<int, Dim>& core_e,
                 int overlap_l,
                 double mu_, double c_,
                 double a_, double b_,
                 double ua_, double ub_,
                 Forcing forcing_func);

    // Build local sparse matrix for this subdomain
    void build_matrix();

    // Assemble RHS from forcing then apply interface BC
    void assemble_rhs();

    // Apply interface Dirichlet values on extended x-boundaries
    // bc_left_plane / bc_right_plane size = plane_size
    void apply_dirichlet(const std::vector<double>& bc_left_plane,
                         const std::vector<double>& bc_right_plane);

    // Solve local sparse system
    void solve();

    // METHODS FOR DATA EXCHANGE

    // Extract data of one face (front = true -> front face, false -> back face)
    std::vector<double> extract_face(int dim, bool front) const;

    // Update values in overlpap region with received data
    void update_overlap(int dim, bool front, const std::vector<double>& data);

    // Save old solution to check convergence
    void save_old();
    double local_error_sqr() const;

    // Useful for save phase
    const std::array<int, Dim>& get_core_start() const { return core_start; }
    const std::array<int, Dim>& get_core_end() const { return core_end; }
    const std::array<int, Dim>& get_core_count() const { return core_count; }
    double value_at_local_index(int i) const { return u[i]; }

    // Helpers 
    int lidx(const Index& p) const;  // Map local indices to local linear index
    Index unflatten(int idx) const;  // Map local linear index to local indices (relative to ext_start)
    bool is_in_overlap(const Index& global_p) const; 
    bool is_on_global_boundary(const Index& global_p) const;
    Coord node_to_coords(const Index& global_p) const;

    void write_local_vti(const std::string& filename, int rank) const;

private:
    int N;                 // Global number of nodes per dimension

    // Core (non-overlapped) region info
    std::array<int, Dim> core_start;     // Global start indices of core region
    std::array<int, Dim> core_end;       // Global end indices of core region
    std::array<int, Dim> core_count;     // Number of nodes in core region

    // Extended (overlapped) region info
    std::array<int, Dim> ext_start;      // Global start indices of extended region
    std::array<int, Dim> ext_end;        // Global end indices of extended region
    std::array<int, Dim> ext_count;      // Number of nodes in extended region
    
    int overlap;
    int local_dofs;  // Number of local DOFs (extended region)

    double mu, c;
    double a, b, h;
    double ua, ub;

    Forcing forcing;

    Eigen::SparseMatrix<double, Eigen::RowMajor> A;
    Eigen::VectorXd rhs;
    Eigen::VectorXd u;
    Eigen::VectorXd u_old;

    bool matrix_built = false;
};

// ======================================================
// ================= SCHWARZ ITERATIONS =================
// ======================================================
//
// Each MPI process manages a LocalProblem that has a "core" 
// region (its own part, non-overlapping) and an an "overlap" 
// region that extends into neighboring subdomains.
// The method iteratively solves local problems on each 
// extended subdomain, exchanging boundary values between
// neighboring processes until global convergence.
// 
// Parameters:
//   -  N_per_dim:    number of nodes per dimension (global)
//   -  mpi_rank:     rank of the current process
//   -  mpi_size:     total number of processes
//   -  overlap_l:    number of overlapping nodes
//   -  mu_, c_, a_, b_, ua_, ub_: PDE parameters
//   -  max_iter_:    maximum number of Schwarz iterations
//   -  tol_:         convergence tolerance
//   -  forcing_func: forcing function f(x,y,z)

template <int Dim>
class SchwarzSolver {
public:
    static_assert(Dim==1 || Dim == 2 || Dim == 3, "Dim must be 1, 2 or 3");

    using Coord = std::array<double, Dim>;
    using Forcing = std::function<double(const Coord&)>;

    SchwarzSolver(int N_per_dim,
                 int mpi_rank, int mpi_size,
                 int overlap_l, double mu, double c,
                 double a, double b,
                 double ua, double ub,
                 int max_iter, double tol,
                 Forcing forcing_func);

    ~SchwarzSolver();

    void run();

private:
    int N;          

    int rank, size;
    int overlap;

    double mu, c;
    double a, b, h;
    double ua, ub;

    int max_iter;
    double tol;

    std::array<int, Dim> core_start;
    std::array<int, Dim> core_end;

    std::vector<int> neighbors;          // Size 2*Dim: [x_neg, x_pos, y_neg, y_pos, z_neg, z_pos]
    std::array<int, Dim> proc_coords;    // Process coordinates in the grid
    std::array<int, Dim> dims_topology;  // Number of processes in each dimension

    LocalProblem<Dim>* local;
    Forcing forcing;

    void gather_and_save(const std::string& base_name = "solution");
};


// Explicit instantiations (defined in schwarz_solver.cpp)

extern template class LocalProblem<1>;
extern template class LocalProblem<2>;
extern template class LocalProblem<3>;

extern template class SchwarzSolver<1>;
extern template class SchwarzSolver<2>;
extern template class SchwarzSolver<3>;

} // namespace schwarz

