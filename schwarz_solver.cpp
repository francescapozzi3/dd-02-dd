#include "schwarz_solver.hpp"

#include <numeric>
#include <algorithm>
#include <iomanip>

namespace schwarz {

// ======================================================
// ==================== LOCAL PROBLEM ===================
// ======================================================

template <int Dim>
LocalProblem<Dim>::LocalProblem(int N_global,
                                const std::array<int, Dim>& core_s,
                                const std::array<int, Dim>& core_e,
                                int overlap_l,
                                double mu_, double c_,
                                double a_, double b_,
                                double ua_, double ub_,
                                Forcing forcing_func)
    : N(N_global),
      core_start(core_s),
      core_end(core_e),
      overlap(overlap_l),
      mu(mu_),
      c(c_),
      a(a_),
      b(b_),
      ua(ua_),
      ub(ub_),
      forcing(std::move(forcing_func))  // std::move to avoid replication
                                        // if possible (efficency)
{
    h = (b - a) / (N - 1);

    // Define extended region (with overlap): Core ± overlap, clamped to [0, N-1]
    local_dofs = 1;
    for (int d = 0; d < Dim; ++d) {
        core_count[d] = core_end[d] - core_start[d] + 1;

        ext_start[d] = std::max(0, core_start[d] - overlap);
        ext_end[d] = std::min(N - 1, core_end[d] + overlap);

        ext_count[d] = ext_end[d] - ext_start[d] + 1;

        local_dofs *= ext_count[d];
    }

    // Preallocate RHS and solution vectors
    u.setZero(local_dofs);
    u_old.setZero(local_dofs);
    rhs.setZero(local_dofs);
}


// Convert local multi-dimensional indices to local linear index

template <int Dim>
int LocalProblem<Dim>::lidx(const Index& p) const {
    int idx = 0;
    int stride = 1;
    for (int i = 0; i < Dim; ++i) {
        idx += (p[i] - ext_start[i]) * stride;
        stride *= ext_count[i];
    }
    return idx;
}


// Inverse of lidx (linear -> multi-dimensional)
template <int Dim>
std::array<int, Dim> LocalProblem<Dim>::unflatten(int idx) const {
    std::array<int, Dim> p;
    for (int i = 0; i < Dim; ++i) {
        p[i] = (idx % ext_count[i]) + ext_start[i];
        idx /= ext_count[i];
    }
    return p;
}


template <int Dim>
bool LocalProblem<Dim>::is_in_overlap(const std::array<int, Dim>& p) const {
    for (int d = 0; d < Dim; ++d) {
        if (p[d] < core_start[d] || p[d] > core_end[d]) return true;
    }
    return false;
}


template <int Dim>
bool LocalProblem<Dim>::is_on_global_boundary(const std::array<int, Dim>& p) const {
    for (int d = 0; d < Dim; ++d) {
        if (p[d] == 0 || p[d] == N - 1) return true;
    }
    return false;
}


template <int Dim>
std::array<double, Dim> LocalProblem<Dim>::node_to_coords(const std::array<int, Dim>& p) const {
    std::array<double, Dim> coords;
    for (int d = 0; d < Dim; ++d) {
        coords[d] = a + p[d] * h;
    }
    return coords;
}


// Build the local sparse matrix A for the local problem on the extended domain
// and solve -mu * Δu + c * u = f with Dirichlet BC on global domain boundaries.
//
// Finite differences scheme on uniform grid with mesh size h:
//   -  1D stencil: 3-point FD
//   -  2D stencil: 5-point FD
//   -  3D stencil: 7-point FD

template <int Dim>
void LocalProblem<Dim>::build_matrix() {
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(local_dofs * (2 * Dim + 1));  // Approximate number of non-zeros
    const double h2 = h * h;

    for (int i = 0; i < local_dofs; ++i) {
        auto p = unflatten(i);  // Multi-dimensional indices

        // Global Dirichlet boundary conditions
        if (is_on_global_boundary(p)) {
            triplets.push_back(Eigen::Triplet<double>(i, i, 1.0));  // u = ua or ub
            continue;
        }

        // Interior boundary conditions (overlap region)
        // If the node is outside of core region, its value will be updated from neighbors.
        if (is_in_overlap(p)) {
            triplets.push_back(Eigen::Triplet<double>(i, i, 1.0));  // u = value from neighbor
            continue;
        }

        // Interior core region: build FD stencil
        double diag = c + 2.0 * Dim * mu / h2;  // Diagonal entry
        triplets.push_back(Eigen::Triplet<double>(i, i, diag));

        // Off-diagonal entries for each dimension
        for (int d = 0; d < Dim; ++d) {
            // Negative direction
            auto p_back = p;
            p_back[d] -= 1;
            triplets.push_back(Eigen::Triplet<double>(i, lidx(p_back), -mu / h2));

            // Positive direction
            auto p_front = p;
            p_front[d] += 1;
            triplets.push_back(Eigen::Triplet<double>(i, lidx(p_front), -mu / h2));
        }
    }

    A.resize(local_dofs, local_dofs);
    A.setFromTriplets(triplets.begin(), triplets.end());
    A.makeCompressed();
    matrix_built = true;
}


// Assemble the RHS vector from the forcing function on the extended domain

template <int Dim>
void LocalProblem<Dim>::assemble_rhs() {
    rhs.setZero();

    for (int i = 0; i < local_dofs; ++i) {
        auto p = unflatten(i);  // Multi-dimensional indices
        
        if (is_on_global_boundary(p)) {
            if (p[0] == 0) {
                // Left global boundary
                rhs(i) = ua;
            } else if (p[0] == N - 1) {
                // Right global boundary
                rhs(i) = ub;
            } else {
                // Other global boundaries (if any)
                rhs(i) = 0.0;  // Homogeneous Dirichlet for other boundaries
            }
        } else if (is_in_overlap(p)) {
            // Use current u
            rhs(i) = u(i);
        } else {
            // Interior core region: evaluate forcing function
            rhs(i) = forcing(node_to_coords(p));
        }
    }
}


// Solve the local linear system A * u = rhs using preconditioned BiCGSTAB method

template <int Dim>
void LocalProblem<Dim>::solve() {
    if (!matrix_built) build_matrix();

    // Assemble RHS using current u (containing updated overlap values)
    assemble_rhs();

    // Solver setup and solve
    Eigen::BiCGSTAB<Eigen::SparseMatrix<double>, Eigen::DiagonalPreconditioner<double>> solver;

    solver.compute(A);
    u = solver.solve(rhs);
}


// Extract the face data to send to neighboring subdomains in dimension 'dim'.
// If front=true, extract the face at the "end" of the core region in that
// dimension; otherwise, extract the face at the "start" of the core region.

template <int Dim>
std::vector<double> LocalProblem<Dim>::extract_face(int dim, bool front) const {
    std::vector<double> face_data;
    
    // Define the range of iteration
    std::array<int, Dim> start = ext_start;
    std::array<int, Dim> end = ext_end;
    
    // If front=true (send to right), take the end of my core
    // If front=false (send to left), take the start of my core
    int fixed_idx = front ? core_end[dim] : core_start[dim];

    // Run through all local DOFs and extract those on the specified face
    for (int i = 0; i < local_dofs; ++i) {
        auto p = unflatten(i);
        
        // If the coordinate in the exchange dimension matches the boundary
        if (p[dim] == fixed_idx) {
            // Ensure the point is within the core region in other dimensions
            bool pure_core = true;
            for(int d = 0; d < Dim; ++d) {
                if (d != dim) {
                   if (p[d] < core_start[d] || p[d] > core_end[d]) pure_core = false;
                }
            }
            if (pure_core) face_data.push_back(u(i));
        }
    }
    return face_data;
}


// Insert received face data from neighboring subdomains into the overlap region

template <int Dim>
void LocalProblem<Dim>::update_overlap(int dim, bool front, const std::vector<double>& data) {
    // Index where to write the received data:
    //   -  If it's overlap FRONT (front=true), it's core_end + 1
    //   -  If it's overlap BACK (front=false), it's core_start - 1

    // Target index in dimension 'dim' to write data
    int target_idx = front ? (core_end[dim] + 1) : (core_start[dim] - 1);
    
    // Check if target_idx is within extended region
    if (target_idx < ext_start[dim] || target_idx > ext_end[dim]) return;  // No overlap to update

    int counter = 0;
    for (int i = 0; i < local_dofs; ++i) {
        auto p = unflatten(i);
        if (p[dim] == target_idx) {
            // Verify that the point is within the core region in other dimensions
            bool pure_core_others = true;
            for(int d = 0; d < Dim; ++d) {
                if (d != dim) {
                    // Must write only in the part that corresponds to the core of the other axes
                    if (p[d] < core_start[d] || p[d] > core_end[d]) pure_core_others = false;
                }
            }
            if (pure_core_others && counter < data.size()) {
                u(i) = data[counter++];
            }
        }
    }
}

template <int Dim>
void LocalProblem<Dim>::save_old() { u_old = u; }

template <int Dim>
double LocalProblem<Dim>::local_error_sqr() const {
    return (u - u_old).squaredNorm();
}



// ======================================================
// ================= SCHWARZ ITERATIONS =================
// ======================================================

template <int Dim>
SchwarzSolver<Dim>::SchwarzSolver(int N_per_dim,
                                  int mpi_rank, int mpi_size,
                                  int overlap_l, double mu_, double c_,
                                  double a_, double b_,
                                  double ua_, double ub_,
                                  int max_iter_, double tol_,
                                  Forcing forcing_func)
    : N(N_per_dim),
      rank(mpi_rank),
      size(mpi_size),
      overlap(overlap_l),
      mu(mu_),
      c(c_),
      a(a_),
      b(b_),
      ua(ua_),
      ub(ub_),
      max_iter(max_iter_),
      tol(tol_),
      forcing(std::move(forcing_func))
{
    h = (b - a) / (N - 1);

    // Define process topology in Dim dimensions (e.g. for 8 proc in 3D -> 2x2x2)
    if constexpr (Dim == 1) {
        dims_topology = {size};
    } else if constexpr (Dim == 2) {
        // Find most square-like factors of size
        int best_i = 1;
        int best_diff = size;
        for (int i = 1; i * i <= size; ++i) {
            if (size % i == 0) {
                int j = size / i;
                if (std::abs(i - j) < best_diff) {
                    best_i = i;
                    best_diff = std::abs(i - j);
                }
            }
        }
        dims_topology = {best_i, size / best_i};
    }
    else {
        // Find most cubic-like factors of size

        // Find factors i, j, k such that i * j * k = size and minimize |i - j| + |j - k| + |k - i|
        int best_i = 1, best_j = 1, best_k = size;
        int best_diff = std::abs(1 - 1) + std::abs(1 - size) + std::abs(size - 1);
        
        for (int i = 1; i * i * i <= size; ++i) {
            if (size % i == 0) {  // If i is a factor
                int remaining = size / i;
                for (int j = i; j * j <= remaining; ++j) {
                    if (remaining % j == 0) {  // If j is a factor
                        int k = remaining / j;
                        // Compute how "balanced" the factors are
                        int diff = std::abs(i - j) + std::abs(j - k) + std::abs(k - i);
                        // Update best factors if more balanced
                        if (diff < best_diff) {
                            best_i = i;
                            best_j = j;
                            best_k = k;
                            best_diff = diff;
                        }
                    }
                }
            }
        }
        dims_topology = {best_i, best_j, best_k};
    }

    // Print process topology for debugging
    if (rank == 0) {
        std::cout << "Domain decomposition: ";
        for (int d = 0; d < Dim; ++d) {
            std::cout << dims_topology[d];
            if (d < Dim - 1) std::cout << " x ";
        }
        std::cout << " = " << size << " processes" << std::endl;
    }

    // Determine process coordinates in the topology grid
    int r = rank;
    for (int i = 0; i < Dim; ++i) {
        proc_coords[i] = r % dims_topology[i];
        r /= dims_topology[i];
    }

    // Determine core limits for each dimension
    for (int i = 0; i < Dim; ++i) {
        int base = N / dims_topology[i];
        int rem = N % dims_topology[i];
        core_start[i] = proc_coords[i] * base + std::min(proc_coords[i], rem);
        int count = base + (proc_coords[i] < rem ? 1 : 0);
        core_end[i] = core_start[i] + count - 1;
    }

    // Identify neighbors (use MPI_PROC_NULL if no neighbor in that direction)
    neighbors.assign(2 * Dim, MPI_PROC_NULL);
    for (int d = 0; d < Dim; ++d){
        // Negative direction (e.g. left in x, down in y, back in z)
        auto get_rank = [&](const std::array<int, Dim>& coords) {
            int r = 0;
            int stride = 1;
            for (int i = 0; i < Dim; ++i) {
                r += coords[i] * stride;
                stride *= dims_topology[i];
            }
            return r;
        };

        // Negative neighbor
        if (proc_coords[d] > 0) {
            auto neighbor_coords = proc_coords;
            neighbor_coords[d] -= 1;
            neighbors[2 * d] = get_rank(neighbor_coords);
        }

        // Positive neighbor
        if (proc_coords[d] < dims_topology[d] - 1) {
            auto neighbor_coords = proc_coords;
            neighbor_coords[d] += 1;
            neighbors[2 * d + 1] = get_rank(neighbor_coords);
        }
    }

    // Create local problem instance
    local = new LocalProblem<Dim>(N, core_start, core_end,
                                 overlap, mu, c,
                                 a, b,
                                 ua, ub,
                                 forcing);
}

template <int Dim>
SchwarzSolver<Dim>::~SchwarzSolver() {
    delete local;
}


// Run the Schwarz iterative solver until convergence or maximum iterations reached.

template <int Dim>
void SchwarzSolver<Dim>::run() {
    // Initial local solve
    local->solve();

    int tag = 0;
    MPI_Status status;
    double global_err = 1e10;
    int iter = 0;

    while (iter < max_iter && global_err > tol) {
        local->save_old();

        // MULTI-DIMENSIONAL DATA EXCHANGE
        for (int d = 0; d < Dim; ++d) {
            // 1. Send BACK, receive FRONT (to update my FRONT overlap)
            // If I have a FRONT neighbor (neighbors[2*d+1]), it will send me data.
            // I send to the neighbor behind (neighbors[2*d]).
            
            // Send to Dest (Back), Recv from Source (Front)
            std::vector<double> send_back = local->extract_face(d, false);  // Extract BACK face of core
            std::vector<double> recv_front(send_back.size());

            MPI_Sendrecv(send_back.data(), (int)send_back.size(), MPI_DOUBLE, neighbors[2*d], tag,
                         recv_front.data(), (int)recv_front.size(), MPI_DOUBLE, neighbors[2*d+1], tag,
                         MPI_COMM_WORLD, &status);
            
            if (neighbors[2*d+1] != MPI_PROC_NULL) {
                local->update_overlap(d, true, recv_front);
            }

            // 2. Send FRONT, receive BACK
            std::vector<double> send_front = local->extract_face(d, true);
            std::vector<double> recv_back(send_front.size());

            MPI_Sendrecv(send_front.data(), (int)send_front.size(), MPI_DOUBLE, neighbors[2*d+1], tag,
                         recv_back.data(), (int)recv_back.size(), MPI_DOUBLE, neighbors[2*d], tag,
                         MPI_COMM_WORLD, &status);

            if (neighbors[2*d] != MPI_PROC_NULL) {
                local->update_overlap(d, false, recv_back);
            }
        }

        // Solve local problem with updated overlap values
        local->solve();

        // Calculate Error
        double loc_err = local->local_error_sqr();
        MPI_Allreduce(&loc_err, &global_err, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        global_err = std::sqrt(global_err);

        if (rank == 0 && iter % 10 == 0) {
            std::cout << "Iteration: " << iter << ", Error: " << global_err << std::endl;
        }
        iter++;
    }

    if (rank == 0) std::cout << "  Final Iteration: " << iter << ", Error: " << global_err << std::endl;
    
    gather_and_save();
}


// Write the local solution to a VTI file for visualization

template <int Dim>
void LocalProblem<Dim>::write_local_vti(const std::string& filename, int rank) const {
    std::ofstream ofs(filename);
    if (!ofs) {
        std::cerr << "ERROR: Cannot open file " << filename << std::endl;
        return;
    }

    // Compute global bounds of the core region
    // VTK uses WholeExtent = [x_min, x_max, y_min, y_max, z_min, z_max]
    int core_bounds[6] = {0, 0, 0, 0, 0, 0};
    
    // Map active dimensions to VTK extent format
    for(int d=0; d<Dim; ++d) {
        core_bounds[2*d]     = core_start[d];
        core_bounds[2*d + 1] = core_end[d];
    }

    ofs << "<?xml version=\"1.0\"?>\n";
    // type="ImageData" for structured uniform grids
    ofs << "<VTKFile type=\"ImageData\" version=\"1.0\" byte_order=\"LittleEndian\">\n";
    
    // WholeExtent here refers to the current piece in this file
    ofs << "  <ImageData WholeExtent=\"" 
        << core_bounds[0] << " " << core_bounds[1] << " "
        << core_bounds[2] << " " << core_bounds[3] << " "
        << core_bounds[4] << " " << core_bounds[5] << "\" "
        << "Origin=\"" << a << " " << (Dim>1?a:0) << " " << (Dim>2?a:0) << "\" "
        << "Spacing=\"" << h << " " << (Dim>1?h:1.0) << " " << (Dim>2?h:1.0) << "\">\n";

    ofs << "    <Piece Extent=\"" 
        << core_bounds[0] << " " << core_bounds[1] << " "
        << core_bounds[2] << " " << core_bounds[3] << " "
        << core_bounds[4] << " " << core_bounds[5] << "\">\n";

    ofs << "      <PointData Scalars=\"u\">\n";
    ofs << "        <DataArray type=\"Float64\" Name=\"u\" format=\"ascii\">\n";

    // Iterate over core region and write values ordered by z, y, x (row-major w.r.t. axes)
    
    // Setup loop bounds (default 0 to 0 for unused dimensions)
    int z_s = (Dim > 2) ? core_start[2] : 0;
    int z_e = (Dim > 2) ? core_end[2]   : 0;
    int y_s = (Dim > 1) ? core_start[1] : 0;
    int y_e = (Dim > 1) ? core_end[1]   : 0;
    int x_s = core_start[0];
    int x_e = core_end[0];

    // Counter for formatting
    int count = 0; 
    
    // Nested loops over dimensions (if unused, outer loops run once)
    for (int k = z_s; k <= z_e; ++k) {
        for (int j = y_s; j <= y_e; ++j) {
            for (int i = x_s; i <= x_e; ++i) {
                // Construct local coordinate to access u
                std::array<int, Dim> p;
                p[0] = i; 
                if constexpr (Dim > 1) p[1] = j; 
                if constexpr (Dim > 2) p[2] = k;
                
                // Use lidx to get the value from vector 'u'
                ofs << std::scientific << std::setprecision(16) << u(lidx(p)) << " ";
                
                // Format: 6 values per line
                if (++count % 6 == 0) ofs << "\n";
            }
        }
    }

    ofs << "\n";
    ofs << "        </DataArray>\n";
    ofs << "      </PointData>\n";
    ofs << "    </Piece>\n";
    ofs << "  </ImageData>\n";
    ofs << "</VTKFile>\n";
    
    ofs.close();
}


// Gather local solutions from all processes and save the global solution to a PVTI file.

template <int Dim>
void SchwarzSolver<Dim>::gather_and_save(const std::string& base_name) {
    // 1. Each process writes its own .vti
    std::string my_filename = base_name + "_" + std::to_string(rank) + ".vti";
    local->write_local_vti(my_filename, rank);
    
    // 2. Prepare a buffer with my core bounds (start and end for each dimension)
    std::vector<int> my_bounds(Dim * 2);
    auto c_start = local->get_core_start();
    auto c_end = local->get_core_end(); 
    
    for(int d = 0; d < Dim; ++d) {
        my_bounds[2*d]     = c_start[d];
        my_bounds[2*d + 1] = c_end[d];
    }

    // 3. Rank 0 collects the bounds from all processes
    std::vector<int> all_bounds;
    if (rank == 0) {
        all_bounds.resize(size * Dim * 2);
    }

    MPI_Gather(my_bounds.data(), Dim * 2, MPI_INT,
               all_bounds.data(), Dim * 2, MPI_INT,
               0, MPI_COMM_WORLD);

    // 4. Write the master .pvti file
    if (rank == 0) {
        std::string pvti_filename = base_name + ".pvti";
        std::ofstream ofs(pvti_filename);
        
        if (!ofs) {
            std::cerr << "ERROR: Cannot create PVTI file " << pvti_filename << std::endl;
            return;
        }

        // Global extent of the domain
        int glob_bounds[6] = {0, N-1, 0, (Dim>1 ? N-1 : 0), 0, (Dim>2 ? N-1 : 0)};

        ofs << "<?xml version=\"1.0\"?>\n";
        ofs << "<VTKFile type=\"PImageData\" version=\"1.0\" byte_order=\"LittleEndian\">\n";
        ofs << "  <PImageData WholeExtent=\"" 
            << glob_bounds[0] << " " << glob_bounds[1] << " "
            << glob_bounds[2] << " " << glob_bounds[3] << " "
            << glob_bounds[4] << " " << glob_bounds[5] << "\" "
            << "GhostLevel=\"0\" " // GhostLevel 0 because we save only core regions
            << "Origin=\"" << a << " " << (Dim>1?a:0) << " " << (Dim>2?a:0) << "\" "
            << "Spacing=\"" << h << " " << (Dim>1?h:1.0) << " " << (Dim>2?h:1.0) << "\">\n";
            
        ofs << "    <PPointData Scalars=\"u\">\n";
        ofs << "      <PDataArray type=\"Float64\" Name=\"u\"/>\n";
        ofs << "    </PPointData>\n";

        // Extract base filename without path
        size_t last_slash = base_name.find_last_of("/\\");
        std::string base_filename = (last_slash != std::string::npos) 
                                    ? base_name.substr(last_slash + 1) 
                                    : base_name;

        // Link all pieces (files written by individual ranks)
        for (int r = 0; r < size; ++r) {
            // Extract the bounds received for rank 'r'
            int r_bounds[6] = {0, 0, 0, 0, 0, 0};
            for(int d = 0; d < Dim; ++d) {
                r_bounds[2*d]     = all_bounds[r*(Dim*2) + 2*d];
                r_bounds[2*d + 1] = all_bounds[r*(Dim*2) + 2*d + 1];
            }

            std::string piece_filename = base_filename + "_" + std::to_string(r) + ".vti";

            ofs << "    <Piece Extent=\"" 
                << r_bounds[0] << " " << r_bounds[1] << " "
                << r_bounds[2] << " " << r_bounds[3] << " "
                << r_bounds[4] << " " << r_bounds[5] << "\" "
                << "Source=\"" << piece_filename << "\"/>\n";
        }

        ofs << "  </PImageData>\n";
        ofs << "</VTKFile>\n";
        
        std::cout << "Saved PVTI solution to " << pvti_filename << std::endl;
        std::cout << "  - Referenced " << size << " VTI piece files" << std::endl;
        ofs.close();
    }
    MPI_Barrier(MPI_COMM_WORLD);
}


// Explicit template instantiations

template class LocalProblem<1>;
template class LocalProblem<2>;
template class LocalProblem<3>;

template class SchwarzSolver<1>;
template class SchwarzSolver<2>;
template class SchwarzSolver<3>;

} // namespace schwarz
