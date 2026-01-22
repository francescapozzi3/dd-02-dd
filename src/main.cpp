#include "ras_2d.hpp"

#include <algorithm>
#include <vector>
#include <iostream>
#include <cmath>
#include <chrono>


// ======================================================================
// INPUT UTILITY
// ======================================================================

template <typename T>
void ask_param(const std::string& msg, T& value) {
    std::cout << msg << " [" << value << "]: ";
    std::string line;
    std::getline(std::cin, line);  // to read a line of text from input stream 

    // Default
    if (line.empty() || line == "." || line == "-") return; 

    // Convert the user’s input string into a number of type T using a stringstream, 
    // and only update the variable if the conversion succeeded
    std::stringstream ss(line);
    T tmp;
    if (ss >> tmp)
        value = tmp;
}


// ======================================================================
// MAIN PROGRAM
// ======================================================================

int main(int argc, char** argv) {

  // Initialize MPI communication
  MPI_Init(&argc, &argv);

  int rank = 0, size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Global grid
  int Nx = 100, Ny = 100;

  // Physical domain [0, 1] x [0, 1]
  double Lx = 1.0;
  double Ly = 1.0;
  const double hx = Lx / (Nx - 1);
  const double hy = Ly / (Ny - 1);

  // PDE coefficients: -mu * Lapl(u) + c * u = f
  double mu = 1.0;  // Diffusion
  double c = 0.1;   // Reaction

  // Solver parameters
  int overlap = 2;  // Overlap size in number of nodes
  int max_it = 2000;
  double tol = 1e-06;
  int restart = 50;

  // Parsing arguments
  if (argc >= 3) { Nx      = std::stoi(argv[1]);   Ny = std::stoi(argv[2]); }
  if (argc >= 5) { Lx      = std::stoi(argv[3]);   Ly = std::stoi(argv[4]); }
  if (argc >= 6)   mu      = std::stod(argv[5]);
  if (argc >= 7)   c       = std::stod(argv[6]);
  if (argc >= 8)   overlap = std::stoi(argv[7]);
  if (argc >= 9)   max_it  = std::stoi(argv[8]);
  if (argc >= 10)  tol     = std::stod(argv[9]);
  if (argc >= 11)  restart = std::stoi(argv[10]);
  if (argc >= 13) { Ncx    = std::stoi(argv[11]);  Ncy = std::stoi(argv[12]); }

  if (argc < 2)
  {
    // ===== INTERACTIVE PARAMETER INPUT (only on Rank 0) =====
    if (rank == 0) {
        std::cout << "\n Press ENTER, '.' or '-' to keep default values (shown in brackets)\n\n";

        ask_param("Number of grid nodes along x", Nx);
        ask_param("Number of grid nodes along y", Ny);
        ask_param("Domain length along x", Lx);
        ask_param("Domain length along y", Ly);
        ask_param("Diffusion coefficient mu", mu);
        ask_param("Reaction coefficient c", c);
        ask_param("Overlap size (in number of nodes)", overlap);
        ask_param("Maximum number of iterations", max_it);
        ask_param("Tolerance", tol);
        ask_param("Restart for GMRES", restart);
        ask_param("Number of coarse grid nodes along x", Ncx);
        ask_param("Number of coarse grid nodes along y", Ncy);
    }

    // BROADCAST parameters to all ranks
    MPI_Bcast(&Nx,           1, MPI_INT,    0, MPI_COMM_WORLD);
    MPI_Bcast(&Ny,           1, MPI_INT,    0, MPI_COMM_WORLD);
    MPI_Bcast(&Lx,           1, MPI_INT,    0, MPI_COMM_WORLD);
    MPI_Bcast(&Ly,           1, MPI_INT,    0, MPI_COMM_WORLD);
    MPI_Bcast(&mu,           1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&c,            1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&overlap,      1, MPI_INT,    0, MPI_COMM_WORLD);
    MPI_Bcast(&max_it,       1, MPI_INT,    0, MPI_COMM_WORLD);
    MPI_Bcast(&tol,          1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&restart,      1, MPI_INT,    0, MPI_COMM_WORLD);
    MPI_Bcast(&Ncx,          1, MPI_INT,    0, MPI_COMM_WORLD);
    MPI_Bcast(&Ncy,          1, MPI_INT,    0, MPI_COMM_WORLD);
  }

  if (size == 1 && overlap > 0) {
    if (rank == 0) {
      std::cout << "\nWARNING: Running with 1 processor." << std::endl;
      std::cout << "           Setting overlap=0.\n" << std::endl;
    }
    overlap = 0;  // Force no overlap for sequential run
  }

  // Coarse grid
  int Ncx = Partition::find_best_coarse_grid(Nx, 20); 
  int Ncy = Partition::find_best_coarse_grid(Ny, 20);

  // Processes topology (MPI cartesian grid): divide processes into a Px*Py grid
  int dims[2] = {0, 0};     // 0 allows MPI to choose best subdivision
  MPI_Dims_create(size, 2, dims);

  // Create cartesian communicator
  int periods[2] = {0, 0};  // No periodicity for Dirichlet
  int reorder = 1;
  MPI_Comm cart;
  MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &cart);

  // Update rank and coors into new communicator
  int cart_rank;
  MPI_Comm_rank(cart, &cart_rank);
  int coords[2];  // coords[0] = x, coords[1] = y
  MPI_Cart_coords(cart, cart_rank, 2, coords);

  // Identify neighbors using shift:
  //   -  Left / right -> 0 axis
  //   -  Down / up    -> 1 axis
  int left, right, down, up;
  MPI_Cart_shift(cart, 0, 1, &left, &right);
  MPI_Cart_shift(cart, 1, 1, &down, &up);

  // Print solver configuration
  if (cart_rank == 0) {
    std::cout << "======== PARALLEL RAS SOLVER 2D =======" << std::endl;
    std::cout << "  Grid size:             " << Nx << " x " << Ny << std::endl;
    std::cout << "  Coarse grid size:      " << Ncx << " x " << Ncy << std::endl;
    std::cout << "  Processes:             " << size << " (" << dims[0] << " x " << dims[1] << " grid)" << std::endl;
    std::cout << "  Overlap:               " << overlap << " cells" << std::endl;
    std::cout << "  Diffusion coefficient: " << mu << std::endl;
    std::cout << "  Reaction coefficient:  " << c << std::endl;
    std::cout << "  Coarse grid size:      " << Ncx << " x " << Ncy << std::endl;
    std::cout << "------------------------------------" << std::endl;
  }


  // DOMAIN DECOMPOSITION: compute core region
  int core_i0, core_nx;  // Start index and length along x
  int core_j0, core_ny;  // Start index and length along y

  // Partition x axis based on process coord in the grid (coords[0])
  Partition::compute_1d_partition(Nx, dims[0], coords[0], core_i0, core_nx);

  // Partition y axis based on process coord in the grid (coords[1])
  Partition::compute_1d_partition(Ny, dims[1], coords[1], core_j0, core_ny);

  // Create local problem
  LocalProblem* local = new LocalProblem(core_i0, core_j0, core_nx, core_ny, 
                                         overlap, Nx, Ny, hx, hy, mu, c);

  // Check if LU factorization succeeded
  if (!local->is_lu_ok()) {
    std::cerr << "Local LU factorization failed on rank " << cart_rank << "!" << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  // Create CoarseSolver object
  CoarseSolver *coarse = new CoarseSolver(Nx, Ny, Ncx, Ncy, mu, c, rank);

  // Create solver
  Solver solver(cart, cart_rank, size,
                Nx, Ny,
                core_i0, core_nx,
                core_j0, core_ny,
                hx, hy, mu, c,
                left, right, down, up,
                local, coarse);

  // Start timer
  MPI_Barrier(cart);  // Synchronize before timing
  auto start = std::chrono::high_resolution_clock::now();

  // Execution
  solver.run(max_it, tol, restart);

  // Stop timer
  MPI_Barrier(cart);  // Synchronize after solving
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

  if (cart_rank == 0) {
    std::cout << "\n======== TIMING ========" << std::endl;
    std::cout << "Total solver time: " << duration.count() << " ms" << std::endl;
    std::cout << "                   " << duration.count() / 1000.0 << " s" << std::endl;
  }

  delete local;
  delete coarse;

  MPI_Finalize();
  return 0;
}









