#include "ras_2d.hpp"

#include <algorithm>
#include <vector>
#include <iostream>
#include <cmath>


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
    std::cout << "====== PARALLEL RAS SOLVER 2D ======" << std::endl;
    std::cout << "  Grid size:             " << Nx << " x " << Ny << std::endl;
    std::cout << "  Processes:             " << size << " (" << dims[0] << " x " << dims[1] << " grid)" << std::endl;
    std::cout << "  Overlap:               " << overlap << " cells" << std::endl;
    std::cout << "  Diffusion coefficient: " << mu << std::endl;
    std::cout << "  Reaction coefficient:  " << c << std::endl;
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

  // Create solver
  Solver solver(cart, cart_rank, size,
                Nx, Ny,
                core_i0, core_nx,
                core_j0, core_ny,
                hx, hy, mu, c,
                left, right, down, up,
                local);

  // Execution
  solver.run(max_it, tol, restart);

  delete local;

  MPI_Finalize();
  return 0;
}









