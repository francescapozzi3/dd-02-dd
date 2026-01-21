#ifndef HHH_LinearAlgebraTraits__HH
#define HHH_LinearAlgebraTraits__HH
#include <Eigen/SparseCore>
#include <Eigen/SparseLU>
#include <mpi.h>
#include <cmath>
#include <memory>
/*!
 * Adaptors to interface with Eigen
 */

// MODIFIED

namespace LinearAlgebra
{
/*!
 * @tparam Derived The type of the actual Eigen sparse vector
 * @param a a sparse Eigen vector
 * @param b a sparse Eigen vector
 * @return the dot product
 */
template <class Derived>
decltype(auto)
dot(const Eigen::SparseMatrixBase<Derived> &a,
    const Eigen::SparseMatrixBase<Derived> &b)
{
  double local_dot = a.dot(b);
  double global_dot = 0.0;
  MPI_Allreduce(&local_dot, &global_dot, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return global_dot;
}
/*!
 * @tparam Derived The type of the actual Eigen dense vector
 * @param a a dense Eigen vector
 * @param b a dense Eigen vector
 * @return the dot product
 */
template <class Derived>
decltype(auto)
dot(const Eigen::MatrixBase<Derived> &a, const Eigen::MatrixBase<Derived> &b)
{
  double local_dot = a.dot(b);
  double global_dot = 0.0;
  MPI_Allreduce(&local_dot, &global_dot, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return global_dot;
}
/*!
 * @tparam Derived The type of the actual Eigen sparse matrix (or vector)
 * @param a a sparse Eigen matrix
 * @return the 2-norm
 */
template <class Derived>
decltype(auto)
norm(const Eigen::SparseMatrixBase<Derived> &a)
{
  return std::sqrt(dot(a, a));
}
/*!
 * @tparam Derived The type of the actual Eigen dense matrix (or vector)
 * @param a a denst Eigen matrix
 * @return the 2-norm
 */
template <class Derived>
decltype(auto)
norm(const Eigen::MatrixBase<Derived> &a)
{
  return std::sqrt(dot(a, a));
}

} // namespace LinearAlgebra
#endif
