import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from mpl_toolkits.mplot3d import Axes3D  
import matplotlib.cm as cm

fname = "solution.csv"
created_demo = False

if not os.path.exists(fname):
    print("The file 'solution.csv' was not found. I'll generate a demo solution and save it as 'solution.csv'.")
    # demo parameters (match the defaults in C++ code)
    Nx = 51; Ny = 51; mu = 0.01; c = 5.0
    hx = 1.0 / (Nx - 1); hy = 1.0 / (Ny - 1)
    N = Nx * Ny
    # assemble sparse matrix A (full grid including Dirichlet boundaries as identity rows)
    rows = []
    cols = []
    vals = []
    b = np.zeros(N)
    for j in range(Ny):
        for i in range(Nx):
            k = i + j * Nx
            xg = i * hx; yg = j * hy
            if i == 0 or i == Nx-1 or j == 0 or j == Ny-1:
                # Dirichlet
                rows.append(k); cols.append(k); vals.append(1.0)
                b[k] = 0.0
            else:
                idx2 = 1.0/(hx*hx); idy2 = 1.0/(hy*hy)
                center = mu * (2.0*idx2 + 2.0*idy2) + c
                offx = -mu * idx2
                offy = -mu * idy2
                rows.extend([k,k,k,k,k])
                cols.extend([k, k-1, k+1, k-Nx, k+Nx])
                vals.extend([center, offx, offx, offy, offy])
                b[k] = 1.0  # f=1
    A = sp.csr_matrix((vals,(rows,cols)), shape=(N,N))
    print("Solve reference problem (direct sparse solve)")
    u_demo = spla.spsolve(A, b)
    # write solution.csv
    with open(fname, "w") as f:
        f.write("x,y,u\n")
        for j in range(Ny):
            for i in range(Nx):
                f.write(f"{i*hx},{j*hy},{u_demo[i + j*Nx]}\n")
    created_demo = True
    print("Demo saved in 'solution.csv'.")

# Load solution.csv created by your code (or the demo)
df = pd.read_csv(fname)
# determine grid
x_unique = np.sort(df['x'].unique())
y_unique = np.sort(df['y'].unique())
Nx = len(x_unique); Ny = len(y_unique)
hx = x_unique[1] - x_unique[0] if Nx>1 else 1.0
hy = y_unique[1] - y_unique[0] if Ny>1 else 1.0
# reshape u into a 2D array with shape (Ny, Nx) in plotting (Y rows)
U_file = df['u'].values.reshape((Ny, Nx))

# build reference sparse system - same discretization and BCs
mu = 0.01; c = 5.0  # adjust if you used different values in the MPI run
N = Nx * Ny
rows = []; cols = []; vals = []; b = np.zeros(N)
for j in range(Ny):
    for i in range(Nx):
        k = i + j * Nx
        if i == 0 or i == Nx-1 or j == 0 or j == Ny-1:
            rows.append(k); cols.append(k); vals.append(1.0)
            b[k] = 0.0
        else:
            idx2 = 1.0/(hx*hx); idy2 = 1.0/(hy*hy)
            center = mu * (2.0*idx2 + 2.0*idy2) + c
            offx = -mu * idx2
            offy = -mu * idy2
            rows.extend([k,k,k,k,k])
            cols.extend([k, k-1, k+1, k-Nx, k+Nx])
            vals.extend([center, offx, offx, offy, offy])
            b[k] = 1.0

A = sp.csr_matrix((vals,(rows,cols)), shape=(N,N))

print("Solve reference problem (direct sparse solve)...")
u_ref = spla.spsolve(A, b)
U_ref = u_ref.reshape((Ny,Nx))

# compute difference and norms
diff = U_file - U_ref
l2_err = np.sqrt(np.sum(diff**2)) / np.sqrt(np.sum(U_ref**2))
max_abs = np.max(np.abs(diff))

print(f"Grid: Nx={Nx}, Ny={Ny}, hx={hx:.6g}, hy={hy:.6g}")
print(f"Relative L2 error (file vs ref): {l2_err:.6e}")
print(f"Max abs difference: {max_abs:.6e}")

# create mesh for plotting
X, Y = np.meshgrid(x_unique, y_unique)

# ---------- Efficient 2D visualization: pcolormesh ----------
plt.figure(figsize=(6,5))
plt.title("MPI solution (pcolormesh)")
pcm = plt.pcolormesh(X, Y, U_file, shading='auto')
plt.colorbar(pcm)
plt.xlabel("x"); plt.ylabel("y")
# downsample if we have to many points
num_points = X.size
max_scatter = 2000
if num_points <= max_scatter:
    plt.scatter(X.ravel(), Y.ravel(), s=6, c='k', alpha=0.25)
else:
    # random sample of the points to show
    idx = np.random.choice(num_points, size=max_scatter, replace=False)
    plt.scatter(X.ravel()[idx], Y.ravel()[idx], s=6, c='k', alpha=0.25)
plt.tight_layout()
plt.savefig("solution_mpi_pmesh.png")

# --- Plot 1: MPI solution (contourf) ---
plt.figure(figsize=(6,5))
plt.title("MPI slution - contourf")
plt.contourf(X, Y, U_file, levels=50)
plt.colorbar()
plt.xlabel("x"); plt.ylabel("y")
plt.tight_layout()
plt.savefig("solution_mpi.png")

# --- Plot 2: Reference direct solve (contourf) ---
plt.figure(figsize=(6,5))
plt.title("Reference solution (direct sparse solve)")
plt.contourf(X, Y, U_ref, levels=50)
plt.colorbar()
plt.xlabel("x"); plt.ylabel("y")
plt.tight_layout()
plt.savefig("solution_ref.png")

# --- Plot 3: Difference ---
plt.figure(figsize=(6,5))
plt.title("Difference (MPI - reference)")
plt.contourf(X, Y, diff, levels=50)
plt.colorbar()
plt.xlabel("x"); plt.ylabel("y")
plt.tight_layout()
plt.savefig("solution_diff.png")

# --- Plot 4: central slice comparison ---
yc = 0.5
j_center = np.argmin(np.abs(y_unique - yc))
plt.figure(figsize=(7,4))
plt.title(f"Central cut y≈{y_unique[j_center]:.4f} (row j={j_center})")
# MPI: solid line with circle marker
plt.plot(x_unique, U_file[j_center,:], label="MPI (file)", linestyle='-', linewidth=2, marker='o', markersize=4)
# Reference: dashed line with square marker
plt.plot(x_unique, U_ref[j_center,:], label="Reference (direct)", linestyle='--', linewidth=2, marker='s', markersize=2)
plt.xlabel("x"); plt.ylabel("u(x,y)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("solution_slice.png")

# ---------- 3D surface plot (MPI) ----------
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
ax.set_title('3D surface: MPI solution')
max_side = 200  #if Nx o Ny > max_side, downsample for performance
step_x = max(1, Nx // max_side)
step_y = max(1, Ny // max_side)
Xs = X[::step_y, ::step_x]
Ys = Y[::step_y, ::step_x]
Us = U_file[::step_y, ::step_x]
ax.plot_surface(Xs, Ys, Us, rstride=1, cstride=1, cmap=cm.viridis, linewidth=0, antialiased=True)
ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('u')
plt.tight_layout()
plt.savefig('solution_3d_surface_mpi.png')

# ---------- 3D surface plot (Reference) ----------
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
ax.set_title('3D surface: Reference solution')
Xs = X[::step_y, ::step_x]
Ys = Y[::step_y, ::step_x]
Us_ref = U_ref[::step_y, ::step_x]
ax.plot_surface(Xs, Ys, Us_ref, rstride=1, cstride=1, cmap=cm.inferno, linewidth=0, antialiased=True)
ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('u')
plt.tight_layout()
plt.savefig('solution_3d_surface_ref.png')

print("Plots saved in: solution_mpi.png, solution_mpi_pmesh.png, solution_ref.png, solution_diff.png, solution_slice.png, solution_3d_surface_mpi.png, solution_3d_surface_ref.png")
if created_demo:
    print("(Note: 'solution.csv' was generated as a demo solution.)")
