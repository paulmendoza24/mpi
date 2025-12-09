import numpy as np
import time
from numba import jit, prange

# -----------------------------
# Simulation parameters
# -----------------------------
num_points = 10_000_000  # Total number of spatial grid points (N)
num_timesteps = 100  # Number of time iterations (T)

# Stencil coefficients (a*u[i-1] + b*u[i] + c*u[i+1])
coef_left = 0.25  # coefficient 'a'
coef_center = 0.5  # coefficient 'b'
coef_right = 0.25  # coefficient 'c'

# -----------------------------
# Allocate arrays
# u  = current solution
# un = next solution
# We allocate +2 for ghost cells at boundaries
# -----------------------------
u_current = np.zeros(num_points + 2)
u_next = np.zeros(num_points + 2)

# Initial condition: spike in center
u_current[num_points // 2] = 1.0


# -----------------------------
# JIT-compiled update function
# Computes u_next based on u_current
# -----------------------------
@jit(nopython=True, parallel=True)
def compute_next_timestep(
    u_current, u_next, coef_left, coef_center, coef_right, num_points
):
    for i in prange(1, num_points + 1):
        u_next[i] = (
            coef_left * u_current[i - 1]
            + coef_center * u_current[i]
            + coef_right * u_current[i + 1]
        )


# -----------------------------
# Copy u_next -> u_current
# -----------------------------
@jit(nopython=True, parallel=True)
def copy_solution(u_current, u_next, num_points):
    for i in prange(1, num_points + 1):
        u_current[i] = u_next[i]


# -----------------------------
# Main simulation loop
# -----------------------------
start_time = time.time()

for t in range(num_timesteps):
    compute_next_timestep(
        u_current, u_next, coef_left, coef_center, coef_right, num_points
    )
    copy_solution(u_current, u_next, num_points)

end_time = time.time()

print(f"Total simulation time: {end_time - start_time:.3f} seconds")
