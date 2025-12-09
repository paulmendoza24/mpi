# 🚀 1D Stencil Simulation with MPI, Numba, and Vectorization

![Python](https://img.shields.io/badge/Python-3.12-blue)
![NumPy](https://img.shields.io/badge/NumPy-1.26.4-brightgreen)
![Numba](https://img.shields.io/badge/Numba-0.59.1-orange)
![MPI](https://img.shields.io/badge/MPI-enabled-blue)
![License](https://img.shields.io/badge/License-MIT-green)

This project implements a **high-performance 1D stencil simulation** using:

- **MPI** for distributed-memory parallelism
- **Numba** for JIT acceleration
- **Vectorized interior updates** for maximum speed

The stencil updates a 1D field according to:

\[
u_i^{t+1} = a \cdot u_{i-1}^t + b \cdot u_i^t + c \cdot u_{i+1}^t
\]

---

## ✨ Features

- ⚡ Fast MPI + Numba + vectorized 1D stencil
- 🧮 Handles **10 million+ grid points**
- 🔁 Multiple timesteps with halo exchange
- 📏 Ghost cells (halos) at boundaries
- 🚀 Non-blocking MPI for overlapping communication and computation

---

## 📦 Requirements

Install dependencies using `pip`:

```bash
pip install -r requirements.txt
```
requirements.txt:
```
numpy
numba
mpi4py
```
---
## ▶️ How to Run
# Linux / macOS
Run with 4 processes:
```
mpiexec -n 4 python stencil_mpi_numba.py
```
# Windows (MS-MPI)
1. Install [Microsoft MPI](https://learn.microsoft.com/en-us/message-passing-interface/microsoft-mpi)
2. Install mpi4py:
```
pip install mpi4py
```
3. Run with 4 processes:
```
mpiexec -n 4 python stencil_mpi_numba.py
```
Output:
```
FAST MPI + NUMBA + VECTORIZED time = 0.563421 s
```
---
# 🧠 Code Overview
## 1️⃣ MPI Setup
```
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
```
- rank → process ID
- size → total number of processes
- Each process handles localN = N // size grid points
---
## 2️⃣ Array Allocation with Ghost Cells
```
u = np.zeros(localN + 2, dtype=np.float64)
un = np.zeros(localN + 2, dtype=np.float64)
```
- +2 for halos at left and right boundaries
- Initial condition: spike at global center:
```
global_center = N // 2
if rank == global_center // localN:
    local_index = (global_center % localN) + 1
    u[local_index] = 1.0
```
---
## 3️⃣ Vectorized Interior Update
```
un[2:localN] = 0.25*u[1:localN-1] + 0.5*u[2:localN] + 0.25*u[3:localN+1]
```
- Only updates interior points (halo-independent)
- Fully vectorized with NumPy → very fast
---
## 4️⃣ Numba-accelerated Boundary Update
```
from numba import njit

@njit
def update_boundaries(u, un, localN):
    un[1] = 0.25*u[0] + 0.5*u[1] + 0.25*u[2]         # left boundary
    un[localN] = 0.25*u[localN-1] + 0.5*u[localN] + 0.25*u[localN+1]  # right
```
- Handles stencil at boundaries
- Uses halo values from neighboring processes
---
## 5️⃣ Non-blocking MPI Halo Exchange
```
reqs = []
reqs.append(comm.Irecv(u[0:1], source=left, tag=1))
reqs.append(comm.Irecv(u[localN+1:localN+2], source=right, tag=0))
reqs.append(comm.Isend(u[1:2], dest=left, tag=0))
reqs.append(comm.Isend(u[localN:localN+1], dest=right, tag=1))

MPI.Request.Waitall(reqs)
```
- Sends/receives halos asynchronously
- Overlaps communication with interior updates
---
## 6️⃣ Swap Arrays
```
u, un = un, u
```
- Avoids copying arrays → improves performance
---
## ⚙️ Performance
| Grid points | Timesteps | Processes | Time (s) |
| ----------- | --------- | --------- | -------- |
| 10,000,000  | 100       | 4         | 0.56     |
Vectorized interior + Numba boundaries + MPI overlapping communication achieves high efficiency.
---
## 📁 File Structure
```
├── stencil_mpi_numba.py  # Main simulation script
├── requirements.txt      # Dependencies
└── README.md             # Project documentation
```
