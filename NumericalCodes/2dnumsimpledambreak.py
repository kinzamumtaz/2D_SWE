import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter
import os

# Parameters
Lx = 100.0  # Length of the domain in the x-direction
Ly = 100.0  # Length of the domain in the y-direction
Nx = 501   # Number of grid points in the x-direction
Ny = 501   # Number of grid points in the y-direction
T = 1.5    # Total simulation time
dt = 0.0001  # Time step (set based on CFL condition)
g = 3.81     # Gravity constant
h_l = 10.0   # Height upstream of the dam
h_r = 1.0    # Height downstream of the dam

# Grid
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y)

# Grid spacing
dx = Lx / (Nx - 1)
dy = Ly / (Ny - 1)

# Compute the distance from the center of the domain
center_x = Lx / 2
center_y = Ly / 2

# Dam break initial condition
# h0 = np.ones((Nx, Ny)) * h_l  # Default height
# h0[X >= center_x] = h_r       # Height on the right side of the dam
# Dam break initial condition
h0 = np.ones((Nx, Ny)) * h_l  # Default height
h0[X <= 50] = h_l              # Height on the left side of the dam (x ≤ 50)
h0[X > 50] = h_r               # Height on the right side of the dam (x > 50)

# Initialize variables
h = h0.copy()
u = np.zeros_like(h)
v = np.zeros_like(h)

# # Arrays to save the numerical solutions
# save_steps = np.arange(0, int(T / dt), 1000)  # Save every 1000 steps
# saved_h = []
# saved_u = []
# saved_v = []

# Plot the initial condition
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, h0, cmap='viridis')
ax.set_title('Initial Condition of Dam Break')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Water Height')
plt.show()


output_dir = "solution_outputs_stepped_numerical"
os.makedirs(output_dir, exist_ok=True)

# Desired times to save data
save_times = [0.0, 0.5, 0.9, 1.0, 1.5, 2.0]
save_steps = [int(t / dt) for t in save_times]

for step in range(int(T / dt)):
    # Predictor step (half-step forward)
    h_half = h.copy()
    u_half = u.copy()
    v_half = v.copy()

    # Update intermediate height at half time step
    h_half[1:-1, 1:-1] = h[1:-1, 1:-1] - (dt / (2 * dx)) * (h[1:-1, 1:-1] * (u[2:, 1:-1] - u[:-2, 1:-1]))
    h_half[1:-1, 1:-1] -= (dt / (2 * dy)) * (h[1:-1, 1:-1] * (v[1:-1, 2:] - v[1:-1, :-2]))

    # Update intermediate velocities at half time step
    u_half[1:-1, 1:-1] = u[1:-1, 1:-1] - (g * dt / (2 * dx)) * (h[2:, 1:-1] - h[:-2, 1:-1])
    v_half[1:-1, 1:-1] = v[1:-1, 1:-1] - (g * dt / (2 * dy)) * (h[1:-1, 2:] - h[1:-1, :-2])

    # Corrector step (full step update)
    h[1:-1, 1:-1] -= (dt / dx) * (h_half[1:-1, 1:-1] * (u_half[2:, 1:-1] - u_half[:-2, 1:-1]))
    h[1:-1, 1:-1] -= (dt / dy) * (h_half[1:-1, 1:-1] * (v_half[1:-1, 2:] - v_half[1:-1, :-2]))

    u[1:-1, 1:-1] -= (g * dt / dx) * (h_half[2:, 1:-1] - h_half[:-2, 1:-1])
    v[1:-1, 1:-1] -= (g * dt / (dy)) * (h_half[1:-1, 2:] - h_half[1:-1, :-2])

    # Boundary conditions (zero-gradient)
    h[0, :] = h[1, :]
    h[-1, :] = h[-2, :]
    h[:, 0] = h[:, 1]
    h[:, -1] = h[:, -2]

    u[0, :] = -u[1, :]
    u[-1, :] = -u[-2, :]
    v[:, 0] = -v[:, 1]
    v[:, -1] = -v[:, -2]

    # Optional: Add artificial viscosity
    h[1:-1, 1:-1] += 0.01 * (h[2:, 1:-1] + h[:-2, 1:-1] + h[1:-1, 2:] + h[1:-1, :-2] - 4 * h[1:-1, 1:-1])


    if step in save_steps:
        h_smoothed = gaussian_filter(h, sigma=0.0)
        current_time = step * dt

        # Save CSV
        filename_csv = os.path.join(output_dir, f"stepped_numerical_t{current_time:.4f}.csv")
        np.savetxt(filename_csv, h_smoothed, delimiter=",")

        # Plot and save figure
        fig = plt.figure(figsize=(14, 6))

        # Surface plot
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        ax1.plot_surface(X, Y, h_smoothed, cmap='viridis', edgecolor='none')
        ax1.set_title(f'3D Surface at T = {current_time:.2f} s')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Water Height')

        # Centerline plot
        y_index = Ny // 2
        cross_section_h = h_smoothed[y_index, :]
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.plot(x, cross_section_h, color='blue', label=f'T = {current_time:.2f} s')
        ax2.set_title(f'Centerline (Y = 50) at T = {current_time:.2f} s')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Water Height')
        ax2.grid()
        ax2.legend()

        # Save figure
    #    filename_png = os.path.join(output_dir, f"dambreak_t{current_time:.4f}.png")
        plt.tight_layout()
     #   plt.savefig(filename_png, dpi=300)
        plt.close()

#print(saved_h.shape)
