import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter

# Parameters
Lx = 100.0  # Length of the domain in the x-direction (m)
Ly = 100.0  # Length of the domain in the y-direction (m)
Nx = 401    # Number of grid points in the x-direction
Ny = 401    # Number of grid points in the y-direction
T = 2.0    # Total simulation time (s)
dt = 0.0001 # Time step (s)
g = 3.81    # Gravity constant (m/s^2)
h_l = 5.0  # Height upstream or in the rectangle (m)
h_r = 0.2   # Height elsewhere (m)

# Grid
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y, indexing='ij')

# Grid spacing
dx = Lx / (Nx - 1)
dy = Ly / (Ny - 1)

# Initial condition with rectangular profile
h0 = np.ones((Nx, Ny)) * h_r  # Default height

# Define the rectangular region with higher water height
rect_x_start, rect_x_end = 30, 70  # Rectangle start and end in x-direction (m)
rect_y_start, rect_y_end = 30, 70   # Rectangle start and end in y-direction (m)

# Apply the higher water height in the rectangular region
rect_indices = (X >= rect_x_start) & (X <= rect_x_end) & (Y >= rect_y_start) & (Y <= rect_y_end)
h0[rect_indices] = h_l  # Higher water height in the rectangle

# Plot the initial profile as a surface plot
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
surface_plot = ax.plot_surface(X, Y, h0, cmap='viridis', edgecolor='none')
ax.set_title('Rectangular Dam Break')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Water Surface Height (m)')
plt.show()


# Initialize variables
h = h0.copy()
u = np.zeros_like(h)
v = np.zeros_like(h)

# Lax-Wendroff Scheme
for step in range(int(T / dt)):
    # Predictor step (half-step forward)
    h_half = h.copy()
    u_half = u.copy()
    v_half = v.copy()

    h_half[1:-1, 1:-1] = h[1:-1, 1:-1] - (dt / (2 * dx)) * (h[1:-1, 1:-1] * (u[2:, 1:-1] - u[:-2, 1:-1]))
    h_half[1:-1, 1:-1] -= (dt / (2 * dy)) * (h[1:-1, 1:-1] * (v[1:-1, 2:] - v[1:-1, :-2]))

    u_half[1:-1, 1:-1] = u[1:-1, 1:-1] - (g * dt / (2 * dx)) * (h[2:, 1:-1] - h[:-2, 1:-1])
    v_half[1:-1, 1:-1] = v[1:-1, 1:-1] - (g * dt / (2 * dy)) * (h[1:-1, 2:] - h[1:-1, :-2])

    # Corrector step (full step update)
    h[1:-1, 1:-1] -= (dt / dx) * (h_half[1:-1, 1:-1] * (u_half[2:, 1:-1] - u_half[:-2, 1:-1]))
    h[1:-1, 1:-1] -= (dt / dy) * (h_half[1:-1, 1:-1] * (v_half[1:-1, 2:] - v_half[1:-1, :-2]))

    u[1:-1, 1:-1] -= (g * dt / dx) * (h_half[2:, 1:-1] - h_half[:-2, 1:-1])
    v[1:-1, 1:-1] -= (g * dt / (2 * dy)) * (h_half[1:-1, 2:] - h_half[1:-1, :-2])

    # Boundary conditions (zero-gradient)
    u[:, 0] = u[:, 1]
    u[:, -1] = u[:, -2]
    u[0, :] = u[1, :]
    u[-1, :] = u[-2, :]

    v[:, 0] = v[:, 1]
    v[:, -1] = v[:, -2]
    v[0, :] = v[1, :]
    v[-1, :] = v[-2, :]
    # Visualization every 100 steps
    if step % 100 == 0:
        h_smoothed = gaussian_filter(h, sigma=4.0)
        current_time = step * dt
    
        fig = plt.figure(figsize=(14, 6))
    
        # Surface Plot
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        ax1.plot_surface(X, Y, h_smoothed, cmap='viridis', edgecolor='none')
        ax1.set_title(f'3D Surface at T = {current_time:.2f} s')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Water Height')
    
        # Heatmap
        ax2 = fig.add_subplot(1, 2, 2)
        heatmap = ax2.imshow(h_smoothed.T, extent=[0, Lx, 0, Ly], origin='lower', cmap='viridis', aspect='auto')
        plt.colorbar(heatmap, ax=ax2, label='Water Height')
        ax2.set_title(f'Contour Plot at T = {current_time:.2f} s')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
    
        plt.tight_layout()
        plt.show()
 