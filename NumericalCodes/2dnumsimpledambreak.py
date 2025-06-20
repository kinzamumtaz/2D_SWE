import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter

# Parameters
Lx = 100.0  # Length of the domain in the x-direction
Ly = 100.0  # Length of the domain in the y-direction
Nx = 401   # Number of grid points in the x-direction
Ny = 401   # Number of grid points in the y-direction
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


    # Boundary conditions (zero-gradient)
    # u[:, 0] = u[:, 1]
    # u[:, -1] = u[:, -2]
    # u[0, :] = u[1, :]
    # u[-1, :] = u[-2, :]

    # v[:, 0] = v[:, 1]
    # v[:, -1] = v[:, -2]
    # v[0, :] = v[1, :]
    # v[-1, :] = v[-2, :]
    #reflective boundary

    # Save data at specified steps
    # if step in save_steps:

    #     h_smoothed = gaussian_filter(h, sigma=2.0)
    #     saved_h.append(h_smoothed.copy())

    if step % 1000 == 0:
        h_smoothed = gaussian_filter(h, sigma=0.0)
        current_time = step * dt
        
        fig = plt.figure(figsize=(14, 6))
    
        # Surface Plot
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        ax1.plot_surface(X, Y, h_smoothed, cmap='viridis', edgecolor='none')
        ax1.set_title(f'3D Surface at T = {current_time:.2f} s')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Water Height')
    
        # Contour Plot
        # ax2 = fig.add_subplot(1, 2, 2)
        # contour = ax2.contourf(X, Y, h_smoothed, cmap='viridis')
        # fig.colorbar(contour, ax=ax2, label='Water Height')
        # ax2.set_title(f'Contour Plot at T = {current_time:.2f} s')
        # ax2.set_xlabel('X')
        # ax2.set_ylabel('Y')
    
        # plt.tight_layout()
        # plt.show()

    # Visualization at every 10 steps
    # if step % 10000 == 0:
    #     current_time = step * dt
        
    #     fig = plt.figure(figsize=(14, 6))
        
    #     # 3D surface plot
    #     ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    #     ax1.plot_surface(X, Y, h_smoothed, cmap="viridis", edgecolor="none")
    #     ax1.set_xlabel("X")
    #     ax1.set_ylabel("Y")
    #     ax1.set_zlabel("Water Height (h)")
    #     ax1.set_title(f"Surface Plot at T = {current_time:.2f} s")

        # Cross-section plot at fixed y-coordinate (center of domain)
        y_index = Ny // 2  # Index for y = Ly / 2
        cross_section_h = h_smoothed[y_index, :]
        x_values = np.linspace(0, Lx, Nx)

        ax2 = fig.add_subplot(1, 2, 2)
        ax2.plot(x_values, cross_section_h, label=f"T = {current_time:.2f} s", color="blue")
        ax2.set_title(f'Cross-section Y = 50.0, Time = {current_time:.2f} [s]')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Water Height (h)')
        ax2.grid()
        ax2.legend()
        plt.tight_layout()
        plt.show()

# Convert saved data to arrays
#saved_h = np.array(saved_h)

# Save the numerical solutions
#np.save("h_numerical.npy", saved_h)


#print(saved_h.shape)
