#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 19 17:55:59 2025

@author: apple
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter

# Parameters
Lx = 100.0  # Length of the domain in the x-direction (m)
Ly = 100.0  # Length of the domain in the y-direction (m)
Nx = 501    # Number of grid points in the x-direction
Ny = 501    # Number of grid points in the y-direction
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

# Parameters for Gaussian
center_x, center_y = 50.0, 50.0
sigma = 10.0

# Compute initial condition h(x, y)
h0= 10.0 * np.exp(-((X - center_x)**2 + (Y - center_y)**2) / (2 * sigma**2))

# Optional: Plot it
plt.imshow(h0, extent=[0, 100, 0, 100], origin='lower', cmap='viridis')
plt.title("Initial Condition: Gaussian Water Height")
plt.colorbar(label="Water Height")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()


# Initialize variables
h = h0.copy()
u = np.zeros_like(h)
v = np.zeros_like(h)
import os


# Prepare output directory
output_dir = "solution_outputs_gaussian_numerical"
os.makedirs(output_dir, exist_ok=True)

# Desired save times (in seconds)
save_times = [0.0, 0.5, 0.8, 1.0, 1.5, 2.0]
save_steps = [int(t / dt) for t in save_times]

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


    if step in save_steps:
        #h_smoothed = gaussian_filter(h, sigma=4.0)
        current_time = step * dt

        # Save CSV file
        filename_csv = os.path.join(output_dir, f"gaussian_t{current_time:.4f}.csv")
        np.savetxt(filename_csv, h, delimiter=",")

        # Create surface + heatmap side by side
        fig = plt.figure(figsize=(14, 6))

        # 3D Surface Plot
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        ax1.plot_surface(X, Y, h, cmap='viridis', edgecolor='none')
        ax1.set_title(f'3D Surface at T = {current_time:.2f} s')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Water Height')

        # Heatmap Plot (with imshow)
        ax2 = fig.add_subplot(1, 2, 2)
        im = ax2.imshow(h, extent=[0, Lx, 0, Ly], origin='lower',
                        cmap='viridis', aspect='auto')
        plt.colorbar(im, ax=ax2, label='Water Height')
        ax2.set_title(f'Heatmap at T = {current_time:.2f} s')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')

        plt.tight_layout()

        # Save plot as image
     #   filename_png = os.path.join(output_dir, f"rectangular_t{current_time:.4f}.png")
      #  plt.savefig(filename_png, dpi=300)
        plt.show()

 