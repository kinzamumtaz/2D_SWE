#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 22:46:52 2025

@author: apple
"""
# For different IC
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter

# Parameters
Lx = 100.0  # Length of the domain in the x-direction
Ly = 100.0  # Length of the domain in the y-direction
Nx = 401   # Number of grid points in the x-direction
Ny = 401   # Number of grid points in the y-direction
T = 2.0   # Total simulation time
dt = 0.0001  # Time step (set based on CFL condition)
g = 2.0     # Gravity constant
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
# center_x = Lx / 2
# center_y = Ly / 2
# r = np.sqrt((X - center_x)**2 + (Y - center_y)**2)

# sigma = 10.0       # Standard deviation (controls the spread)
# h_peak = 10.0      # Peak height of the Gaussian

# # Compute the Gaussian distribution
# h0 = h_peak * np.exp(-((X - center_x)**2 + (Y - center_y)**2) / (2 * sigma**2))

center_x, center_y = 50.0, 50.0  # Center of the dam
r_dam = 20.0                     # Radius of the dam
h_inner = 10.0                   # Height inside the dam
h_outer = 1.0                    # Height outside the dam

# Grid points
x = np.linspace(0, 100, 401)     # Define your x-coordinates
y = np.linspace(0, 100, 401)     # Define your y-coordinates
X, Y = np.meshgrid(x, y)         # Create 2D grid
r = np.sqrt((X - center_x)**2 + (Y - center_y)**2)  # Compute radial distance

# Initial condition
h0 = np.where(r <= r_dam, h_inner, h_outer)

h = h0.copy()
u = np.zeros_like(h)
v = np.zeros_like(h)


fig = plt.figure(figsize=(14, 6))
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.plot_surface(X, Y, h, cmap='viridis', edgecolor='none')
ax1.set_title(f'Initial Condition - Surface')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Water Height')

ax2 = fig.add_subplot(1, 2, 2)
im = ax2.imshow(h, extent=[0, Lx, 0, Ly], origin='lower',
                cmap='viridis', aspect='auto')
plt.colorbar(im, ax=ax2, label='Water Height')
ax2.set_title(f'Initial Condition - Contour')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')

plt.tight_layout()
plt.show()  # instead of save

# Arrays to save the numerical solutions
saved_h = []

# Times to save
save_times = [0.0, 0.5, 0.8, 1.0, 1.5, 2.0]
save_steps = [int(t / dt) for t in save_times]

# Create output folder
import os
os.makedirs("solution_outputs_numerical_circular", exist_ok=True)

# Time integration loop
for step in range(int(max(save_times) / dt) + 1):
    # Predictor step (half-step forward)
    h_half = h.copy()
    u_half = u.copy()
    v_half = v.copy()

    h_half[1:-1, 1:-1] = h[1:-1, 1:-1] - (dt / (2 * dx)) * (h[1:-1, 1:-1] * (u[2:, 1:-1] - u[:-2, 1:-1]))
    h_half[1:-1, 1:-1] -= (dt / (2 * dy)) * (h[1:-1, 1:-1] * (v[1:-1, 2:] - v[1:-1, :-2]))

    u_half[1:-1, 1:-1] = u[1:-1, 1:-1] - (g * dt / (2 * dx)) * (h[2:, 1:-1] - h[:-2, 1:-1])
    v_half[1:-1, 1:-1] = v[1:-1, 1:-1] - (g * dt / (2 * dy)) * (h[1:-1, 2:] - h[1:-1, :-2])

    h[1:-1, 1:-1] -= (dt / dx) * (h_half[1:-1, 1:-1] * (u_half[2:, 1:-1] - u_half[:-2, 1:-1]))
    h[1:-1, 1:-1] -= (dt / dy) * (h_half[1:-1, 1:-1] * (v_half[1:-1, 2:] - v_half[1:-1, :-2]))

    u[1:-1, 1:-1] -= (g * dt / dx) * (h_half[2:, 1:-1] - h_half[:-2, 1:-1])
    v[1:-1, 1:-1] -= (g * dt / dy) * (h_half[1:-1, 2:] - h_half[1:-1, :-2])

    # Boundary conditions
    u[:, 0] = u[:, 1]; u[:, -1] = u[:, -2]
    u[0, :] = u[1, :]; u[-1, :] = u[-2, :]
    v[:, 0] = v[:, 1]; v[:, -1] = v[:, -2]
    v[0, :] = v[1, :]; v[-1, :] = v[-2, :]

    current_time = step * dt
    if step in save_steps:
        h_smoothed = gaussian_filter(h, sigma=3.0)
        saved_h.append(h_smoothed.copy())

        np.savetxt(f"solution_outputs_numerical_circular/numerical_circular_t{current_time:.4f}.csv",
                   h_smoothed, delimiter=",")

        # Optional: Plot
        fig = plt.figure(figsize=(14, 6))
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        ax1.plot_surface(X, Y, h_smoothed, cmap='viridis', edgecolor='none')
        ax1.set_title(f'3D Surface at T = {current_time:.2f} s')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Water Height')

        ax2 = fig.add_subplot(1, 2, 2)
        im = ax2.imshow(h_smoothed.T, extent=[0, Lx, 0, Ly], origin='lower',
                        cmap='viridis', aspect='auto')
        plt.colorbar(im, ax=ax2, label='Water Height')
        ax2.set_title(f'Contour Plot at T = {current_time:.2f} s')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')

        plt.tight_layout()
        plt.savefig(f"solution_outputs_numerical_circular/numerical_circular_t{current_time:.4f}.png", dpi=300)
        plt.show()


