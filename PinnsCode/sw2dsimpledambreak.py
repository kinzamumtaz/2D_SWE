!pip install deepxde
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import deepxde as dde
from deepxde.backend import tf
import pandas as pd
import os
dde.config.set_random_seed(42)
dde.config.real.set_float32()

dim_input = 3
dim_output = 3

scale_h = 1000.0

Time = 2.0

X_min = 0.0
X_max = 100.0
Y_min = 0.0
Y_max = 100.0
X_0 = 50.0
g = 9.81


def on_initial(_, on_initial):
    return on_initial

def boundary_x1(x, on_boundary):
    return on_boundary and np.isclose(x[0], X_min)

def boundary_x2(x, on_boundary):
    return on_boundary and np.isclose(x[0], X_max)
def boundary_y1(x, on_boundary):
    return on_boundary and np.isclose(x[1], Y_min)

def boundary_y2(x, on_boundary):
    return on_boundary and np.isclose(x[1], Y_max)
def func_IC_h(x):
    h_r = 10.0  # Height on the right side
    h_l = 1.0  # Height on the left side
    return np.where(x[:, 0:1] <= X_0, h_r, h_l)

# def func_IC_h(x):
#     center_x=50.0
#     center_y=50.0
#     radius=20
#     return 10.0 * ((x[:, 0:1] - center_x) * (x[:, 0:1] - center_x) + \
#                     (x[:, 1:2] - center_y) * (x[:, 1:2] - center_y) <= \
#                         (radius*radius)) + \
#             1.0 * ((x[:, 0:1] - center_x) * (x[:, 0:1] - center_x) + \
#                   (x[:, 1:2] - center_y) * (x[:, 1:2] - center_y) > \
#                       (radius*radius))
# Create a meshgrid for plotting
# Create a meshgrid for plotting
N = 1000
X_plot, Y_plot = np.meshgrid(np.linspace(0, 100, N), np.linspace(0, 100, N))
X_flat = X_plot.flatten()
Y_flat = Y_plot.flatten()
T_flat = np.zeros_like(X_flat)  # Assuming t=0 for the initial condition
Q_plot = np.column_stack((X_flat, Y_flat, T_flat))
h_plot = func_IC_h(Q_plot).reshape(N, N)

# Plot the initial condition with a surface plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X_plot, Y_plot, h_plot, cmap='viridis')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Water Surface Height')
ax.set_title('Dam Break Initial Condition at t = 0')
plt.show()


def func_IC_u(x):
  return np.zeros([len(x), 1])

def func_IC_v(x):
  return np.zeros([len(x), 1])

def func_BC_all(x):
  return np.zeros([len(x), 1])

def pde(x, y):
    h = y[:, 0:1]
    u = y[:, 1:2]
    v = y[:, 2:3]

    U1 = h
    U2 = h * u
    U3 = h * v

    E1 = h * u
    E2 = h * u * u + 0.5 * h * h * g
    E3 = h * u * v

    G1 = h * v
    G2 = h * v * u
    G3 = h * v * v + 0.5 * h*h * g

    E1_x = tf.gradients(E1, x)[0][:, 0:1]
    E2_x = tf.gradients(E2, x)[0][:, 0:1]
    E3_x = tf.gradients(E3, x)[0][:, 0:1]

    G1_y = tf.gradients(G1, x)[0][:, 1:2]
    G2_y = tf.gradients(G2, x)[0][:, 1:2]
    G3_y = tf.gradients(G3, x)[0][:, 1:2]

    U1_t = tf.gradients(U1, x)[0][:, 2:3]
    U2_t = tf.gradients(U2, x)[0][:, 2:3]
    U3_t = tf.gradients(U3, x)[0][:, 2:3]

    equaz_1 = U1_t + E1_x + G1_y
    equaz_2 = U2_t + E2_x + G2_y
    equaz_3 = U3_t + E3_x + G3_y

    return [equaz_1, equaz_2, equaz_3]

# Modify boundary conditions accordingly

geom = dde.geometry.Rectangle([X_min, Y_min],[X_max, Y_max])
timedomain = dde.geometry.TimeDomain(0.0, Time)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

IC_h = dde.IC(geomtime, func_IC_h, on_initial, component = 0)
IC_u = dde.IC(geomtime, func_IC_u, on_initial, component = 1)
IC_v = dde.IC(geomtime, func_IC_v, on_initial, component = 2)

BC_u1 = dde.DirichletBC(geomtime, func_BC_all,
boundary_x1, component = 1)
BC_u2 = dde.DirichletBC(geomtime, func_BC_all,
boundary_x2, component = 1)
BC_v1 = dde.DirichletBC(geomtime, func_BC_all,
boundary_y1, component = 2)
BC_v2 = dde.DirichletBC(geomtime, func_BC_all,
boundary_y2, component = 2)

IC_BC = [IC_h, IC_u, IC_v, BC_u1, BC_u2, BC_v1, BC_v2]

data = dde.data.TimePDE(
    geomtime, pde, IC_BC,
    num_domain=20000,
    num_boundary=1000,
    num_initial=20000)

net = dde.maps.FNN(
    layer_sizes=[dim_input] + [100] * 5 + [dim_output],
    activation="tanh",
    kernel_initializer="Glorot uniform")
#net.apply_output_transform(lambda x, y: func_IC_h_circular(x,y))

model = dde.Model(data, net)

model.compile('adam', lr=0.0001)

#model.train(iterations=12000)
os.makedirs("solution_outputs_2dstep", exist_ok=True)
# Train the model and capture the training history
losshistory, train_state = model.train(iterations=20000)

loss_train = losshistory.loss_train

# Compute total loss by summing all components at each epoch
total_loss = [sum(l) for l in loss_train]
df = pd.DataFrame({"Step": range(1, len(total_loss)+1), "Loss": total_loss})
df.to_csv("solution_outputs_2dstep/training_loss_2dstep.csv", index=False)

plt.figure()
plt.semilogy(total_loss, color='red', label="Training Loss")
plt.xlabel("# Steps")
plt.ylabel("Loss")
plt.title("Loss History")
plt.legend()
plt.tight_layout()
plt.savefig("solution_outputs_2dstep/training_loss_plot_2dstep.png", dpi=300)
plt.close()

N_x = 500
N_y =500

# Plot for Side-by-Side Surface (2D) and Cross-section (1D)
cross_section_y = 50.0  # Example: Fix Y = 50

X_line = np.linspace(X_min, X_max, N_x)
Y_fixed = np.ones_like(X_line) * cross_section_y

X_plot, Y_plot = np.meshgrid(
    np.linspace(X_min, X_max, N_x), np.linspace(Y_min, Y_max, N_y)
)
X_flat = X_plot.flatten()
Y_flat = Y_plot.flatten()

plot_times = [0.0, 0.2, 0.5, 0.8, 0.9, 1.0]

for i, t in enumerate(plot_times):
    T_fixed = np.ones_like(X_line) * t
    T_flat = np.ones_like(X_flat) * t

    # Cross-section prediction
    Q_fixed = np.column_stack((X_line, Y_fixed, T_fixed))
    W_fixed = model.predict(Q_fixed)
    h_fixed = W_fixed[:, 0]

    # Surface prediction
    Q_plot = np.column_stack((X_flat, Y_flat, T_flat))
    W_plot = model.predict(Q_plot)
    Z_plot = W_plot[:, 0].reshape(N_y, N_x)

    # Save height surface as CSV
    filename_csv = f"solution_outputs_2dstep/2dstep_t{t:.2f}.csv"
    pd.DataFrame(Z_plot).to_csv(filename_csv, index=False)
    print(f"Saved: {filename_csv}")

    # Plot and save PNG
    fig = plt.figure(figsize=(14, 6))

    # Plot 1: Surface plot
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot_surface(X_plot, Y_plot, Z_plot, cmap="viridis", edgecolor="none")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Water Height (h)")
    ax1.set_title(f"Surface Plot at T = {t:.2f} s")

    # Plot 2: Cross-section
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(X_line, h_fixed, label=f"T = {t:.2f} s", color="blue")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Water Height (h)")
    ax2.set_title(f"Cross-section at Y = {cross_section_y}, T = {t:.2f} s")
    ax2.grid()
    ax2.legend()

    plt.tight_layout()

    plt.show()
