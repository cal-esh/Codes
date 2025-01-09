from data_process import analyze_power_flow
from utils import null, svds
import numpy as np
import math, scipy
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset, inset_axes
from progress import printProgressBar

plt.rcParams['font.family'] = 'Times New Roman'
line_styles = {
    'solid': '-',
    'dotted': ':',
    'dashed': '--',
    'dashdot': '-.',
    'densely dotted': (0, (1, 1)),
    'loosely dotted': (0, (5, 10)),
    
    'densely dashed': (0, (5, 1)),
    'loosely dashed': (0, (5, 5)),
    
    'densely dashdot': (0, (3, 1, 1, 1)),
    'loosely dashdot': (0, (3, 5, 1, 5))
}

color_list = ['#000000',     # Black
              '#FF0000',     # Strong Red
              '#0000FF',     # Strong Blue
              '#008000',     # Strong Green
              '#8B4513',     # Brown
              '#808080',     # Gray
              '#FF6347',     # Red
              '#00FF00',     # Green
              '#808000',     # Olive
              '#800080']     # Strong Purple

def swing(t, delta_t, y, u, H, P, D, E, Y_normal, Y_fault):
  """Simulate power generators swing dynamics!"""
  N = len(y)//2
  u = np.zeros(N) if u is None else u
  Y_fault = Y_normal if Y_fault is None else Y_fault
  y_p = np.zeros(2*N)
  Y = Y_normal
  k = [
    delta_t * (1 / H[i + 1][i + 1]) * (
      P[i + 1] -
      Y[i + 1][i + 1].real * E[i + 1] ** 2 -
      sum(E[i + 1] * E[j] * Y[i + 1][j].real for j in range(len(Y[i])) if j != i + 1)
    ) for i in range(len(H) - 1)
  ]
  if t < T_fault_in:
    Y = Y_normal
  elif t >= T_fault_in and t < T_fault_end:
    Y = Y_fault
  else:
    Y = Y_normal
  y_p[:N] = [delta_t * y[N + i] + y[i] + u[i] for i in range(N)]
  for i in range(1, N + 1):
    y_p[N - 1 + i] = -D[i][i] * (y[N - 1 + i] + u[i - 1]) + P[i] - \
      Y[i][i].real * E[i] ** 2 - E[i] * E[0] * (Y[i][0].real * math.cos(y[i - 1])
                                                + Y[i][0].imag * math.sin(y[i - 1]))
    
    for j in range(1, len(Y[i])):
      if j != i:
        y_p[N - 1 + i] -= E[i] * E[j] * (Y[i][j].real * math.cos(y[i - 1] - y[j - 1]) +
                                         Y[i][j].imag * math.sin(y[i - 1] - y[j - 1]))
    
    y_p[N - 1 + i] = delta_t * (1 / H[i][i]) * y_p[N - 1 + i] + y[N - 1 + i] - k[i - 1]
  return y_p

def droop(t, delta_t, y, u, Ki, Q, Kp, E, Y_normal, Y_fault):
  """Simulate power generators swing dynamics."""
  N = len(y)//2
  u = np.zeros(len(y)) if u is None else u
  Y_fault = Y_normal if Y_fault is None else Y_fault
  y_p = np.zeros(2*N)
  Y = Y_normal
  k = [
    delta_t * (1 / Ki[i + 1][i + 1]) * (
      Q[i + 1] -
      Y[i + 1][i + 1].imag * E[i + 1] ** 2 -
      sum(E[i + 1] * E[j] * Y[i + 1][j].imag for j in range(len(Y[i])) if j != i + 1)
    ) for i in range(len(Ki) - 1)
  ]
  if t < T_fault_in:
    Y = Y_normal
  elif t >= T_fault_in and t < T_fault_end:
    Y = Y_fault
  else:
    Y = Y_normal
  y_p[:N] = [delta_t * y[N + i] + y[i] + u[i] for i in range(N)]
  for i in range(1, N + 1):
    y_p[N - 1 + i] = -Kp[i][i] * (y[N - 1 + i] + u[i - 1]) + Q[i] - \
      Y[i][i].imag * E[i] ** 2 - E[i] * E[0] * (Y[i][0].real * math.sin(y[i - 1])
                                                - Y[i][0].imag * math.cos(y[i - 1]))
    
    for j in range(1, len(Y[i])):
      if j != i:
        y_p[N - 1 + i] -= E[i] * E[j] * (Y[i][j].real * math.sin(y[i - 1] - y[j - 1]) -
                                         Y[i][j].imag * math.cos(y[i - 1] - y[j - 1]))

    y_p[N - 1 + i] = delta_t * (1 / Ki[i][i]) * y_p[N - 1 + i] + y[N - 1 + i] - k[i - 1]
  return y_p

case = 'IEEE_39'
falut_between_nodes = (16, 17)
k_a = 1
k_f = 0.1591549

# Call the function from the data_process module
P, Qe, E, H, D, gamma, impedance_matrix, impedance_matrix_faulted = analyze_power_flow()
#P = np.array([1.2500, 2.4943, 3.2500, 3.1600, 2.5400, 3.2500, 2.8000, 2.7000, 4.1500, 5.0000])
m = len(gamma)
gamma_0 = [-np.radians(i) for i in gamma]
#y_0 = gamma_0[1:] + m*[0]

y_0 = np.zeros(2*m)
xf = y_0
# sampling time
delta_t = 2.5e-4
#delta_t = 2e-2
# final simulation time
T_sim = 15
# vector of simulation times
t_c = 5
tspan = np.arange(0, T_sim, delta_t)
# fault initial time
T_fault_in = 4
# fault final time
T_fault_end = 4.525

u = None
angle = []
k = 0

for t in tspan:
    A_0 = swing(t, delta_t, y_0, u, H, P, D, E, Y_normal = impedance_matrix, Y_fault = impedance_matrix_faulted)
    angle.append(A_0)
    y_0 = A_0
angle = np.array(angle)
x_0 = angle[int(t_c/delta_t)]

n_state = int(np.shape(angle)[1]/2)
fig, ax = plt.subplots()
# Add dashed line at x = T_fault_in with length of maximum angle
max_angle = k_a*np.max(angle[:, :n_state].flatten())
min_angle = k_a*np.min(angle[:, :n_state].flatten())
ax.plot([T_fault_in, T_fault_in], [min_angle, max_angle], linestyle='-', color='gray', alpha=0.2)
ax.plot([T_fault_end, T_fault_end], [min_angle, max_angle], linestyle='-', color='gray', alpha=0.2)

handles = []
for i in range(n_state):
    #line_style = list(line_styles.keys())[i % len(line_styles)]
    line_style = 'solid' 
    line_handle, = ax.plot(tspan, k_a*angle[:, i], linestyle=line_styles[line_style], color=color_list[i], label=f"G# {i+2}")
    handles.append(line_handle)

ax.fill_between([T_fault_in, T_fault_in, T_fault_end, T_fault_end], [min_angle, max_angle, max_angle, min_angle], min_angle, facecolor='gray', alpha=0.2)
plt.ylabel(r'$\Delta \theta~(\degree)$', fontsize=14, fontname='Times New Roman')
plt.xlabel(r'time (s)', fontsize=14, fontname='Times New Roman')
# Change font type and size for tick labels
ax.tick_params(axis='both', which='major', labelsize=14, direction='in', pad=8)
# Change font type for legend
plt.legend(fontsize=14, loc='upper left', prop={'family': 'Times New Roman'})
# Check if there are handles to create legend
if handles:
    plt.legend(handles=handles, loc="upper left", fontsize=14)
else:
    plt.legend(loc="upper left", fontsize=14)
plt.subplots_adjust(left=0.11, right=0.99, top=0.99, bottom=0.11)
#plt.savefig(f'../figs/{case}_theta.pdf')

fig, ax = plt.subplots()
# Add dashed line at x = T_fault_in with length of maximum angle
max_freq = k_f*np.max(angle[:, n_state:2*n_state].flatten())
min_freq = k_f*np.min(angle[:, n_state:2*n_state].flatten())
ax.plot([T_fault_in, T_fault_in], [min_freq, max_freq], linestyle='-', color='gray', alpha=0.2)
ax.plot([T_fault_end, T_fault_end], [min_freq, max_freq], linestyle='-', color='gray', alpha=0.2)

handles = []
for i in range(n_state, 2*n_state):
    line_style = 'solid' 
    line_handle, = ax.plot(tspan, k_f*angle[:, i], linestyle=line_styles[line_style], color=color_list[i-n_state], label=f"G# {i-n_state+2}")
    handles.append(line_handle)

ax.fill_between([T_fault_in, T_fault_in, T_fault_end, T_fault_end], [min_freq, max_freq, max_freq, min_freq], min_freq, facecolor='gray', alpha=0.2)
plt.ylabel(r'$\Delta {\omega}$ (Hz)', fontsize=14, fontname='Times New Roman')
plt.xlabel(r'time (s)', fontsize=14, fontname='Times New Roman')
# Change font type and size for tick labels
ax.tick_params(axis='both', which='major', labelsize=14, direction='in', pad=8)
# Change font type for legend
plt.legend(fontsize=14, loc='upper left', prop={'family': 'Times New Roman'})
# Check if there are handles to create legend
if handles:
    plt.legend(handles=handles, loc="upper left", fontsize=14)
else:
    plt.legend(loc="upper left", fontsize=14)

plt.subplots_adjust(left=0.11, right=0.99, top=.99, bottom=0.11)
#plt.savefig(f'../figs/{case}_omega.pdf')
plt.show()

############# DROOP LOW GAIN
# number of inputs
v_0 = np.zeros(2*m)
u = None
voltage = []
K_i = 0.1*np.eye(len(gamma)+1)
K_p = 0.1*np.eye(len(gamma)+1)
for t in tspan:
    A_0 = droop(t, delta_t, v_0, u, K_i, Qe, K_p, E, Y_normal = impedance_matrix, Y_fault = impedance_matrix_faulted)
    voltage.append(A_0)
    v_0 = A_0
voltage = np.array(voltage)
voltage_normalized = voltage * np.concatenate((np.zeros(n_state), E[1:n_state+1])) / np.max(voltage, axis=0)

fig, ax = plt.subplots()
# Add dashed line at x = T_fault_in with length of maximum voltage
max_v = np.max(voltage_normalized[:, n_state:2*n_state].flatten())
min_v = np.min(voltage_normalized[:, n_state:2*n_state].flatten())

ax.plot([T_fault_in, T_fault_in], [min_v, max_v], linestyle='-', color='gray', alpha=0.2)
ax.plot([T_fault_end, T_fault_end], [min_v, max_v], linestyle='-', color='gray', alpha=0.2)

handles = []
for i in range(n_state, 2*n_state):
    line_style = 'solid' 
    line_handle, = ax.plot(tspan, voltage_normalized[:, i], linestyle=line_styles[line_style], color=color_list[i-n_state], label=f"G# {i-n_state+2}")
    handles.append(line_handle)

ax.fill_between([T_fault_in, T_fault_in, T_fault_end, T_fault_end], [min_v, max_v, max_v, min_v], min_v, facecolor='gray', alpha=0.2)
plt.ylabel(r'Voltage (p.u.)', fontsize=14, fontname='Times New Roman')
plt.xlabel(r'time (s)', fontsize=14, fontname='Times New Roman')
# Change font type and size for tick labels
ax.tick_params(axis='both', which='major', labelsize=14, direction='in', pad=8)
# Change font type for legend
plt.legend(fontsize=14, loc='upper left', prop={'family': 'Times New Roman'})
# Check if there are handles to create legend
if handles:
    plt.legend(handles=handles, loc="upper right", fontsize=14)
else:
    plt.legend(loc="upper right", fontsize=14)
plt.subplots_adjust(left=0.1, right=0.99, top=.99, bottom=0.11)
#plt.xlim(0, 15)

axins = inset_axes(ax, width="30%", height="30%", loc='center')
for i in range(n_state, 2*n_state):
    #line_style = list(line_styles.keys())[i-n_state % len(line_styles)]
    line_style = 'solid'
    axins.plot(tspan, voltage_normalized[:, i], linestyle=line_styles[line_style], color=color_list[i-n_state])
    #axins.plot(tspan, voltage_normalized[:, i], linestyle=line_styles[list(line_styles.keys())[i-n_state % len(line_styles)]], color='black')
axins.set_xlim(3.5, 5.5)
axins.set_ylim(0.75, 1.1)
plt.xticks(visible=False)
plt.yticks(visible=False)
mark_inset(ax, axins, loc1=2, loc2=1, fc="none", ec="0.5")
#plt.savefig(f'../figs/{case}_voltage.pdf')
plt.show()
