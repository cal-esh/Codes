import numpy as np
import manual_data_process as mdp
import matplotlib.pyplot as plt
import warnings
#warnings.filterwarnings("ignore", category=np.ComplexWarning)

def analyze_power_flow():
    gamma = mdp.gamma
    H = mdp.H
    P = mdp.P
    Q = mdp.Q
    D = mdp.D
    E = mdp.E
    impedance_matrix = mdp.impedance_matrix
    impedance_matrix_faulted = mdp.impedance_matrix_faulted
    return P, Q, E, H, D, gamma, impedance_matrix, impedance_matrix_faulted

if __name__ == "__main__":
    case_data = case14()
    P, Q, E, H, D, gamma, impedance_matrix, impedance_matrix_faulted = analyze_power_flow()
    # Print the results
    print("Impedance Matrix:")
    print(impedance_matrix)

    print(f'  Net PF: P = {P}')
    print(f'  Net PF: Q = {Q}')
    print(f'  Initial Phase (gamma_0): {gamma}')
    print(f'  Inertia (H): {H}')
    print(f'  Damping (D): {D}')
    print(f'  Terminal Voltage (E): {E}')
    print('---')
