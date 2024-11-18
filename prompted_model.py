import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize

# Constants
g = 9.81  # Gravity (m/s^2)
A_r = 0.0832  # Cross-sectional area of tank (m^2)
D = 0.00794  # Tube diameter (m)
h0 = 0.08  # Initial water height (m)
hf = 0.02  # Final water height (m)
delta_z = 0.04  # Fixed vertical drop of tube (m)

# Experimental data
experimental_data = {
    0.2: 199,  # in seconds
    0.3: 214,
    0.4: 266,
    0.6: 288
}

# Dynamic velocity calculation with length-dependent correction factor
def velocity(h, L, C0, k):
    if h <= 0:
        return 0  # Avoid negative or zero height
    C_L = C0 + k * L  # Length-dependent correction factor
    return C_L * np.sqrt(2 * g * h)

# Time-to-drain integrand
def dt_dh(h, L, C0, k):
    v = velocity(h, L, C0, k)
    if v <= 0:
        return np.inf  # Prevent division by zero
    return A_r / ((np.pi * D**2 / 4) * v)

# Total drain time calculation
def drain_time(L, C0, k):
    result, _ = quad(dt_dh, hf, h0, args=(L, C0, k), limit=1000, epsabs=1e-8, epsrel=1e-8)
    return result

# Horizontal range calculation
def horizontal_range(L, C0, k):
    if L <= delta_z:
        return 0  # Prevent invalid tube lengths
    theta = np.arcsin(delta_z / L)  # Tube angle
    v_exit = velocity(hf, L, C0, k)  # Exit velocity at final height
    t_flight = np.sqrt(2 * delta_z / g)  # Time of free fall
    return v_exit * t_flight * np.cos(theta)

def error_function(params):
    C0, k = params  # Unpack parameters
    errors = []
    for L, exp_time in experimental_data.items():
        comp_time = drain_time(L, C0, k)
        errors.append((comp_time - exp_time) ** 2)  # Squared error
    return sum(errors)  # Return scalar value


# Multi-objective optimization function
def objective(L, C0, k):
    t_drain = drain_time(L, C0, k)  # Drainage time
    h_range = horizontal_range(L, C0, k)  # Horizontal range
    if t_drain <= 0 or h_range <= 0:
        return np.inf  # Penalize invalid results
    # Combine objectives: minimize t_drain, maximize h_range
    return t_drain / 300 - h_range / 5  # Adjust weights as needed

# Optimize correction factor parameters
initial_guess = [0.6, 0.1]
result = minimize(lambda params: error_function(params), initial_guess, bounds=[(0.1, 1.0), (0.0, 1.0)])
C0_opt, k_opt = result.x

# Optimize tube length for the multi-objective function
L_bounds = (0.1, 1.0)  # Length range
L_result = minimize(lambda L: objective(L, C0_opt, k_opt), x0=0.5, bounds=[L_bounds])
optimal_L = L_result.x[0]

# Results
print("\nOptimal Tube Length:")
print(f"Optimal L: {optimal_L:.3f} m")
print(f"Drain Time at Optimal L: {drain_time(optimal_L, C0_opt, k_opt):.2f} s")
print(f"Horizontal Range at Optimal L: {horizontal_range(optimal_L, C0_opt, k_opt):.2f} m")

# Validation results
computed_times = {L: drain_time(L, C0_opt, k_opt) for L in experimental_data.keys()}
print("\nValidation Results:")
for L, exp_time in experimental_data.items():
    comp_time = computed_times[L]
    error = abs(comp_time - exp_time)
    print(f"Tube Length: {L:.2f} m | Experimental: {exp_time:.2f} s | Computed: {comp_time:.2f} s | Error: {error:.2f} s")
