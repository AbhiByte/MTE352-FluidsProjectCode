import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar

# Constants
g = 9.81  # Gravity (m/s^2)
A_r = 0.1  # Cross-sectional area of tank (m^2)
D = 0.00794  # Tube diameter (m)
h0 = 0.08  # Initial water height (m)
hf = 0  # Final water height (m)

# Friction factor
f = 0.02  # Assumed constant for validation; can refine later

# Velocity calculation including friction
def velocity(h, L):
    return np.sqrt(2 * g * h / (1 + f * L / D))

# Time-to-drain integrand
def dt_dh(h, L):
    v = velocity(h, L)
    return A_r / ((np.pi * D**2 / 4) * v)

# Total drain time calculation
def drain_time(L):
    result, _ = quad(dt_dh, hf, h0, args=(L,))
    return result

# Validation against experimental data
experimental_data = {
    0.2: 199,  # in seconds
    0.3: 214,
    0.4: 266,
    0.6: 288
}

# Compute drain times
computed_times = {L: drain_time(L) for L in experimental_data.keys()}

# Compare and print results
print("Validation Results:")
for L, exp_time in experimental_data.items():
    comp_time = computed_times[L]
    error = abs(comp_time - exp_time)
    print(f"Length: {L:.2f} m | Experimental: {exp_time:.2f} s | Computed: {comp_time:.2f} s | Error: {error:.2f} s")

# Horizontal range calculation
def horizontal_range(L):
    h_exit = h0 - hf
    v_exit = velocity(h_exit, L)
    t_flight = np.sqrt(2 * h_exit / g)
    return v_exit * t_flight

# Multi-objective optimization (drain time + range)
def objective(L):
    t_drain = drain_time(L)
    h_range = horizontal_range(L)
    # Normalize objectives: smaller values are better
    return t_drain / 300 - h_range / 10  # Weighting factors can be adjusted

# Optimize tube length
result = minimize_scalar(objective, bounds=(0.1, 1.0), method='bounded')
optimal_L = result.x

# Results
print("\nOptimization Results:")
print(f"Optimal Tube Length: {optimal_L:.2f} m")
print(f"Drain Time at Optimal Length: {drain_time(optimal_L):.2f} s")
print(f"Horizontal Range at Optimal Length: {horizontal_range(optimal_L):.2f} m")
