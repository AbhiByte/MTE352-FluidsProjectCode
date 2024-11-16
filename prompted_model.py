import numpy as np
from scipy.integrate import quad

# Constants
g = 9.81  # Gravity (m/s^2)
A_r = 0.0832  # Cross-sectional area of tank (m^2)
D = 0.00794  # Tube diameter (m)
h0 = 0.08  # Initial water height (m)
hf = 0.02  # Final water height (m)
delta_z = 0.04  # Fixed vertical drop of tube (m)

# Calibrated correction factor (based on experimental data at L = 0.2 m)
C = 0.6  # Adjust this factor to match experimental results

# Velocity calculation with correction factor
def velocity(h):
    if h <= 0:
        return 0  # Avoid negative or zero height
    return C * np.sqrt(2 * g * h)

# Time-to-drain integrand
def dt_dh(h, L):
    v = velocity(h)
    if v <= 0:
        return np.inf  # Prevent division by zero
    return A_r / ((np.pi * D**2 / 4) * v)

# Total drain time calculation
def drain_time(L):
    result, _ = quad(dt_dh, hf, h0, args=(L,), limit=1000, epsabs=1e-8, epsrel=1e-8)
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
print("\nValidation Results:")
for L, exp_time in experimental_data.items():
    comp_time = computed_times[L]
    error = abs(comp_time - exp_time)
    print(f"Length: {L:.2f} m | Experimental: {exp_time:.2f} s | Computed: {comp_time:.2f} s | Error: {error:.2f} s")
