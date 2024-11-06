import decimal
from decimal import Decimal, getcontext

# Set the precision
getcontext().prec = 110

# Constants
c = 299792458  # Speed of light in m/s
h = Decimal('6.62607015e-34')  # Planck constant in J*s
k_B = Decimal('1.38e-23')  # Boltzmann constant in J/K
TSR = Decimal(c**2) / k_B  # Temperature to Speed Ratio in K*m/s
Q = Decimal('2') ** (Decimal('1') / Decimal('12'))  # Fractal structure parameter

def chudnovsky_pi(n_terms):
    C = Decimal(426880) * Decimal(10005).sqrt()
    K = Decimal(6)
    M = Decimal(1)
    X = Decimal(1)
    L = Decimal(13591409)
    S = Decimal(0)  # Initialize S to 0
    for i in range(1, n_terms + 1):  # Change to n_terms + 1 to avoid i = 0
        if i == 1:  # Handle the first iteration separately
            M = Decimal(1)  # Set M to 1 for the first iteration
        else:
            divisor = Decimal((i-1)**3)
            if divisor == Decimal(0):  # Check for zero divisor
                divisor = Decimal(1)  # Set divisor to 1 to avoid division by zero
            M = (K**3 - Decimal(16)*K) * M // divisor
        L += Decimal(545140134)
        X *= Decimal(-262537412640768000)
        term = Decimal(M * L) / X
        # Apply fractal Q to the term
        term *= Q ** (Decimal(i) / Decimal(n_terms))
        # Apply TSR and quantum fluctuations conservatively
        if term < Decimal('1e-15'):
            TSR_rel = TSR
            ΔTSR_q = Decimal('0')
        else:
            TSR_rel = TSR / Decimal((1 - (term**2 / Decimal(c**2))).sqrt())
            ΔTSR_q = h * term
        S += term * TSR_rel + ΔTSR_q
        K += Decimal(12)
    pi = C / S
    return +pi

# Example usage
pi = chudnovsky_pi(100)
print(f"π ≈ {pi}")

# Save results to a JSON file
import json
with open('pi_simulation_results.json', 'w') as f:
    json.dump({"pi_approx": str(pi)}, f)
