"""Check the balance between vascular supply and consumption."""
import numpy as np
from config import *

O2 = np.full((ROWS, COLS), 0.5 * 1e-3)
GLUCOSE = np.full((ROWS, COLS), 4 * 1e-3)
CO2 = np.full((ROWS, COLS), 2 * 1e-3)

def get_vitality():
    o2_component = VITALITY_ENERGY["phi"] * O2/(O2+CELLULAR_RESPIRATION["ch_O2"])+VITALITY_ENERGY["k_W"]
    glucose_component = GLUCOSE/(GLUCOSE+CELLULAR_RESPIRATION["ch_g"])
    co2_component = np.exp(-5 * (CO2/CELLULAR_RESPIRATION["ch_CO2"]-1)**4 * (CO2-CELLULAR_RESPIRATION["ch_CO2"]>0))
    return o2_component * glucose_component * co2_component

# Calculate consumption
reaction = -CELLULAR_RESPIRATION["gamma_0"]*get_vitality() - CELLULAR_RESPIRATION["eps_O2"]*O2

# Calculate vascular supply
delta_di = (1-(1-CELLULAR_RESPIRATION["d_O2"]/HEMODYNAMICS_TME["dt_p"])**2)**2
P_i = (1-delta_di)*(HEMODYNAMICS_TME["L0_p"]/HEMODYNAMICS_TME["k_iEC"]*(40-5-HEMODYNAMICS_TME["delta_vEC"]*(HEMODYNAMICS_TME["pi_lum"]-HEMODYNAMICS_TME["pi_insEC"])))
vascular_base = HEMODYNAMICS_TME["k_iEC"] * (P_i * HEMODYNAMICS_TME["S_VEC"] * 0.1 * CELLULAR_RESPIRATION["c_pO2"] + HEMODYNAMICS_TME["S_VEC"] * 0.1 * (CELLULAR_RESPIRATION["c_pO2"] - O2)*P_i/(np.exp(P_i)-1))

print("=== BALANCE ANALYSIS ===\n")
print(f"P_i: {P_i:.6e}")
print(f"delta_di: {delta_di:.6e}")
print(f"\nMean consumption rate: {np.mean(reaction):.6e} mM/s")
print(f"Mean vascular supply: {np.mean(vascular_base):.6e} mM/s")
print(f"\nRatio (supply/consumption): {abs(np.mean(vascular_base)/np.mean(reaction)):.2f}x")

dt = 1e-2
print(f"\nWith dt={dt}:")
print(f"  Consumption per step: {dt * np.mean(reaction):.6e}")
print(f"  Vascular supply per step: {dt * np.mean(vascular_base):.6e}")
print(f"  Net change: {dt * (np.mean(vascular_base) + np.mean(reaction)):.6e}")

print("\n=== PROBLEM ===")
print("Vascular supply is overwhelming consumption!")
print(f"Supply is {abs(np.mean(vascular_base)/np.mean(reaction)):.1f}x stronger than consumption")
print("\nSuggested fix: Scale down vascular supply by a factor")
suggested_scale = abs(np.mean(reaction) / np.mean(vascular_base)) * 2  # Make supply ~2x consumption
print(f"Recommended scaling factor: {suggested_scale:.6e}")

