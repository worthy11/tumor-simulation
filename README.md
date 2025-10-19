# Simulating _in vitro_ glioblastoma growth based on studies conducted on mice

## Equations

- Species concentration (O2, CO2, VEGF, MMP, ECM):
- Parameters: dc*i/dt + nabla * r_f * u*ins \* c_i = D_i nabla^2 c_i + R_i + S_i

![alt text](image.png)

![alt text](image-1.png)

- d_i - particle size of solute
- d_p - pore size of porous media
- delta_di = (1-(1-d_i/d_p)^2)^2
- delta_v - average oncotic reflection coefficient of plasma proteis
- d_v - angiogenic neo-vessel diameter
- d_c - neo-vessel characteristic diameter
- S_v - surface area per unit volume for transvascular exchange
- p_lum - intravascular blood pressure
- p_ins - interstitial fluid pressure (IFP)
- pi_lum - oncotic pressure of intravascular plasma
- pi_ins - oncotic pressure of interstitial fluid
- k_i - permeability coefficient of neo-vessel wall
- L_p - hydraulic conductivity of neo-vessel wall
- P_i - transvascular Peclet number (ratio of convection to diffusion across neo-vessel wall) = (1-delta_di)(L_p/k_i(p_lum-p_ins-delta_v(pi_lum-pi_ins)))
-
