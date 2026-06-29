# Simulation and Analysis of Proton Dynamics in a Cyclotron Model

This project presents a numerical simulation of the classical dynamics of a proton inside a cyclotron. It utilizes the **fourth-order Runge-Kutta (RK4) method** to solve the equations of motion under the influence of a radiofrequency (RF) electric field and a uniform magnetic field.

This development is part of my **2025A Modular Basic Training Cycle Integrative Work** evaluation for the Bachelor of Physics at **CUCEI, University of Guadalajara**.

## Context and Objectives
Proton therapy is a crucial application of particle physics in medicine, allowing beams to be directed at tumor tissues with high precision and minimizing damage to healthy tissues by leveraging the **Bragg peak**.

The main objectives of this study are:
* **Analyze** the trajectory and the evolution of a proton's kinetic energy in the presence of electromagnetic fields.
* **Compare** the analytical and numerical (RK4) approaches implemented in Python.
* **Evaluate** the model's capacity through exhaustive statistical analysis.

## Theoretical Framework
The particle dynamics are governed by the **Lorentz Force**:

$$\vec{F} = q (\vec{E} + \vec{v} \times \vec{B})$$

The cyclotron system is divided into two primary regions:
1. **Vacuum Region (Gap):** Where the electric field performs work on the particle, increasing its speed and kinetic energy.
2. **Dee Region:** Where a uniform magnetic field acts perpendicularly to the velocity, generating circular motion with a radius of curvature known as the **Larmor radius**.

## 🛠️ Simulation Parameters
Values and constants used for modeling:

| Parameter | Symbol | Value |
| :--- | :---: | :--- |
| Magnetic Field | $B_0$ | 2 T |
| RF Voltage | $V_0$ | 60 kV |
| Gap Separation | $d$ | 0.05 m |
| Initial Electric Field | $E_0$ | 1.2 x 10^6 V/m |
| Dee Radius | $R$ | 1 m |
| Proton Mass | $m$ | 1.672 x 10^-27 kg |
| Cyclotron Frequency | $\omega_c$ | 191.51 MHz |

## Project Structure
The code is organized into three main Python scripts:

* `src/NonRelativisticCyclotronSimulation.py`: Numerical simulation using **RK4** with a $10^{-12}$ s time step.
* `src/AnalyticalSolGraph.py`: Implementation of exact analytical parametric equations to obtain the theoretical trajectory.
* `src/CorrelationAndDataAnalysis.py`: Processing of results and statistical comparisons (correlation, errors, and offsets).

## Highlighted Results
* **Perfect Correlation:** A **1.0** correlation was achieved between the numerical and analytical simulations for time, position, and energy.
* **Trajectory:** The proton completes approximately **25.27 turns** before reaching the simulation radius limit, determined by the amount of data a single Excel sheet hold, for simplicity.
* **Synchronization:** Analysis determined that the RF voltage frequency must increase as the particle accelerates to compensate for its shorter residence time in the *gap*.
* **Precision:** The average phase offset detected was only **0.012 rad** (0.69°).

## Requirements
Environment used to run the simulations:
* **Language:** Python 3.9.13
* **IDE:** PyCharm Professional Edition 2022.2.1
* **Libraries:** NumPy, Pandas, and Matplotlib

## Authors
* **Jorge Ramírez López** - Development and Simulation - jorge.ramirez9331@alumnos.udg.mx
* **Advisor:** Mario Bolívar Gaeta Verdín, Ph.D.
* **Institution:** Physics Department, CUCEI, University of Guadalajara
