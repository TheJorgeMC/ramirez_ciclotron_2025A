import numpy as np
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')

# Constantes
B = 2  # Campo magnético inicial en Tesla
V_o = 5e2
V = 60e3  # Voltaje en kV
d = 0.05  # Separación entre los Dees en metros
E0 = V / d  # Campo eléctrico
R = 1  # Radio del área del Dee

# Propiedades del protón
proton = {
    'q': 1.6e-19,  # Carga en coulombs
    'm': 1.672e-27,  # Masa del protón en kg
    'p': np.array([1.672e-27*V_o, 0.0, 0.0]),  # Pequeño impulso inicial
    'position': np.array([0.0, 0.0, 0.0])
}

# Áreas para gráficos
left_theta = np.linspace(np.pi / 2, 3 * np.pi / 2, 100)
x1 = -d / 2 + R * np.cos(left_theta)
y1 = R * np.sin(left_theta)

right_theta = np.linspace(-np.pi / 2, np.pi / 2, 100)
x2 = d / 2 + R * np.cos(right_theta)
y2 = R * np.sin(right_theta)

x_vacio = [-d / 2, d / 2, d / 2, -d / 2, -d / 2]
y_vacio = [-R, -R, R, R, -R]


# Campo eléctrico (oscilatorio)
def E_field(pos_x, pos_y, pos_z, t):
    E = np.zeros(3)
    if -d / 2 <= pos_x <= d / 2 and -R <= pos_y <= R:
        E[0] = E0 * np.sign(proton['p'][0])
    return E[0]


# Campo magnético
def B_field(pos_x, pos_y):
    if pos_x < -d / 2 or pos_x > d / 2:
        return np.array([0, 0, B])
    return np.array([0, 0, 0])


# RK4 Método
def rk4_step(position, momentum, h, t):
    def velocity(momentum):
        return momentum / proton['m']

    def acceleration(position, momentum, t):
        E = np.array([E_field(*position, t), 0, 0])
        B = B_field(position[0], position[1])
        v = velocity(momentum)
        return proton['q'] * (E + np.cross(v, B))

    k1_p = h * acceleration(position, momentum, t)
    k1_x = h * velocity(momentum)

    k2_p = h * acceleration(position + 0.5 * k1_x, momentum + 0.5 * k1_p, t + 0.5 * h)
    k2_x = h * velocity(momentum + 0.5 * k1_p)

    k3_p = h * acceleration(position + 0.5 * k2_x, momentum + 0.5 * k2_p, t + 0.5 * h)
    k3_x = h * velocity(momentum + 0.5 * k2_p)

    k4_p = h * acceleration(position + k3_x, momentum + k3_p, t + h)
    k4_x = h * velocity(momentum + k3_p)

    new_position = position + (k1_x + 2 * k2_x + 2 * k3_x + k4_x) / 6
    new_momentum = momentum + (k1_p + 2 * k2_p + 2 * k3_p + k4_p) / 6

    return new_position, new_momentum


# Simulación
h = 1e-12
steps = 1048575
trajectory_x, trajectory_y, times, E_values, kinetic_energies = [], [], [], [], []
time = 0

for _ in range(steps):
    trajectory_x.append(proton['position'][0])
    trajectory_y.append(proton['position'][1])
    times.append(time)

    E = E_field(*proton['position'], time)
    E_values.append(E)

    kinetic_energy = 0.5 * proton['m'] * np.linalg.norm(proton['p'] / proton['m']) ** 2 / 1.60218e-13  # en MeV
    kinetic_energies.append(kinetic_energy)

    proton['position'], proton['p'] = rk4_step(proton['position'], proton['p'], h, time)
    time += h

# Exportar datos a Excel
data = {
    "Time (s)": times,
    "X Position (m)": trajectory_x,
    "Y Position (m)": trajectory_y,
    "Kinetic Energy (MeV)": kinetic_energies,
    "Electric Field (V/m)": E_values
}
df = pd.DataFrame(data)
df.to_csv("simulation_results.csv", index=False)

# Gráfica trayectoria
plt.figure(figsize=(8, 8))
plt.fill_between(x1, y1, color='blue', alpha=0.3, label="Dee Izquierdo")
plt.fill_between(x2, y2, color='orange', alpha=0.3, label='Dee Derecho')
plt.fill(x_vacio, y_vacio, color="purple", alpha=0.3, label="Región Vacía")
plt.plot(trajectory_x, trajectory_y, label='Trayectoria del protón', color='red')

plt.xlabel('Posición X [m]', fontsize=20)  # Aumentamos a 20
plt.ylabel('Posición Y [m]', fontsize=20)  # Aumentamos a 20
plt.title('Trayectoria del protón', fontsize=24)  # Aumentamos a 24

plt.axis('equal')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.xticks(fontsize=18)  # Aumentamos a 18
plt.yticks(fontsize=18)  # Aumentamos a 18

plt.legend(fontsize=18)  # Aumentamos a 18

# Gráfica campo eléctrico
E_min = min(E_values)
E_max = max(E_values)
plt.figure(figsize=(10, 5))
plt.plot(times, E_values, label='Campo eléctrico [V/m]', color='green')
plt.axhline(y=E_min, color='blue', linestyle='--', label=f'Mínimo: {E_min:.2e} [V/m]')
plt.axhline(y=E_max, color='red', linestyle='--', label=f'Máximo: {E_max:.2e} [V/m]')

plt.xlabel('Tiempo [s]', fontsize=20)  # Aumentamos a 20
plt.ylabel('Campo eléctrico [V/m]', fontsize=20)  # Aumentamos a 20
plt.title('Campo eléctrico en función del tiempo', fontsize=24)  # Aumentamos a 24

plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.xticks(fontsize=18)  # Aumentamos a 18
plt.yticks(fontsize=18)  # Aumentamos a 18

plt.legend(fontsize=18)  # Aumentamos a 18

# Gráfica energía cinética
plt.figure(figsize=(10, 5))
plt.plot(times, kinetic_energies, label='Energía Cinética [MeV]', color='blue')
plt.axhline(y=min(kinetic_energies), color='purple', linestyle='--', label=f'Energía Cinética Inicial: {min(kinetic_energies):.2e} [MeV]')

plt.xlabel('Tiempo [s]', fontsize=24)  # Aumentamos a 20
plt.ylabel('Energía Cinética [MeV]', fontsize=24)  # Aumentamos a 20
plt.title('Energía Cinética en función del tiempo', fontsize=28)  # Aumentamos a 24

plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.xticks(fontsize=20)  # Aumentamos a 18
plt.yticks(fontsize=20)  # Aumentamos a 18

plt.legend(fontsize=20)  # Aumentamos a 18

plt.show()
