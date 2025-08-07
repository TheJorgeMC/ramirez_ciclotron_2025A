import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd  # Para guardar datos en Excel

matplotlib.use('TkAgg')

# Constantes
m = 1.672e-27  # Masa del protón (kg)
q = 1.6e-19  # Carga del protón (C)
V_o = 5e2  # Velocidad inicial en x (m/s)
V = 60e3  # Voltaje del campo eléctrico (V)
d = 0.05  # Separación entre los Dees (m)
E0 = V / d  # Campo eléctrico (V/m)
B = 2  # Campo magnético (T)
R = 1  # Radio de los Dees (m)

t_max = 1.048574e-6 # Tiempo maximo de simulacion
dt = 1e-12  # Paso temporal (s)

# Definir las regiones del ciclotrón
left_dee_theta = np.linspace(np.pi / 2, 3 * np.pi / 2, 100)
right_dee_theta = np.linspace(-np.pi / 2, np.pi / 2, 100)

x_left_dee = -d / 2 + R * np.cos(left_dee_theta)
y_left_dee = R * np.sin(left_dee_theta)

x_right_dee = d / 2 + R * np.cos(right_dee_theta)
y_right_dee = R * np.sin(right_dee_theta)

x_vacio = [-d / 2, d / 2, d / 2, -d / 2, -d / 2]
y_vacio = [-R, -R, R, R, -R]


# Campo eléctrico
def E_field_x(pos_x, t):
    E = np.zeros(3)
    if -d / 2 <= pos_x <= d / 2:
        E[0] = E0 * np.sign(v_x)
    return E[0]


# Campo magnético
def B_field(pos_x, pos_y):
    if pos_x < -d / 2 or pos_x > d / 2:
        return np.array([0, 0, B])
    return np.array([0, 0, 0])

# Tiempo
t = np.arange(0, t_max, dt)
t_prime = 0

# Listas para almacenar la trayectoria
times, total_x, total_y, kinetic_energies, E_values, region = [], [], [], [], [], []

# Condiciones iniciales
x, y = 0.0, 0.0
v_x, v_y = V_o, 0
x_o, y_o = 0, 0
v_xo, v_yo = V_o, 0
omega = 0

# Booleans para checar la posicion
prev_region = None
In_Vacio = True
In_Dee_Left = False
In_Dee_Right = False
nueva_region = ''
region_cambia = ''

for i in range(len(t)):
    # Determinar la región actual inmediatamente después de actualizar x
    if abs(x) < d / 2:
        nueva_region = "Vacio"
    elif x <= -d / 2:
        nueva_region = "Dee_1"
    else:
        nueva_region = "Dee_2"

    # Si hay un cambio de región, actualizar inmediatamente
    if nueva_region != prev_region:
        prev_region = nueva_region  # Actualizar la región previa

        # Actualizar variables de estado
        In_Vacio = (nueva_region == "Vacio")
        In_Dee_Left = (nueva_region == "Dee_1")
        In_Dee_Right = (nueva_region == "Dee_2")

        # Actualizar condiciones iniciales al cambiar de región
        t_prime = t[i]
        x_o, y_o = x, y
        v_xo, v_yo = v_x, v_y

    # Cálculo de trayectoria
    dt = t[i] - t_prime
    if In_Vacio:
        omega = q * E_field_x(x_o, t[i]) / m
        if omega != 0:
            x = 0.5 * omega * (dt ** 2) + v_xo * dt + x_o
            y = v_yo * dt + y_o
            v_x = omega * dt + v_xo
            v_y = v_yo
        else:
            print(f'ERROR: omega es cero en t = {t[i]} (campo eléctrico), evitando división por cero.')
            break
    elif In_Dee_Left or In_Dee_Right:
        omega = q * B_field(x_o, y_o)[2] / m
        if omega != 0:
            x = (v_xo / omega) * np.sin(omega * dt) + x_o
            y = (v_xo / omega) * (np.cos(omega * dt) - 1) + y_o
            v_x = v_xo * np.cos(omega * dt)
            v_y = -v_xo * np.sin(omega * dt)
        else:
            print(f'ERROR: omega es cero en t = {t[i]} (campo magnético), evitando división por cero.')
            break
    else:
        print(f'ERROR: La partícula no se detectó en ninguna región dentro del Ciclotrón')
        break

    # Almacenar valores
    total_x.append(x)
    total_y.append(y)
    K = 0.5 * m * (v_x ** 2 + v_y ** 2) / 1.60218e-13  # en MeV
    kinetic_energies.append(K)
    E = E_field_x(x, t[i])
    E_values.append(E)
    times.append(t[i])
    region.append(nueva_region)

# Exportar datos a Excel
data = {
    "Time (s)": times,
    "X Position (m)": total_x,
    "Y Position (m)": total_y,
    "Kinetic Energy (MeV)": kinetic_energies,
    "Electric Field (V/m)": E_values,
}
df = pd.DataFrame(data)
df.to_csv("graph_results.csv", index=False)

print("Datos exportados a 'graph_results.csv'.")

# Graficar la trayectoria
plt.figure(figsize=(8, 8))
plt.fill_between(x_left_dee, y_left_dee, color="blue", alpha=0.3, label="Área Dee Izquierdo")
plt.fill_between(x_right_dee, y_right_dee, color="orange", alpha=0.3, label="Área Dee Derecho")
plt.fill(x_vacio, y_vacio, color="purple", alpha=0.3, label="Región Vacía")
plt.plot(total_x, total_y, color='red', label="Trayectoria esperada del protón")

plt.xlabel("x [m]", fontsize=20)
plt.ylabel("y [m]", fontsize=20)
plt.title("Trayectoria Esperada de la Partícula en el Ciclotrón", fontsize=24)

plt.legend(fontsize=18)

plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

plt.axis("equal")

plt.show()
