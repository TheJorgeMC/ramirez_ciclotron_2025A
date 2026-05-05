# Simulación y Análisis de la Dinámica de un Protón en un Modelo de Ciclotrón

Este proyecto presenta una simulación numérica de la dinámica clásica de un protón dentro de un ciclotrón. Utiliza el método de **Runge-Kutta de cuarto orden (RK4)** para resolver las ecuaciones de movimiento bajo la influencia de un campo eléctrico de radiofrecuencia (RF) y un campo magnético uniforme.

El desarrollo forma parte de mi **Trabajo integrador de ciclo de formación modular básico 2025A** de la Licenciatura en Física en el **CUCEI, Universidad de Guadalajara**.

## Contexto y Objetivos
La terapia con protones es una aplicación crucial de la física de partículas en medicina, permitiendo dirigir haces hacia tejidos tumorales con alta precisión y minimizar el daño a tejidos sanos mediante el aprovechamiento del **pico de Bragg**.

Los objetivos principales de este estudio son:
* **Analizar** la trayectoria y la evolución de la energía cinética de un protón en presencia de campos electromagnéticos.
* **Comparar** los enfoques analíticos y numéricos (RK4) implementados en Python.
* **Evaluar** la capacidad del modelo mediante un análisis estadístico exhaustivo.

## Marco Teórico
La dinámica de la partícula está regida por la **Fuerza de Lorentz**:

$$\vec{F} = q (\vec{E} + \vec{v} \times \vec{B})$$

El sistema del ciclotrón se divide en dos regiones principales:
1. **Región de vacío (Gap):** Donde el campo eléctrico realiza trabajo sobre la partícula, incrementando su rapidez y energía cinética.
2. **Región de los Dees:** Donde actúa un campo magnético uniforme de forma perpendicular a la velocidad, generando un movimiento circular con un radio de curvatura conocido como **radio de Larmor**.

## 🛠️ Parámetros de Simulación
Valores y constantes utilizados para la modelación:

| Parámetro | Símbolo | Valor |
| :--- | :---: | :--- |
| Campo Magnético | $B_0$ | 2 T |
| Voltaje de RF | $V_0$ | 60 kV |
| Separación entre Dees | $d$ | 0.05 m |
| Campo Eléctrico Inicial | $E_0$ | 1.2 x 10^6 V/m |
| Radio de los Dees | $R$ | 1 m |
| Masa del protón | $m$ | 1.672 x 10^-27 kg |
| Frecuencia Ciclotrónica | $\omega_c$ | 191.51 MHz |

## Estructura del Proyecto
El código está organizado en tres scripts principales de Python:

* `src/NonRelativisticCyclotronSimulation.py`: Simulación numérica mediante **RK4** con paso de tiempo de $10^{-12}$ s.
* `src/AnalyticalSolGraph.py`: Implementación de las ecuaciones analíticas paramétricas para obtener la trayectoria teórica exacta.
* `src/CorrelationAndDataAnalysis.py`: Procesamiento de resultados y comparaciones estadísticas (correlación, errores y desfases).

## Resultados Destacados
* **Correlación Perfecta:** Se obtuvo una correlación de **1.0** entre la simulación numérica y la analítica en tiempo, posición y energía.
* **Trayectoria:** El protón completa aproximadamente **25.27 vueltas** antes de alcanzar el límite del radio de diseño.
* **Sincronización:** El análisis determinó que la frecuencia del voltaje de RF debe incrementarse conforme la partícula acelera para compensar su menor tiempo de permanencia en el *gap*.
* **Precisión:** El desfase de fase promedio detectado fue de apenas **0.012 rad** (0.69°).

## Requerimientos
Entorno utilizado para ejecutar las simulaciones:
* **Lenguaje:** Python 3.9.13
* **IDE:** PyCharm Professional Edition 2022.2.1
* **Librerías:** NumPy, Pandas y Matplotlib

## Autores
* **Jorge Ramírez López** - Desarrollo y Simulación - jorge.ramirez9331@alumnos.udg.mx
* **Asesor:** Mario Bolívar Gaeta Verdín
* **Institución:** Departamento de Física, CUCEI, Universidad de Guadalajara
