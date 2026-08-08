<div style="text-align:center; font-family:Arial;">

  <p align="center">
    <img src="Logo-Escuela-ingeniería.png" alt="Logo Escuela de Ingeniería" width="150"/>
  </p>

  <h2>ESCUELA COLOMBIANA DE INGENIERÍA JULIO GARAVITO</h2>
  <h3>UNIVERSIDAD</h3>
  <p><strong>VIGILADA MINEDUCACIÓN</strong></p>

  <h3>Departamento de Matemáticas</h3>
  <h2>Guía de trabajo 1: Campo de pendientes</h2>
  <p><strong>Competencias:</strong> R2, RM2, CM2, SP2, C3 - 2026</p>
  <p><strong>Integrante:</strong> Juan Miguel Nope Ascencio</p>
  <p><strong>Integrante:</strong> Sebastian Lopez Rincon</p>

</div>


# Guía de trabajo 1: Campo de pendientes


# Ejercicio 1 - Campos de pendientes y soluciones particulares

Usando un asistente computacional, se graficaron los campos de pendientes y las soluciones particulares de las siguientes ecuaciones diferenciales:

---

## a) \( y' = -y - \sin(x), \; y(0)=1 \)

<p align="center">
  <img src="Grafica-a.jpeg" alt="Gráfica ejercicio 1-a" width="400"/>
</p>

---

## b) \( y' = x + y, \; y(-2)=2 \)

<p align="center">
  <img src="Grafica-b.jpeg" alt="Gráfica ejercicio 1-b" width="400"/>
</p>

---

## c) \( y' = -x^2 + \sin(y) \)

<p align="center">
  <img src="Grafica-c.jpeg" alt="Gráfica ejercicio 1-c" width="400"/>
</p>


---

## d) \( (x^2+1)y' + 3xy = 6x \)


<p align="center">
  <img src="Grafica-d.jpeg" alt="Gráfica ejercicio 1-d" width="400"/>
</p>

---

## e) \( y' = x e^y \)

<p align="center">
  <img src="Grafica-e.jpeg" alt="Gráfica ejercicio 1-e" width="400"/>
</p>

---

## f) \( y' = x - y, \; y(1)=1 \)


<p align="center">
  <img src="Grafica-f.jpeg" alt="Gráfica ejercicio 1-f" width="400"/>
</p>


```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def graficar_edo(
    f,
    x_range=(-3, 3),
    y_range=(-3, 3),
    cond_inicial=None,
    titulo="",
    familia=None,
    n=23
):
    """
    Grafica:
    - Campo de pendientes con flechas
    - Familia de soluciones, si se proporciona
    - Solución particular, si se proporciona condición inicial
    """


    x_vals = np.linspace(x_range[0], x_range[1], n)
    y_vals = np.linspace(y_range[0], y_range[1], n)

    X, Y = np.meshgrid(x_vals, y_vals)

    with np.errstate(
        divide="ignore",
        invalid="ignore",
        over="ignore"
    ):
        M = f(X, Y)

    M = np.where(np.isfinite(M), M, np.nan)

    U = np.ones_like(M)
    V = M

    # Normalización
    N = np.sqrt(U**2 + V**2)

    U = U / N
    V = V / N

    plt.figure(figsize=(10, 7))


    # Campo de pendientes
    plt.quiver(
        X, Y,
        U, V,

        angles="xy",
        scale_units="xy",
        scale=8,

        width=0.0025,

        headwidth=4,
        headlength=5,
        headaxislength=4,

        color="black"
    )

    if familia is not None:

        familia(
            f,
            x_range,
            y_range
        )

    if cond_inicial is not None:

        x0, y0 = cond_inicial

        x_der = np.linspace(
            x0,
            x_range[1],
            600
        )

        sol_der = solve_ivp(
            f,
            [x0, x_range[1]],
            [y0],
            t_eval=x_der,
            rtol=1e-9,
            atol=1e-11
        )


        # -----------------------------
        # Hacia la izquierda
        # -----------------------------

        x_izq = np.linspace(
            x0,
            x_range[0],
            600
        )

        sol_izq = solve_ivp(
            f,
            [x0, x_range[0]],
            [y0],
            t_eval=x_izq,
            rtol=1e-9,
            atol=1e-11
        )


        plt.plot(
            sol_der.t,
            sol_der.y[0],
            linewidth=3,
            color="red",
            label=f"Solución particular y({x0})={y0}"
        )

        plt.plot(
            sol_izq.t,
            sol_izq.y[0],
            linewidth=3,
            color="red"
        )


        # Punto inicial

        plt.scatter(
            x0,
            y0,
            color="red",
            s=50,
            zorder=5
        )

    plt.xlim(x_range)
    plt.ylim(y_range)

    plt.xlabel("x")
    plt.ylabel("y")

    plt.title(titulo)

    plt.axhline(
        0,
        linewidth=0.8,
        alpha=0.5
    )

    plt.axvline(
        0,
        linewidth=0.8,
        alpha=0.5
    )

    plt.grid(
        True,
        alpha=0.25
    )

    if cond_inicial is not None:
        plt.legend()

    plt.tight_layout()
    plt.show()

graficar_edo(
    lambda x, y: -y - np.sin(x),

    x_range=(-3, 3),
    y_range=(-3, 3),

    cond_inicial=(0, 1),

    titulo="a) y' = -y - sin(x)"
)

graficar_edo(
    lambda x, y: x + y,

    x_range=(-3, 3),
    y_range=(-5, 5),

    cond_inicial=(-2, 2),

    titulo="b) y' = x + y"
)
graficar_edo(
    lambda x, y: -x**2 + np.sin(y),

    x_range=(-3, 3),
    y_range=(-3, 3),

    cond_inicial=(0, 0),

    titulo="c) y' = -x² + sin(y)"
)

graficar_edo(
    lambda x, y: (6*x - 3*x*y) / (x**2 + 1),

    x_range=(-3, 3),
    y_range=(-3, 3),

    cond_inicial=(0, 1),

    titulo="d) (x² + 1)y' + 3xy = 6x"
)

graficar_edo(
    lambda x, y: x * np.exp(y),

    x_range=(-2, 2),
    y_range=(-3, 3),

    cond_inicial=(0, 0),

    titulo="e) y' = x e^y"
)

graficar_edo(
    lambda x, y: x - y,

    x_range=(-3, 3),
    y_range=(-3, 3),

    cond_inicial=(1, 1),

    titulo="f) y' = x - y"
)

```



## Punto 2: Campos de Direcciones y Estabilidad

Análisis de estabilidad para las ecuaciones diferenciales autónomas:

![Análisis de Puntos Críticos y Campos de Direcciones del Punto 2](Figure_1.png)

### Puntos de Equilibrio:
* **a) $\frac{dy}{dx} = y(3-y)(y-2)$** $\rightarrow$ $y = 3$ (Estable), $y = 2$ (Inestable), $y = 0$ (Estable)
* **b) $\frac{dy}{dx} = y^2 - y^3$** $\rightarrow$ $y = 1$ (Estable), $y = 0$ (Semiestable)
* **c) $\frac{dy}{dx} = (y+2)(10+3y-y^2)$** $\rightarrow$ $y = 5$ (Estable), $y = -2$ (Semiestable)
* **d) $\frac{dy}{dx} = y^5 - 4y^3 - 5y^2$** $\rightarrow$ $y \approx 2.79$ (Inestable), $y = 0$ (Semiestable), $y = -1$ (Inestable), $y \approx -1.79$ (Estable)
* **e) $\frac{dy}{dx} = (1-y)(y-2)^3$** $\rightarrow$ $y = 2$ (Estable), $y = 1$ (Inestable)

```python
import sys, site


sys.path.append(site.getusersitepackages())

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


# 1. Definición de las Ecuaciones Diferenciales Autónomas

def eq_a(y, x): return y * (3 - y) * (y - 2)
def eq_b(y, x): return y**2 - y**3
def eq_c(y, x): return (y + 2) * (10 + 3*y - y**2)
def eq_d(y, x): return y**5 - 4*y**3 - 5*y**2
def eq_e(y, x): return (1 - y) * (y - 2)**3

# Configuración de ecuaciones con sus Puntos Críticos y Estabilidad
# Formato: (Título, Función, Límites Y, Puntos Críticos [(y_val, tipo, color)])
ecuaciones = [
    ("a) dy/dx = y(3-y)(y-2)", eq_a, (-1, 4), [
        (3.0, "Estable", "green"),
        (2.0, "Inestable", "red"),    
        (0.0, "Estable", "green")
    ]),
    ("b) dy/dx = y² - y³", eq_b, (-1, 2), [
        (1.0, "Estable", "green"),
        (0.0, "Semiestable", "orange")
    ]),
    ("c) dy/dx = (y+2)(10+3y-y²)", eq_c, (-4, 7), [
        (5.0, "Estable", "green"),
        (-2.0, "Semiestable", "orange")
    ]),
    ("d) dy/dx = y⁵ - 4y³ - 5y²", eq_d, (-2, 3.5), [
        (2.791, "Inestable", "red"),
        (0.0, "Semiestable", "orange"),
        (-1.0, "Inestable", "red"),
        (-1.791, "Estable", "green")
    ]),
    ("e) dy/dx = (1-y)(y-2)³", eq_e, (0, 3.2), [
        (2.0, "Estable", "green"),
        (1.0, "Inestable", "red")
    ])
]


# 2. Generación del Campo de Direcciones y Curvas Solución

fig, axes = plt.subplots(3, 2, figsize=(13, 15))
axes = axes.flatten()

x = np.linspace(-3, 3, 20)
x_sol = np.linspace(-3, 3, 200)

for idx, (titulo, f, y_lim, puntos_criticos) in enumerate(ecuaciones):
    ax = axes[idx]
    
    # Malla para el campo de direcciones
    y = np.linspace(y_lim[0], y_lim[1], 20)
    X, Y = np.meshgrid(x, y)
    
    # Campo de vectores (quiver)
    dy = f(Y, X)
    dx = np.ones(dy.shape)
    modulo = np.sqrt(dx**2 + dy**2)
    U = dx / modulo
    V = dy / modulo
    ax.quiver(X, Y, U, V, color='lightgray', alpha=0.7)
    
    # 1. Graficar trayectorias solución dinámicas
    y0_vals = np.linspace(y_lim[0] + 0.1, y_lim[1] - 0.1, 8)
    for y0 in y0_vals:
        sol = odeint(f, y0, x_sol)
        ax.plot(x_sol, sol, color='steelblue', alpha=0.5, linewidth=1.2)

    # 2. ️ Marcar explícitamente los PUNTOS DE EQUILIBRIO y su ESTABILIDAD
    for yc, tipo, color in puntos_criticos:
        # Trazar la solución exacta de equilibrio (línea continua de color)
        sol_eq = odeint(f, yc, x_sol)
        ax.plot(x_sol, sol_eq, color=color, linewidth=2.5, linestyle='--')
        
        #  Añadir etiqueta explicativa de estabilidad en el gráfico
        ax.text(1.2, yc + (y_lim[1]-y_lim[0])*0.02, f"y={yc:g} ({tipo})", 
                color=color, fontweight='bold', fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=color, alpha=0.8))

    ax.set_xlim(-3, 3)
    ax.set_ylim(y_lim)
    ax.set_title(titulo, fontweight='bold', fontsize=11)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid(True, linestyle=':', alpha=0.6)

# Ocultar la última subfigura vacía
fig.delaxes(axes[5])

plt.tight_layout()
plt.savefig('Figure_1.png', dpi=300)
plt.show()
```



# Punto 3 - Modelo Poblacional

Sea \( P(t) \) la población de cierta especie en un parque natural, con \( t \) tiempo en años y \( P \) en miles.  
La ecuación diferencial:



\[
\frac{dP}{dt} = P(P - 1)(2 - P)
\]



describe la tasa de cambio de la población de la especie en el instante \( t \).


## a) Diagrama de fase

El diagrama de fase muestra los puntos de equilibrio en \( P=0, P=1, P=2 \).  
- \( P=0 \) y \( P=2 \) son **estables**.  
- \( P=1 \) es **inestable**.  

<p align="center">
  <img src="fase.png" alt="Diagrama de fase" width="400"/>
</p>

---

## b) Población inicial de 3000 ejemplares

La población decrece y se estabiliza en **2000 ejemplares**.

<p align="center">
  <img src="poblacion_3000.png" alt="Población inicial 3000" width="400"/>
</p>

---

## c) Población inicial de 1500 ejemplares

La población crece y se estabiliza en **2000 ejemplares**.

<p align="center">
  <img src="poblacion_1500.png" alt="Población inicial 1500" width="400"/>
</p>

---

## d) Población inicial de 500 ejemplares

La población decrece hasta la **extinción**.

<p align="center">
  <img src="poblacion_500.png" alt="Población inicial 500" width="400"/>
</p>

---

## e) Población inicial de 900 ejemplares

La población está por debajo del umbral crítico \( P=1 \).  
No puede crecer hasta 1100, en cambio tiende a **0 ejemplares**.

<p align="center">
  <img src="poblacion_900.png" alt="Población inicial 900" width="400"/>
</p>

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Ecuación diferencial del modelo poblacional
def dP_dt(t, P):
    return P * (P - 1) * (2 - P)

# a) Diagrama de fase
def diagrama_fase():
    P_vals = np.linspace(-0.5, 2.5, 200)
    dP_vals = dP_dt(0, P_vals)

    plt.figure(figsize=(8,5))
    plt.axhline(0, color="black", linewidth=0.8)
    plt.plot(P_vals, dP_vals, label="dP/dt vs P")

    # Puntos de equilibrio
    for eq in [0, 1, 2]:
        plt.plot(eq, 0, "ro")
        plt.text(eq, 0.15, f"P={eq}", ha="center")

    plt.title("Diagrama de fase: dP/dt = P(P-1)(2-P)")
    plt.xlabel("Población (miles)")
    plt.ylabel("dP/dt")
    plt.grid(True)
    plt.legend()
    plt.savefig("fase.png")   # guarda imagen
    plt.close()

# Función para simular evolución de la población
def simular(P0, nombre, t_max=20):
    sol = solve_ivp(dP_dt, [0, t_max], [P0], t_eval=np.linspace(0, t_max, 300))
    plt.figure(figsize=(8,5))
    plt.plot(sol.t, sol.y[0], label=f"P(0)={P0*1000:.0f} ejemplares")
    plt.title(f"Evolución poblacional (P0={P0*1000:.0f})")
    plt.xlabel("Tiempo (años)")
    plt.ylabel("Población (miles)")
    plt.grid(True)
    plt.legend()
    plt.savefig(nombre)   # guarda imagen
    plt.close()

if __name__ == "__main__":
    # a) Diagrama de fase
    diagrama_fase()

    # b) Población inicial 3000 ejemplares
    simular(3, "poblacion_3000.png")

    # c) Población inicial 1500 ejemplares
    simular(1.5, "poblacion_1500.png")

    # d) Población inicial 500 ejemplares
    simular(0.5, "poblacion_500.png")

    # e) Población inicial 900 ejemplares
    simular(0.9, "poblacion_900.png")

```


## Punto 4: Modelo Poblacional 

Análisis numérico y campo de direcciones del modelo de crecimiento poblacional:

![Análisis punto 4](Figure_2.png)








### a) 
Para entender la dinámica de la población, primero tomamos la ecuación diferencial autónoma $\frac{dP}{dt} = P(3 - 2P)$ e igualamos la derivada a cero para encontrar los puntos de equilibrio:

$$P(3 - 2P) = 0 \implies P = 0 \quad \text{y} \quad P = 1.5$$

Como la variable $P$ está expresada en miles, $P = 1.5$ equivale a 1500 ejemplares. Evaluando el signo de la derivada en la región físicamente válida ($P \ge 0$), observamos lo siguiente:

* En el intervalo $(0, 1.5)$: La derivada es positiva ($\frac{dP}{dt} > 0$), lo que indica que la población crece de forma continua.
* En el intervalo $(1.5, \infty)$: La derivada es negativa ($\frac{dP}{dt} < 0$), por lo que la población comienza a decrecer.

Con esto clasificamos los puntos críticos: $P = 0$ funciona como un punto inestable (repulsor), mientras que $P = 1.5$ es un punto asintóticamente estable (atractor). Esto nos confirma que cualquier población inicial positiva ($P_0 > 0$) terminará acercándose con el tiempo a los 1500 ejemplares.



### b) 
Si comenzamos con 2000 ejemplares, tenemos la condición inicial $P(0) = 2.0$. Como este valor es mayor al punto de equilibrio ($2.0 > 1.5$), nos ubicamos en la zona donde la tasa de cambio es negativa. 

Por lo tanto, la población empezará a reducirse de forma constante hasta estabilizarse en los 1500 ejemplares:

$$\lim_{t \to \infty} P(t) = 1.5 \quad \text{(1500 ejemplares)}$$



### c) 
Con una población inicial de 100 especímenes ($P(0) = 0.1$), nos encontramos en el intervalo $(0, 1.5)$, donde la derivada es positiva. 

En este caso ocurre lo contrario al punto anterior: la población comenzará a multiplicarse y crecer hasta alcanzar nuevamente la capacidad de equilibrio en los 1500 ejemplares:

$$\lim_{t \to \infty} P(t) = 1.5 \quad \text{(1500 ejemplares)}$$



### d) 
Cuando la población llega a 1500 ejemplares ($P = 1.5$), alcanzamos la solución de equilibrio estable. En este valor la tasa de variación es exactamente cero ($\frac{dP}{dt} = 0$), lo que significa que el número de nacimientos y muertes se compensa por completo. La población dejará de cambiar y se mantendrá constante de forma indefinida ($P(t) = 1.5$ para todo $t \ge 0$).



### e) 
Para llevar las tasas trimestrales a una escala anual, multiplicamos los datos por $4$ trimestres:

* Nacimientos al año: $4 \times 0.150 = 0.600$ miles/año.
* Muertes al año: $4 \times \frac{s}{1000} = \frac{s}{250}$ miles/año.

A partir de esto, podemos plantear el modelo de dos formas según el enfoque:

1. Modelo directo por tasa neta constante:
   $$\frac{dP}{dt} = 0.6 - \frac{s}{250}$$

2. Modelo logístico modificado (sumando el ajuste de tasas):
   $$\frac{dP}{dt} = 3P - 2P^2 + 0.6 - \frac{s}{250}$$



### f)
Analizando la constante de crecimiento $k = 0.6 - \frac{s}{250}$ del modelo directo, el futuro de la población dependerá del valor de $s$:

* Si $s < 150$ muertes/trimestre: Los nacimientos superan a las muertes ($k > 0$), por lo que la población crecerá de forma lineal sin límite hacia el infinito.
* Si $s = 150$ muertes/trimestre: Nacimientos y muertes se igualan ($k = 0$), manteniendo la población estancada en su nivel inicial $P_0$.
* Si $s > 150$ muertes/trimestre: La mortalidad supera a la natalidad ($k < 0$), provocando que la población decrezca progresivamente hasta extinguirse ($P(t) \to 0$).

```python
import sys, site


sys.path.append(site.getusersitepackages())

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.integrate import solve_ivp

# Definicion del modelo poblacional dP/dt = 3P - 2P^2
def dP_dt(t, P):
    return 3*P - 2*P**2

# Configuracion de la figura
fig, ax = plt.subplots(figsize=(9, 6))

t_span = (0, 3)
t_eval = np.linspace(0, 3, 200)

# Graficar campo de pendientes
t_grid = np.linspace(0, 3, 15)
P_grid = np.linspace(0, 2.5, 15)
T, P_m = np.meshgrid(t_grid, P_grid)
SLOPE = dP_dt(0, P_m)
NORM = np.sqrt(1 + SLOPE**2)
ax.quiver(T, P_m, 1/NORM, SLOPE/NORM, color='lightgray', alpha=0.8)

# Condiciones iniciales de los incisos b, c y d
inits = {
    'b) P(0) = 2.0 (2000 ej.)': (2.0, 'crimson'),
    'c) P(0) = 0.1 (100 ej.)': (0.1, 'darkgreen'),
    'd) P(0) = 1.5 (1500 ej. - Equilibrio)': (1.5, 'navy')
}

for label, (P0, color) in inits.items():
    sol = solve_ivp(dP_dt, t_span, [P0], t_eval=t_eval)
    ax.plot(sol.t, sol.y[0], label=label, color=color, linewidth=2.5)

# Linea de equilibrio P = 1.5
ax.axhline(1.5, color='black', linestyle='--', alpha=0.7, label='Equilibrio P = 1.5 (1500 ej.)')

ax.set_title('Comportamiento Poblacional: dP/dt = 3P - 2P²', fontsize=12, fontweight='bold')
ax.set_xlabel('Tiempo t (años)')
ax.set_ylabel('Población P (en miles)')
ax.set_ylim(-0.1, 2.5)
ax.legend(loc='lower right')
ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('Figure_2.png', dpi=300)
plt.show()
```

