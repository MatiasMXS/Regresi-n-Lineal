import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

# Datos
dataset = np.array([
    [3.5, 18],
    [3.69, 15],
    [3.44, 18],
    [3.43, 16],
    [4.34, 15],
    [4.42, 14],
    [2.37, 24],
])

# Configuración inicial
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.2)

ax_iterar = plt.axes([0.6, 0.05, 0.15, 0.075])
ax_final = plt.axes([0.8, 0.05, 0.15, 0.075])

btn_iterar = Button(ax_iterar, 'Iterar')
btn_final = Button(ax_final, 'Finalizar')

# Estado del modelo
estado = {
    'j': 0,
    'w': 0,
    'b': 0,
    'loss': float('inf')
}

# Hiperparámetros
tasa_aprendizaje = 0.01
limite_iteraciones = 20000
tolerancia = 1e-1  # cambia este valor si querés más precisión

# Función de entrenamiento de un paso
def paso_entrenamiento(graficar=True):
    if estado['j'] >= limite_iteraciones or estado['loss'] <= tolerancia:
        return False  # ya entrenado

    w = estado['w']
    b = estado['b']
    L2 = 0
    suma_w = 0
    suma_b = 0

    for x, y in dataset:
        y_pred = w * x + b
        error = y_pred - y
        L2 += error ** 2
        suma_w += 2 * error * x
        suma_b += 2 * error

    estado['loss'] = L2 / len(dataset)
    estado['w'] -= tasa_aprendizaje * (suma_w / len(dataset))
    estado['b'] -= tasa_aprendizaje * (suma_b / len(dataset))
    estado['j'] += 1

    print(f"Iteración {estado['j']}, Loss: {estado['loss']:.6f}")

    if graficar:
        graficar_regresion()
        plt.pause(0.001)

    return True  # puede seguir

# Entrena mostrando animación
def entrenar_con_grafico(event):
    while paso_entrenamiento(graficar=True):
        pass

# Entrena sin graficar hasta llegar a tolerancia
def entrenar_sin_grafico(event):
    while paso_entrenamiento(graficar=False):
        pass
    graficar_regresion()

# Dibuja los datos y la recta
def graficar_regresion():
    x_vals = dataset[:, 0]
    y_vals = dataset[:, 1]
    y_pred = estado['w'] * x_vals + estado['b']

    ax.clear()
    ax.scatter(x_vals, y_vals, color="blue", label="Datos reales")
    ax.plot(x_vals, y_pred, color="red", label=f"Modelo (epoch {estado['j']})")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Regresión Lineal")
    ax.text(0.05, 0.95, f"w = {estado['w']:.4f}", transform=ax.transAxes)
    ax.text(0.05, 0.90, f"b = {estado['b']:.4f}", transform=ax.transAxes)
    ax.text(0.05, 0.85, f"loss = {estado['loss']:.6f}", transform=ax.transAxes)

    ax.legend()
    plt.draw()

# Eventos de botones
btn_iterar.on_clicked(entrenar_con_grafico)
btn_final.on_clicked(entrenar_sin_grafico)

plt.show()
