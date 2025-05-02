import matplotlib.pyplot as plt
import numpy as np
import re

# Datos extraídos
episodes = []
rewards = []
epsilons = []

data = """
*** Ejemplo de datos a ingresar aqui: ***
Episodio: 1, Recompensa: -21.0, Epsilon: 0.9999999999999999
"""

# splits los datos (ignorando frames y checkpoints)
for line in data.split('\n'):
    if 'Episodio:' in line:
        try:
            ep = int(re.search(r'Episodio: (\d+)', line).group(1))
            reward = float(re.search(r'Recompensa: ([-+]?\d+\.\d+)', line).group(1))
            epsilon = float(re.search(r'Epsilon: ([\d.]+)', line).group(1))
            
            episodes.append(ep)
            rewards.append(reward)
            epsilons.append(epsilon)
        except:
            continue  # Ignorar líneas con formato incorrecto

# Calcular promedio móvil (ventana de 10 episodios)
window_size = 10
moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
adjusted_episodes = episodes[window_size-1:]

# Configurar gráficos (solo 2 subplots ahora)
plt.figure(figsize=(15, 8))  # Ajusté el tamaño para mejor visualización

# gráfico de recompensas
plt.subplot(2, 1, 1)
plt.plot(episodes, rewards, 'b-', alpha=0.3, label='Recompensa por episodio')
plt.plot(adjusted_episodes, moving_avg, 'r-', linewidth=2, 
         label=f'Promedio móvil ({window_size} episodios)')
plt.title('Evolución del Aprendizaje - Pong')
plt.xlabel('Episodios')
plt.ylabel('Recompensa')
plt.grid(True, alpha=0.3)
plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
plt.legend()

# Grafico de Epsilon
plt.subplot(2, 1, 2)
plt.plot(episodes, epsilons, 'orange')
plt.title('Exploración (Epsilon)')
plt.xlabel('Episodios')
plt.ylabel('Valor de Epsilon')
plt.yscale('log')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()