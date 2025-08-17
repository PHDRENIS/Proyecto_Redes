# PROYECTO DE REDES NEURONALES Y APRENDIZAJE PROFUNDO
## Entrenamiento de agentes basados en DQN tradicional y DQN resnet para juegos de Atari.

Este proyecto presenta un estudio comparativo entre las redes neuronales profundas (DQN) tradicionales y las DQN residuales (Res-DQN) en el aprendizaje por refuerzo profundo (DRL) para juegos de Atari. El objetivo principal es evaluar el rendimiento, la estabilidad del entrenamiento y las capacidades de generalización de ambas arquitecturas en tres juegos: Pong, Space Invaders y Qbert.

La arquitectura de la DQN tradicional se basa en la arquitectura convolucional estándar, mientras que la Res-DQN integra bloques residuales para mejorar el flujo de gradientes y reducir la ineficiencia de los parámetros. Ambos modelos utilizan hiperparámetros idénticos para asegurar una comparación justa, incluyendo la repetición de experiencias (experience replay), redes objetivo (target networks) y exploración ϵ-greedy. Las métricas clave del proyecto incluyen la convergencia de la recompensa, la eficiencia del entrenamiento y el costo computacional.

**Los resultados obtenidos respaldan la hipótesis inicial, demostrando que las Res-DQN superan a las DQN tradicionales en estabilidad y rendimiento en los tres entornos evaluados**. Las Res-DQN mostraron una convergencia más rápida y suave, con menores fluctuaciones en las recompensas por episodio. En juegos complejos como Space Invaders y Qbert, las Res-DQN mantuvieron promedios de recompensa más altos y una menor volatilidad. La implementación del proyecto utiliza PyTorch con reproducibilidad total, proporcionando un claro punto de referencia para las variantes de DQN en entornos complejos.

#### En este repositorio se encuentran los codigos de DQN tradicional y residual, además de modelos entrenados.

#### Tecpa Cisneros V.H. , Santos Mora R. A. , Leon Castro O.
