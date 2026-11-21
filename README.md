# TP1 – Regresión Softmax para clasificación de texto en 20 Newsgroups

Trabajo práctico de Redes Neuronales. Implementa una **regresión softmax** (regresión logística multinomial: una única capa lineal + `CrossEntropyLoss`, sin capas ocultas) en PyTorch sobre vectores TF-IDF del dataset *20 Newsgroups* (~18.000 posts de Usenet, 20 categorías). El preprocesamiento de texto es configurable y todos sus hiperparámetros se buscan en conjunto con **Optuna** (con pruning) junto con los de entrenamiento. Se miden tiempos de convergencia, se aísla el efecto de cada hiperparámetro y se repiten los experimentos con varias semillas.

## Contenido del repositorio

| Archivo | Descripción |
|---|---|
| `01_Softmax_Regression_20Newsgroups.ipynb` | Notebook con todo el trabajo (código, experimentos, análisis y conclusiones), ejecutado de punta a punta. |
| `requirements.txt` | Dependencias de Python con las versiones usadas. |
| `README.md` | Este archivo. |
| `.gitignore` | Excluye archivos que no deben versionarse (`.DS_Store`, checkpoints de Jupyter, etc.). |
| `results/` | Resultados en CSV (trials, pruning, resúmenes por hiperparámetro, experimentos de un factor por vez, curvas por época, resultados finales por semilla, reporte por clase, confusiones, etc.), las bases de los estudios de Optuna (`optuna_*.db`) y las figuras del notebook en `results/figures/` (PNG numerados por orden de aparición). |
| `artefactos_softmax.zip` | Logs de TensorBoard, base de MLflow y modelos registrados. Ver [Artefactos de seguimiento](#artefactos-de-seguimiento-artefactos_softmaxzip). |

## Artefactos de seguimiento (`artefactos_softmax.zip`)

Los logs de TensorBoard, la base de MLflow y los modelos registrados son numerosos y pesados, por lo que se distribuyen comprimidos. El archivo lo genera la sección 14 del notebook y contiene:

| Contenido del zip | Descripción |
|---|---|
| `runs/` | Logs de TensorBoard: un directorio por trial de la búsqueda, por variante de los experimentos de la sección 9 y por entrenamiento final (uno por semilla). |
| `mlflow.db` | Base SQLite de MLflow con el experimento `softmax_20newsgroups`: hiperparámetros, métricas por época y tiempos de cada run, anidados bajo cada búsqueda. |
| `mlruns/` | Modelo registrado por MLflow en el entrenamiento final de la semilla principal. |
| `results/` | Copia de las tablas CSV, las figuras y las bases de Optuna. |
| `requirements.txt`, `.gitignore` | Copias de los archivos de la raíz. |

Para explorarlos, descomprimir el zip en la raíz del repositorio y ejecutar:

```bash
unzip artefactos_softmax.zip

# Curvas de aprendizaje de todos los entrenamientos
tensorboard --logdir runs

# Parámetros, métricas y tiempos de cada run
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Entorno de ejecución

- El notebook se ejecutó en **Google Colab con una GPU NVIDIA T4**.
- Todos los tiempos reportados (segundos por época, tiempo hasta converger, tiempo de búsqueda) se midieron con `time.perf_counter()` en esa GPU; en otro hardware los valores absolutos cambiarán, aunque no las comparaciones relativas.
- Funciona también en CPU, pero es **bastante más lento**.
- Referencia de tiempo en la T4: la búsqueda de 200 trials tomó unos 626 s y el entrenamiento final (12 épocas), unos 0.7 s.

## Cómo ejecutarlo

### Opción A: Google Colab

1. Subir `01_Softmax_Regression_20Newsgroups.ipynb` a Colab (o abrirlo desde Drive/GitHub).
2. Activar la GPU: *Entorno de ejecución → Cambiar tipo de entorno de ejecución → T4 GPU*.
3. Ejecutar todas las celdas en orden y en una sola sesión. La primera celda instala `optuna` y `mlflow`; el resto de las librerías ya viene en Colab.
4. Al final (sección 14) se genera `artefactos_softmax.zip` y Colab lo descarga automáticamente.

### Opción B: ejecución local

```bash
python -m venv .venv
source .venv/bin/activate        # En Windows: .venv\Scripts\activate

# Instalar PyTorch según el sistema/CUDA (https://pytorch.org/get-started/locally/)
pip install -r requirements.txt

jupyter notebook 01_Softmax_Regression_20Newsgroups.ipynb
```

La primera celda (`!pip install -q optuna mlflow`) puede omitirse si ya se instalaron los requirements.

### Notas

- El notebook descarga automáticamente el dataset (`fetch_20newsgroups`) y los recursos de NLTK (`stopwords`, `wordnet`, `omw-1.4`, `punkt`, `punkt_tab`), por lo que se necesita conexión a internet la primera vez.
- Las semillas están fijadas (`SEED = 42`, `SEEDS = [42, 123, 2024, 7, 99]` para los entrenamientos finales), pero los resultados pueden variar levemente entre ejecuciones por el no determinismo de la GPU.
- Los estudios de Optuna se guardan en `results/optuna_*.db` y **se reanudan** si la sesión se interrumpe. Para repetir la búsqueda desde cero hay que borrar esos archivos.
- El preprocesamiento (lo más lento) se precalcula una sola vez para las 6 combinaciones posibles de stopwords y normalización, y la vectorización TF-IDF se cachea; la primera ejecución de esa celda puede tardar varios minutos.

## Protocolo experimental

- El conjunto `train` oficial se divide una única vez en **entrenamiento (80 %) y validación (20 %)**, de forma estratificada. Esa partición se usa en la búsqueda, en las curvas de aprendizaje y para elegir la cantidad de épocas.
- El conjunto `test` oficial se usa **solo para la evaluación final**: no interviene en ninguna decisión ni se registra por época.
- El entrenamiento final usa todo el `train` durante la mejor época encontrada en validación y se evalúa una única vez sobre el test.
- El vocabulario y los pesos IDF se ajustan solo con el conjunto de entrenamiento.

## Contenido del notebook

| Sección | Qué hace |
|---|---|
| **0. Setup** | Instalación de dependencias, importaciones, semillas, detección de GPU y backend de MLflow. Tabla de herramientas utilizadas y por qué se eligió cada una. |
| **1. Datos** | Carga de las particiones oficiales de 20 Newsgroups (sin encabezados, firmas ni citas) y partición fija entrenamiento/validación. |
| **2. Preprocesamiento** | Tokenización configurable: minúsculas, stopwords, stemming o lematización. |
| **3. Vectorización** | TF-IDF con rango de n-gramas, `max_features` y `min_df` configurables. Una verificación comprueba que los tres hiperparámetros tienen efecto sobre el vocabulario. |
| **4. Modelo** | `SoftmaxRegression`: una única capa lineal que devuelve logits. |
| **5. Entrenamiento** | Bucle con Adam y `weight_decay`; tensores residentes en GPU, métricas por época, tiempos con `perf_counter`, época y tiempo de convergencia, early stopping y pruning. |
| **6. Búsqueda con Optuna** | 200 trials con muestreo TPE y `MedianPruner`. Espacio: `remove_stopwords`, `normalization`, `ngram_range`, `max_features`, `min_df`, `lr`, `weight_decay` y `batch_size`. |
| **7. Entrenamiento final** | Entrena la mejor configuración con todo el train y evalúa una sola vez en el test. |
| **8. Evaluación final** | Accuracy con intervalo de confianza (bootstrap), reporte por clase, matriz de confusión y confusiones más frecuentes. |
| **9. Dinámica del entrenamiento** | Curvas de la configuración óptima y experimentos de un factor por vez (3 semillas): learning rate, batch size, vocabulario, n-gramas y `weight_decay`. |
| **10. Análisis de la búsqueda** | Pruning y tiempo ahorrado, mejores trials, efecto por grupo, `lr` y `weight_decay`, historia de la búsqueda, chequeo de bordes e importancia de hiperparámetros (fANOVA). |
| **11. Robustez frente a las semillas** | Entrenamiento final con 5 semillas y repetición de la búsqueda con otras dos semillas del sampler. |
| **12. Preguntas de reflexión** | Respuestas apoyadas en los resultados del notebook. |
| **13. Conclusiones** | Resultados, búsqueda, efecto de cada hiperparámetro, dinámica y limitaciones. |
| **14. Exportación** | Guarda las tablas finales, genera `requirements.txt`, `.gitignore` y `artefactos_softmax.zip`. |

## Resultados principales

Evaluación en el test oficial (7.532 documentos), media ± desvío entre 5 semillas:

| Métrica | Valor |
|---|---|
| Accuracy en test | 0.6785 ± 0.0013 |
| Macro-F1 en test | 0.6676 ± 0.0019 |
| Accuracy de entrenamiento (modelo final) | 0.9663 |
| Mejor `val_acc` de la búsqueda | 0.7534 |
| Tiempo de entrenamiento final | 0.68 ± 0.04 s (12 épocas, 0.056 s/época) |

- **Mejor configuración (trial 98):** conserva las stopwords, aplica stemming, usa unigramas, `min_df = 2`, `max_features = 20.000`, `lr = 1.5·10⁻²`, `weight_decay = 1.9·10⁻⁷` y `batch_size = 256`, con 376.700 parámetros.
- **Alta varianza:** el error de entrenamiento es bajo y la brecha con el test es de 28.8 puntos. La `val_acc` alcanza su máximo en la época 10 y cae 1.9 puntos hasta la 30, mientras la `train_acc` sigue subiendo.
- **Learning rate:** es el hiperparámetro de mayor efecto (importancia fANOVA 0.69). Con ×0.01 del óptimo no converge en 30 épocas (`val_acc` 0.625) y con ×10 converge unas 4 veces más rápido pero 1.7 puntos peor.
- **Vocabulario y n-gramas:** pasar de 2.000 a 20.000 características sube la `val_acc` 10 puntos con ganancias decrecientes; los bigramas la bajan 4.5 puntos con el vocabulario fijo.
- **Batch size:** afecta al costo más que al rendimiento: con 512 el tiempo por época es unas 13 veces menor y el tiempo hasta converger 4.3 veces menor que con 32, con `val_acc` similar.
- **Regularización:** `weight_decay` no tiene efecto hasta 10⁻⁶; con valores mayores reduce el gap a costa de la `val_acc`.
- **Búsqueda:** el pruning cortó 71 de 200 trials (35.5 %). Los 10 mejores trials difieren en 0.09 puntos, menos que el error de muestreo de la validación (±0.91 puntos), y otras dos búsquedas con distinta semilla del sampler alcanzaron 0.7512 y 0.7548.
- **Errores más frecuentes:** se concentran en tres bloques temáticos: política (talk.politics.misc → talk.politics.guns, 89 documentos), religión y ateísmo, y hardware y sistemas de `comp.*`.
- Más detalle (curvas, tablas, confusiones por clase y limitaciones) en las secciones 9 a 13 del notebook.

## Limitaciones

- Una única partición de validación, fija, y el sesgo optimista de elegir el mejor de 200 trials no se cuantificó.
- Las búsquedas se limitan a 20 épocas y los experimentos controlados a 30, por lo que las variantes con `lr` bajo quedan limitadas por ese tope.
- Los experimentos de un factor por vez parten de una sola configuración y no capturan interacciones.
- Las semillas varían la inicialización y el orden de los lotes, pero no la partición ni el test.
- Todos los tiempos corresponden a una única GPU T4.