# Clasificación de Leucemia mediante Redes Neuronales Convolucionales

## Descripción General

Este proyecto implementa y compara tres arquitecturas diferentes de redes neuronales convolucionales (CNN) para la clasificación de imágenes de células leucémicas. El objetivo principal es distinguir entre cuatro tipos celulares distintos: células Benignas (Benign), Early, Pre y Pro, correspondientes a diferentes estadios de la leucemia mediante análisis automatizado de imágenes.

## Dataset

El proyecto utiliza el conjunto de datos de Leucemia disponible a través de Kaggle, el cual contiene dos versiones de las imágenes:

- **Original**: Imágenes crudas de microscopía de células sanguíneas
- **Segmented**: Versiones preprocesadas y segmentadas de las mismas imágenes

El conjunto de datos se descarga automáticamente mediante la biblioteca `kagglehub` y comprende cuatro clases distintas que representan diferentes estadios y tipos de células leucémicas.

### División de Datos

El conjunto de datos se particiona de la siguiente manera:
- **Conjunto de entrenamiento**: 80% del total de datos
- **Conjunto de validación**: 10% del total de datos
- **Conjunto de prueba**: 10% del total de datos

Todas las imágenes se cargan con dimensiones de 224×224×3 (canales de color RGB) utilizando las utilidades `ImageFolder` y `DataLoader` de PyTorch.

## Arquitecturas de Modelos

### 1. CNN Básica (CNN)

La primera arquitectura implementa una red neuronal convolucional simple con la siguiente estructura:

- **Capa Convolucional 1**: 3 → 6 canales, kernel 5×5
- **MaxPooling**: 2×2, stride 2
- **Capa Convolucional 2**: 6 → 16 canales, kernel 3×3
- **MaxPooling**: 2×2, stride 2
- **Capa Completamente Conectada 1**: 46,656 → 128 neuronas
- **Capa Completamente Conectada 2**: 128 → 64 neuronas
- **Capa de Salida**: 64 → 4 clases

Este modelo base sirve como punto de referencia para evaluar arquitecturas más complejas.

### 2. CNN Dinámica (CNN2)

La segunda arquitectura introduce un diseño configurable que permite definiciones flexibles de capas mediante un parámetro de configuración. Características principales:

- Construcción dinámica de capas convolucionales basada en lista de configuración
- Cálculo automático de dimensiones de mapas de características
- Tres bloques convolucionales con complejidad creciente de filtros:
  - Bloque 1: 3 → 6 canales, kernel 5×5
  - Bloque 2: 6 → 12 canales, kernel 3×3
  - Bloque 3: 12 → 32 canales, kernel 3×3 con padding
- Capas completamente conectadas: FC(n) → 64 → 16 → 4

Esta arquitectura demuestra capacidades mejoradas de extracción de características mediante procesamiento convolucional más profundo.

### 3. Transfer Learning con ResNet-18 (ResNet18Wrapper)

El tercer enfoque aprovecha el transfer learning utilizando un modelo ResNet-18 preentrenado de torchvision. Detalles de implementación:

- Pesos preentrenados inicializados desde ImageNet
- Congelamiento opcional del extractor de características para escenarios de fine-tuning
- Capa completamente conectada personalizada adaptada para clasificación de 4 clases
- Conexiones residuales para mejorar el flujo de gradientes

Este modelo capitaliza representaciones aprendidas de conjuntos de datos de imágenes a gran escala para mejorar el rendimiento de clasificación.

## Configuración de Entrenamiento

### Hiperparámetros

```python
learning_rate = 0.001
batch_size = 64
num_epochs = 100
```

### Optimización

El framework soporta múltiples algoritmos de optimización:
1. Descenso de Gradiente Estocástico (SGD)
2. Adam
3. RMSprop
4. Adadelta

El optimizador predeterminado para todos los experimentos es Adam (opción 2).

### Función de Pérdida

Se emplea Cross-Entropy Loss como función objetivo, apropiada para tareas de clasificación multiclase.

## Métricas de Evaluación

El rendimiento del modelo se evalúa utilizando las siguientes métricas:

1. **Pérdida de Entrenamiento**: Monitoreada a través de las épocas para asegurar convergencia
2. **Exactitud de Validación**: Utilizada para selección de modelos y ajuste de hiperparámetros
3. **Exactitud de Prueba**: Métrica de rendimiento final sobre datos no vistos
4. **F1 Score (Macro)**: Media armónica de precisión y recall a través de todas las clases
5. **Matriz de Confusión**: Representación visual del rendimiento de clasificación por clase

## Detalles de Implementación

### Configuración de Dispositivo

El código detecta y utiliza automáticamente la aceleración por hardware disponible:
- GPU habilitada con CUDA si está disponible
- Alternativa de CPU para sistemas sin GPU

### Persistencia del Modelo

El modelo con mejor rendimiento (basado en exactitud de validación) se guarda durante el entrenamiento como `best_model.pth` para evaluación y despliegue posteriores.

### Visualización

El proyecto incluye utilidades de visualización comprehensivas:
- Curvas de pérdida de entrenamiento
- Progresión de exactitud de validación
- Mapas de calor de matriz de confusión
- Visualización de imágenes de muestra con etiquetas predichas

## Requisitos

### Dependencias Principales

```
torch
torchvision
numpy
matplotlib
kagglehub
scikit-learn
seaborn
```

### Instalación

```bash
pip install torch torchvision numpy matplotlib kagglehub scikit-learn seaborn
```

## Uso

### Ejecución Básica

```python
# Inicializar modelo
model = CNN(learning_rate, loss_function, optimizer, kernel_sizes).to(device)

# Entrenar modelo
train_losses, val_accuracies = train_model(model, train_loader, val_loader, num_epochs, device)

# Evaluar en conjunto de prueba
test_model(model, test_loader, device, train_losses, val_accuracies)
```

### Configuración de Arquitectura Personalizada

Para el modelo CNN dinámico:

```python
# Definir configuración de capas
# Formato: [canales_salida, tamaño_kernel, stride, padding]
config = [
    [6, 5, 1, 0],
    [12, 3, 1, 0],
    [32, 3, 1, 1]
]

# Crear modelo
model2 = CNN2(learning_rate, loss_function, optimizer, config, 
              input_size=224, num_classes=4).to(device)

# Entrenar
train_losses, val_accuracies = train_model(model2, train_loader, val_loader, 
                                          num_epochs, device)
```

### Transfer Learning

```python
# Crear modelo con pesos preentrenados
model3 = ResNet18Wrapper(learning_rate, loss_function, optimizer, 
                         num_classes=4, freeze_features=False).to(device)

# Entrenar
train_losses, val_accuracies = train_model(model3, train_loader, val_loader, 
                                          num_epochs, device)
```

## Estructura del Proyecto

```
.
├── leukemia_classification.py    # Script principal
├── best_model.pth                # Modelo guardado (generado durante entrenamiento)
└── README.md                     # Este archivo
```

## Funciones Auxiliares

### `get_optimizer(optimizer, model, lr)`
Selector de algoritmo de optimización que retorna el optimizador configurado según el parámetro especificado.

### `calculate_w(w, f, p, s)`
Calcula el tamaño de salida de una capa convolucional utilizando la fórmula:
```
W_out = (W_in - F + 2P) / S + 1
```
Donde W es el tamaño, F es el tamaño del filtro, P es el padding y S es el stride.

### `train_model(model, train_loader, val_loader, num_epochs, device)`
Función principal de entrenamiento que ejecuta el ciclo de entrenamiento-validación y guarda el mejor modelo.

### `test_model(model, test_loader, device, train_losses, val_accuracies)`
Evalúa el modelo en el conjunto de prueba y genera visualizaciones de rendimiento.

### `plot_confusion_matrix(model, test_loader, device, class_names)`
Genera y visualiza la matriz de confusión para el modelo evaluado.

## Resultados

Los resultados de cada modelo incluyen:
- Curvas de pérdida de entrenamiento por época
- Curvas de exactitud de validación por época
- Métricas finales en conjunto de prueba (pérdida, exactitud, F1 score)
- Matriz de confusión detallada por clase

## Consideraciones

- El entrenamiento de 100 épocas puede requerir tiempo considerable dependiendo del hardware disponible
- Se recomienda el uso de GPU para acelerar el entrenamiento
- Los modelos preentrenados (ResNet-18) requieren descarga inicial de pesos desde Internet
- El conjunto de datos se descarga automáticamente pero requiere credenciales de Kaggle configuradas

## Trabajo Futuro

Posibles extensiones del proyecto incluyen:
- Implementación de técnicas de aumento de datos (data augmentation)
- Exploración de arquitecturas más profundas (ResNet-50, EfficientNet)
- Optimización de hiperparámetros mediante búsqueda sistemática
- Implementación de técnicas de regularización avanzadas
- Análisis de interpretabilidad mediante mapas de activación de clase (CAM)

## Referencias

- Dataset: [Leukemia Classification Dataset - Kaggle](https://www.kaggle.com/datasets/mehradaria/leukemia)
- Framework: PyTorch
- Arquitectura de referencia: ResNet-18 (He et al., 2016)

## Autores
Emanuel González Quintero
Martín Valencia Vallejo
Nicolás Ospina Torres

Proyecto desarrollado como parte de un estudio de clasificación de leucemia mediante técnicas de aprendizaje profundo.

## Licencia

Este proyecto se proporciona con fines educativos y de investigación.
