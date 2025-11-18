# Clasificador de aprendizaje causal con DAGs

Este repositorio implementa un sistema de clasificación basado en **Modelos Gráficos Causales (DAGs)** que permite realizar inferencia causal y clasificación considerando las relaciones causales entre variables.

## Características principales

- **Modelos Causales**: Implementación de DAGs (Directed Acyclic Graphs) para representar relaciones causales
- **Clasificación Causal**: Algoritmo de clasificación que incorpora conocimiento causal del dominio
- **Análisis de D-Separación**: Verificación de independencias condicionales en el modelo causal
- **Visualización Interactiva**: Herramientas para visualizar la estructura causal y resultados
- **Ejemplos Prácticos**: Casos de uso en medicina y marketing digital

## Instalación

```bash
# Clonar el repositorio
git clone https://github.com/pedroam-dev/DAGs.git
cd DAGs

# Instalar dependencias
pip install -r requirements.txt
```

## Uso rápido

### Ejemplo básico: riesgo cardiovascular

```python
from causal_classifier import CausalDAG, CausalClassifier, generate_causal_data

# Generar datos sintéticos
X, y = generate_causal_data(n_samples=1000)

# Crear DAG causal
dag = CausalDAG(['edad', 'ejercicio', 'dieta', 'presion_arterial', 
                'colesterol', 'peso', 'riesgo_cardiovascular'])

# Definir relaciones causales
dag.add_edge('edad', 'presion_arterial', strength=0.8)
dag.add_edge('ejercicio', 'presion_arterial', strength=-0.9)
dag.add_edge('presion_arterial', 'riesgo_cardiovascular', strength=1.5)

# Entrenar clasificador causal
classifier = CausalClassifier(dag, 'riesgo_cardiovascular')
classifier.fit(X_train, y_train)

# Hacer predicciones
predictions = classifier.predict(X_test)
probabilities = classifier.predict_proba(X_test)
```

### Ejecutar demo completa

```bash
python causal_classifier.py
```

## Estructura del proyecto

```
DAGs/
├── causal_classifier.py    # Implementación principal del clasificador causal
├── marketing_example.py    # Ejemplo avanzado: análisis de marketing digital
├── requirements.txt        # Dependencias del proyecto
├── setup.py                # Script de instalación
└── README.md               # Esta documentación
```

## Conceptos fundamentales

### Modelos gráficos Causales (DAGs)

Un **DAG causal** es un grafo dirigido sin ciclos donde:
- **Nodos**: Representan variables del dominio
- **Aristas**: Representan relaciones causales directas
- **Ausencia de aristas**: Indica independencia causal condicional

### D-Separación

La **d-separación** es un criterio gráfico para determinar independencias condicionales:
- Si X e Y están d-separados dado Z en el DAG, entonces X ⊥ Y | Z
- Permite identificar qué variables son relevantes para la predicción

### Clasificación causal

El clasificador causal incorpora:
1. **Efectos causales directos**: Relaciones padre-hijo en el DAG
2. **Efectos causales indirectos**: Rutas causales a través de múltiples variables
3. **Características causales**: Features derivadas de la estructura causal

## Ejemplos incluidos

### 1. Riesgo Cardiovascular (Medicina)

**Variables**: edad, ejercicio, dieta, presión arterial, colesterol, peso
**Objetivo**: Predecir riesgo cardiovascular
**Relaciones causales**:
- Edad → Presión arterial, Colesterol
- Ejercicio → Presión arterial, Peso
- Dieta → Colesterol, Peso
- Factores de riesgo → Riesgo cardiovascular

### 2. Conversión de marketing (Negocios)

**Variables**: edad, ingresos, canal marketing, interés, exposición, engagement
**Objetivo**: Predecir conversión de clientes
**Relaciones causales**:
- Demografia → Interés en producto
- Canal marketing → Exposición
- Interés + Exposición → Engagement
- Interés + Exposición + Engagement → Conversión

## API principal

### CausalDAG

```python
# Crear DAG
dag = CausalDAG(variables=['A', 'B', 'C'])

# Añadir relaciones causales
dag.add_edge('A', 'B', strength=1.5)
dag.add_edge('B', 'C', strength=0.8)

# Análisis de independencias
is_independent = dag.is_d_separated('A', 'C', ['B'])

# Visualizar
dag.visualize()
```

### CausalClassifier

```python
# Crear clasificador
classifier = CausalClassifier(dag, target_variable='C')

# Entrenar
classifier.fit(X_train, y_train)

# Predecir
predictions = classifier.predict(X_test)
probabilities = classifier.predict_proba(X_test)

# Obtener importancia causal
importance = classifier.get_feature_importance()
```

## Ventajas del enfoque causal

1. **Interpretabilidad**: Las predicciones están basadas en relaciones causales conocidas
2. **Robustez**: Menor sensibilidad a cambios en la distribución de datos
3. **Insights accionables**: Identifica intervenciones que pueden cambiar el resultado
4. **Incorporación de conocimiento**: Utiliza expertise del dominio en el modelo
5. **Transferibilidad**: Mejor generalización a nuevos contextos

## Dependencias

- `numpy>=1.21.0`: Computación numérica
- `pandas>=1.3.0`: Manipulación de datos
- `networkx>=2.6.0`: Análisis de grafos
- `matplotlib>=3.4.0`: Visualización
- `seaborn>=0.11.0`: Visualización estadística
- `scipy>=1.7.0`: Computación científica
- `scikit-learn>=1.0.0`: Aprendizaje automático
- `graphviz>=0.17.0`: Visualización de grafos

## Algoritmos implementados

### 1. Construcción del DAG causal
- Verificación de aciclic en tiempo real
- Validación de relaciones causales
- Cálculo de efectos causales directos e indirectos

### 2. Clasificación causal
- Extracción de características causales
- Cálculo de scores causales basados en el DAG
- Umbralización adaptativa para clasificación

### 3. Análisis de independencias
- Implementación del criterio de d-separación
- Identificación de variables confusoras
- Análisis de rutas causales

## Casos de Uso

- **Medicina**: Diagnóstico médico, análisis de factores de riesgo
- **Marketing**: Análisis de conversión, optimización de campañas
- **Finanzas**: Análisis de riesgo crediticio, detección de fraude
- **Ciencias Sociales**: Análisis de políticas públicas, estudios causales
- **Industria**: Control de calidad, análisis de procesos

## Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. Fork el repositorio
2. Crea una rama para tu feature (`git checkout -b feature/nueva-funcionalidad`)
3. Commit tus cambios (`git commit -am 'Añade nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Crea un Pull Request

