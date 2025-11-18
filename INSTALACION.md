# Instrucciones de Instalación y Ejecución

## Instalación Rápida

### Requisitos
- Python 3.7 o superior
- pip (gestor de paquetes de Python)

### Paso 1: Clonar o descargar el repositorio
```bash
git clone https://github.com/pedroam-dev/DAGs.git
cd DAGs
```

### Paso 2: Instalar dependencias
```bash
pip install -r requirements.txt
```

**Nota**: Si tienes problemas con la instalación, puedes instalar las dependencias individualmente:
```bash
pip install numpy pandas networkx matplotlib seaborn scipy scikit-learn graphviz
```

### Paso 3: Verificar instalación
```bash
python test_quick.py
```

Si ves el mensaje "TODAS LAS PRUEBAS PASARON EXITOSAMENTE", la instalación fue correcta.

## 📖 Ejemplos Disponibles

### 1. Ejemplo Mínimo (Recomendado para empezar)
```bash
python ejemplo_minimo.py
```
**Qué hace**: Introduce conceptos básicos con un ejemplo muy simple (lluvia → paraguas → llegar seco)

### 2. Demo Principal - Riesgo Cardiovascular
```bash
python causal_classifier.py
```
**Qué hace**: Ejemplo médico completo con visualizaciones y análisis causal detallado

### 3. Ejemplo Avanzado - Marketing Digital
```bash
python marketing_example.py
```
**Qué hace**: Análisis de conversión de clientes comparando métodos causales vs tradicionales

### 4. Tutorial Interactivo
```bash
python tutorial.py
```
**Qué hace**: Tutorial paso a paso que explica todos los conceptos teóricos

## Solución de Problemas

### Error: ModuleNotFoundError
**Problema**: `ModuleNotFoundError: No module named 'xxx'`

**Solución**:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Error: Matplotlib/Visualización
**Problema**: Errores con matplotlib o visualizaciones no aparecen

**Solución macOS**:
```bash
pip install matplotlib
# Si usas conda:
conda install matplotlib
```

**Solución Linux**:
```bash
sudo apt-get install python3-tk
pip install matplotlib
```

### Error: GraphViz
**Problema**: `graphviz.backend.ExecutableNotFound`

**Solución macOS**:
```bash
brew install graphviz
pip install graphviz
```

**Solución Linux**:
```bash
sudo apt-get install graphviz
pip install graphviz
```

### Ejecutar sin visualizaciones
Si tienes problemas con las visualizaciones, puedes ejecutar solo el test básico:
```bash
python test_quick.py
```

## Orden Recomendado de Ejecución

Para el mejor aprendizaje, ejecuta los ejemplos en este orden:

1. **Verificación**: `python test_quick.py`
2. **Concepto básico**: `python ejemplo_minimo.py`
3. **Tutorial**: `python tutorial.py` (opcional, para teoría)
4. **Demo principal**: `python causal_classifier.py`
5. **Ejemplo avanzado**: `python marketing_example.py`

## Ejecución en Diferentes Entornos

### Jupyter Notebook
```python
# Copia cualquier archivo .py completo en una celda y ejecuta
exec(open('ejemplo_minimo.py').read())
```

### Google Colab
```python
# Primero instala dependencias
!pip install networkx matplotlib seaborn scipy scikit-learn

# Luego copia y ejecuta el código
```

### Entorno Virtual (Recomendado)
```bash
# Crear entorno virtual
python -m venv venv_causales

# Activar entorno
# En macOS/Linux:
source venv_causales/bin/activate
# En Windows:
venv_causales\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar ejemplos
python ejemplo_minimo.py
```

## ¿Necesitas Ayuda?

Si tienes problemas:

1. Verifica que Python ≥ 3.7 esté instalado: `python --version`
2. Actualiza pip: `pip install --upgrade pip`
3. Reinstala dependencias: `pip install -r requirements.txt --force-reinstall`
4. Ejecuta el test: `python test_quick.py`

## Todo Listo
Una vez que `test_quick.py` ejecute sin errores, ¡todo está funcionando correctamente y puedes disfrutar explorando los clasificadores causales!