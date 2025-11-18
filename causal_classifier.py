"""
Clasificador de Aprendizaje Causal usando DAGs (Directed Acyclic Graphs)
========================================================================

Este módulo implementa un sistema de clasificación basado en modelos gráficos causales,
permitiendo el análisis de relaciones causales entre variables y la inferencia causal.

Autor: Pedro AM
Fecha: Noviembre 2025
"""

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class CausalDAG:
    """
    Clase para representar un Grafo Dirigido Acíclico (DAG) causal.
    
    Esta clase permite construir, manipular y analizar modelos causales
    representados como DAGs, donde los nodos son variables y las aristas
    representan relaciones causales directas.
    """
    
    def __init__(self, variables: List[str]):
        """
        Inicializa el DAG causal.
        
        Args:
            variables: Lista de nombres de variables del modelo causal
        """
        self.variables = variables
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(variables)
        self.structural_equations = {}
        
    def add_edge(self, cause: str, effect: str, strength: float = 1.0):
        """
        Añade una arista causal al DAG.
        
        Args:
            cause: Variable causa
            effect: Variable efecto
            strength: Fuerza de la relación causal (default: 1.0)
        """
        if cause not in self.variables or effect not in self.variables:
            raise ValueError("Las variables deben estar en la lista de variables del DAG")
            
        self.graph.add_edge(cause, effect, weight=strength)
        
        # Verificar que sigue siendo acíclico
        if not nx.is_directed_acyclic_graph(self.graph):
            self.graph.remove_edge(cause, effect)
            raise ValueError("Añadir esta arista crearía un ciclo en el grafo")
    
    def get_parents(self, variable: str) -> List[str]:
        """Obtiene los padres (causas directas) de una variable."""
        return list(self.graph.predecessors(variable))
    
    def get_children(self, variable: str) -> List[str]:
        """Obtiene los hijos (efectos directos) de una variable."""
        return list(self.graph.successors(variable))
    
    def get_ancestors(self, variable: str) -> List[str]:
        """Obtiene todos los ancestros (causas indirectas) de una variable."""
        return list(nx.ancestors(self.graph, variable))
    
    def get_descendants(self, variable: str) -> List[str]:
        """Obtiene todos los descendientes (efectos indirectos) de una variable."""
        return list(nx.descendants(self.graph, variable))
    
    def is_d_separated(self, X: str, Y: str, Z: List[str] = None) -> bool:
        """
        Verifica si X e Y están d-separados dado Z.
        
        La d-separación es un criterio fundamental para determinar
        independencia condicional en DAGs causales.
        """
        if Z is None:
            Z = []
        
        return nx.d_separated(self.graph, {X}, {Y}, set(Z))
    
    def visualize(self, figsize: Tuple[int, int] = (12, 8), 
                 node_color: str = 'lightblue', 
                 edge_color: str = 'gray',
                 title: str = "DAG Causal"):
        """
        Visualiza el DAG causal.
        
        Args:
            figsize: Tamaño de la figura
            node_color: Color de los nodos
            edge_color: Color de las aristas
            title: Título del gráfico
        """
        plt.figure(figsize=figsize)
        
        # Calcular posiciones usando layout jerárquico
        try:
            pos = nx.spring_layout(self.graph, k=3, iterations=50)
        except:
            pos = nx.circular_layout(self.graph)
        
        # Dibujar nodos
        nx.draw_networkx_nodes(self.graph, pos, 
                              node_color=node_color, 
                              node_size=2000, 
                              alpha=0.8)
        
        # Dibujar aristas
        nx.draw_networkx_edges(self.graph, pos, 
                              edge_color=edge_color, 
                              arrows=True, 
                              arrowsize=20,
                              arrowstyle='->')
        
        # Dibujar etiquetas
        nx.draw_networkx_labels(self.graph, pos, 
                               font_size=10, 
                               font_weight='bold')
        
        # Añadir pesos de las aristas
        edge_labels = nx.get_edge_attributes(self.graph, 'weight')
        if edge_labels:
            nx.draw_networkx_edge_labels(self.graph, pos, edge_labels, 
                                        font_size=8)
        
        plt.title(title, size=14, weight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.show()


class CausalClassifier:
    """
    Clasificador basado en modelos gráficos causales.
    
    Este clasificador utiliza un DAG causal para realizar clasificación
    considerando las relaciones causales entre las variables.
    """
    
    def __init__(self, dag: CausalDAG, target_variable: str):
        """
        Inicializa el clasificador causal.
        
        Args:
            dag: DAG causal que define las relaciones entre variables
            target_variable: Variable objetivo a predecir
        """
        self.dag = dag
        self.target_variable = target_variable
        self.feature_variables = [v for v in dag.variables if v != target_variable]
        self.is_fitted = False
        self.scaler = StandardScaler()
        
        # Verificar que la variable objetivo esté en el DAG
        if target_variable not in dag.variables:
            raise ValueError(f"La variable objetivo '{target_variable}' no está en el DAG")
    
    def _calculate_causal_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula características causales basadas en la estructura del DAG.
        
        Args:
            X: DataFrame con las variables de entrada
            
        Returns:
            DataFrame con características causales añadidas
        """
        X_causal = X.copy()
        
        # Añadir características basadas en relaciones causales
        for var in self.feature_variables:
            parents = self.dag.get_parents(var)
            
            if parents and all(p in X.columns for p in parents):
                # Característica: promedio ponderado de causas directas
                parent_values = X[parents].values
                weights = [self.dag.graph[p][var].get('weight', 1.0) for p in parents]
                weighted_avg = np.average(parent_values, weights=weights, axis=1)
                X_causal[f'{var}_causal_avg'] = weighted_avg
                
                # Característica: interacción causal
                if len(parents) >= 2:
                    interaction = np.prod(parent_values, axis=1)
                    X_causal[f'{var}_causal_interaction'] = interaction
        
        return X_causal
    
    def _calculate_causal_score(self, X: pd.DataFrame) -> np.ndarray:
        """
        Calcula un score causal para cada instancia basado en el DAG.
        
        Args:
            X: DataFrame con las variables de entrada
            
        Returns:
            Array con scores causales
        """
        scores = np.zeros(len(X))
        
        # Obtener padres (causas directas) de la variable objetivo
        target_parents = self.dag.get_parents(self.target_variable)
        
        if target_parents:
            for parent in target_parents:
                if parent in X.columns:
                    weight = self.dag.graph[parent][self.target_variable].get('weight', 1.0)
                    scores += weight * X[parent].values
        
        # Considerar efectos indirectos (ancestros)
        target_ancestors = self.dag.get_ancestors(self.target_variable)
        
        for ancestor in target_ancestors:
            if ancestor in X.columns:
                # Calcular el efecto causal indirecto
                paths = list(nx.all_simple_paths(self.dag.graph, ancestor, self.target_variable))
                for path in paths:
                    path_effect = 1.0
                    for i in range(len(path) - 1):
                        edge_weight = self.dag.graph[path[i]][path[i+1]].get('weight', 1.0)
                        path_effect *= edge_weight
                    
                    # Aplicar efecto con decaimiento por distancia
                    distance_decay = 0.8 ** (len(path) - 2)
                    scores += distance_decay * path_effect * X[ancestor].values
        
        return scores
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Entrena el clasificador causal.
        
        Args:
            X: DataFrame con las variables de entrada
            y: Series con la variable objetivo
        """
        # Verificar que todas las variables del DAG estén presentes
        missing_vars = set(self.feature_variables) - set(X.columns)
        if missing_vars:
            raise ValueError(f"Variables faltantes en los datos: {missing_vars}")
        
        # Calcular características causales
        X_causal = self._calculate_causal_features(X)
        
        # Normalizar características
        self.scaler.fit(X_causal)
        X_scaled = pd.DataFrame(
            self.scaler.transform(X_causal),
            columns=X_causal.columns,
            index=X_causal.index
        )
        
        # Calcular scores causales
        self.causal_scores = self._calculate_causal_score(X_scaled)
        
        # Calcular umbral de decisión basado en la distribución de scores
        positive_scores = self.causal_scores[y == 1]
        negative_scores = self.causal_scores[y == 0]
        
        if len(positive_scores) > 0 and len(negative_scores) > 0:
            self.threshold = (np.mean(positive_scores) + np.mean(negative_scores)) / 2
        else:
            self.threshold = np.median(self.causal_scores)
        
        self.is_fitted = True
        
        # Guardar estadísticas de entrenamiento
        self.training_stats = {
            'positive_score_mean': np.mean(positive_scores) if len(positive_scores) > 0 else 0,
            'negative_score_mean': np.mean(negative_scores) if len(negative_scores) > 0 else 0,
            'score_std': np.std(self.causal_scores),
            'threshold': self.threshold
        }
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Realiza predicciones usando el clasificador causal.
        
        Args:
            X: DataFrame con las variables de entrada
            
        Returns:
            Array con las predicciones
        """
        if not self.is_fitted:
            raise ValueError("El clasificador debe ser entrenado antes de hacer predicciones")
        
        # Calcular características causales
        X_causal = self._calculate_causal_features(X)
        
        # Normalizar características
        X_scaled = pd.DataFrame(
            self.scaler.transform(X_causal),
            columns=X_causal.columns,
            index=X_causal.index
        )
        
        # Calcular scores causales
        scores = self._calculate_causal_score(X_scaled)
        
        # Hacer predicciones basadas en el umbral
        predictions = (scores > self.threshold).astype(int)
        
        return predictions
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Calcula probabilidades de clase usando el clasificador causal.
        
        Args:
            X: DataFrame con las variables de entrada
            
        Returns:
            Array con las probabilidades de cada clase
        """
        if not self.is_fitted:
            raise ValueError("El clasificador debe ser entrenado antes de hacer predicciones")
        
        # Calcular características causales
        X_causal = self._calculate_causal_features(X)
        
        # Normalizar características
        X_scaled = pd.DataFrame(
            self.scaler.transform(X_causal),
            columns=X_causal.columns,
            index=X_causal.index
        )
        
        # Calcular scores causales
        scores = self._calculate_causal_score(X_scaled)
        
        # Convertir scores a probabilidades usando función sigmoide
        probabilities = 1 / (1 + np.exp(-(scores - self.threshold)))
        
        # Retornar probabilidades para ambas clases
        prob_matrix = np.column_stack([1 - probabilities, probabilities])
        
        return prob_matrix
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Calcula la importancia de las características basada en la estructura causal.
        
        Returns:
            Diccionario con la importancia de cada característica
        """
        if not self.is_fitted:
            raise ValueError("El clasificador debe ser entrenado primero")
        
        importance = {}
        
        # Importancia basada en conexión directa con el objetivo
        target_parents = self.dag.get_parents(self.target_variable)
        for parent in target_parents:
            if parent in self.feature_variables:
                weight = self.dag.graph[parent][self.target_variable].get('weight', 1.0)
                importance[parent] = abs(weight)
        
        # Importancia basada en conexión indirecta
        target_ancestors = self.dag.get_ancestors(self.target_variable)
        for ancestor in target_ancestors:
            if ancestor in self.feature_variables:
                paths = list(nx.all_simple_paths(self.dag.graph, ancestor, self.target_variable))
                total_effect = 0
                for path in paths:
                    path_effect = 1.0
                    for i in range(len(path) - 1):
                        edge_weight = self.dag.graph[path[i]][path[i+1]].get('weight', 1.0)
                        path_effect *= edge_weight
                    distance_decay = 0.8 ** (len(path) - 2)
                    total_effect += distance_decay * abs(path_effect)
                
                importance[ancestor] = total_effect
        
        # Normalizar importancias
        if importance:
            max_importance = max(importance.values())
            importance = {k: v / max_importance for k, v in importance.items()}
        
        return importance


def generate_causal_data(n_samples: int = 1000, noise_level: float = 0.1) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Genera datos sintéticos que siguen un modelo causal específico.
    
    Este conjunto de datos simula un escenario médico donde:
    - Edad afecta a Presión Arterial y Colesterol
    - Ejercicio afecta a Presión Arterial y Peso
    - Dieta afecta a Colesterol y Peso
    - Presión Arterial, Colesterol y Peso afectan al Riesgo Cardiovascular
    
    Args:
        n_samples: Número de muestras a generar
        noise_level: Nivel de ruido en las relaciones causales
        
    Returns:
        Tuple con DataFrame de características y Series objetivo
    """
    np.random.seed(42)  # Para reproducibilidad
    
    # Variables exógenas (no causadas por otras variables en el modelo)
    edad = np.random.normal(50, 15, n_samples)
    ejercicio = np.random.binomial(1, 0.3, n_samples)  # 30% hace ejercicio regularmente
    dieta = np.random.binomial(1, 0.4, n_samples)      # 40% tiene buena dieta
    
    # Variables endógenas (causadas por otras variables)
    # Presión Arterial: aumenta con edad, disminuye con ejercicio
    presion_arterial = (0.8 * edad + 120 - 15 * ejercicio + 
                       np.random.normal(0, noise_level * 10, n_samples))
    
    # Colesterol: aumenta con edad, disminuye con buena dieta
    colesterol = (1.2 * edad + 150 - 30 * dieta + 
                 np.random.normal(0, noise_level * 20, n_samples))
    
    # Peso: disminuye con ejercicio y buena dieta
    peso = (70 + 0.3 * edad - 8 * ejercicio - 12 * dieta + 
           np.random.normal(0, noise_level * 5, n_samples))
    
    # Riesgo Cardiovascular: función de presión arterial, colesterol y peso
    riesgo_score = (0.02 * presion_arterial + 0.01 * colesterol + 0.05 * peso - 15 +
                   np.random.normal(0, noise_level, n_samples))
    
    # Convertir a variable binaria (1 = alto riesgo, 0 = bajo riesgo)
    riesgo_cardiovascular = (riesgo_score > 0).astype(int)
    
    # Crear DataFrame
    data = pd.DataFrame({
        'edad': edad,
        'ejercicio': ejercicio,
        'dieta': dieta,
        'presion_arterial': presion_arterial,
        'colesterol': colesterol,
        'peso': peso
    })
    
    target = pd.Series(riesgo_cardiovascular, name='riesgo_cardiovascular')
    
    return data, target


def create_medical_dag() -> CausalDAG:
    """
    Crea un DAG causal para el ejemplo médico de riesgo cardiovascular.
    
    Returns:
        DAG causal configurado con las relaciones causales médicas
    """
    variables = ['edad', 'ejercicio', 'dieta', 'presion_arterial', 
                'colesterol', 'peso', 'riesgo_cardiovascular']
    
    dag = CausalDAG(variables)
    
    # Relaciones causales del modelo médico
    # Edad afecta presión arterial y colesterol
    dag.add_edge('edad', 'presion_arterial', strength=0.8)
    dag.add_edge('edad', 'colesterol', strength=1.2)
    dag.add_edge('edad', 'peso', strength=0.3)
    
    # Ejercicio afecta presión arterial y peso
    dag.add_edge('ejercicio', 'presion_arterial', strength=-0.9)
    dag.add_edge('ejercicio', 'peso', strength=-0.8)
    
    # Dieta afecta colesterol y peso
    dag.add_edge('dieta', 'colesterol', strength=-1.0)
    dag.add_edge('dieta', 'peso', strength=-0.7)
    
    # Factores de riesgo afectan riesgo cardiovascular
    dag.add_edge('presion_arterial', 'riesgo_cardiovascular', strength=1.5)
    dag.add_edge('colesterol', 'riesgo_cardiovascular', strength=1.2)
    dag.add_edge('peso', 'riesgo_cardiovascular', strength=1.0)
    
    return dag


def analyze_causal_effects(dag: CausalDAG, data: pd.DataFrame, target: str):
    """
    Analiza los efectos causales en el modelo.
    
    Args:
        dag: DAG causal
        data: Datos para el análisis
        target: Variable objetivo
    """
    print("=== ANÁLISIS DE EFECTOS CAUSALES ===\n")
    
    # Análisis de independencias condicionales
    print("1. INDEPENDENCIAS CONDICIONALES:")
    print("   (basadas en d-separación)\n")
    
    variables = [v for v in dag.variables if v != target]
    
    for i, var1 in enumerate(variables):
        for var2 in variables[i+1:]:
            # Verificar independencia incondicional
            is_independent = dag.is_d_separated(var1, var2, [])
            status = "INDEPENDIENTES" if is_independent else "DEPENDIENTES"
            print(f"   {var1} ⊥ {var2} : {status}")
            
            # Verificar independencia condicional dada la variable objetivo
            is_cond_independent = dag.is_d_separated(var1, var2, [target])
            status_cond = "INDEPENDIENTES" if is_cond_independent else "DEPENDIENTES"
            print(f"   {var1} ⊥ {var2} | {target} : {status_cond}")
            print()
    
    # Análisis de correlaciones en los datos
    print("2. CORRELACIONES OBSERVADAS EN LOS DATOS:\n")
    correlations = data.corr()
    
    # Mostrar correlaciones significativas
    threshold = 0.3
    for i, var1 in enumerate(data.columns):
        for var2 in data.columns[i+1:]:
            corr = correlations.loc[var1, var2]
            if abs(corr) > threshold:
                print(f"   {var1} ↔ {var2}: {corr:.3f}")
    
    print()
    
    # Análisis de efectos causales directos e indirectos
    print("3. EFECTOS CAUSALES HACIA EL OBJETIVO:\n")
    
    target_parents = dag.get_parents(target)
    print(f"   Causas directas de {target}:")
    for parent in target_parents:
        weight = dag.graph[parent][target]['weight']
        print(f"   • {parent} → {target}: {weight:.2f}")
    
    print()
    
    target_ancestors = dag.get_ancestors(target)
    print(f"   Causas indirectas de {target}:")
    for ancestor in target_ancestors:
        if ancestor not in target_parents:
            paths = list(nx.all_simple_paths(dag.graph, ancestor, target))
            print(f"   • {ancestor} → {target}:")
            for path in paths:
                path_str = " → ".join(path)
                path_effect = 1.0
                for i in range(len(path) - 1):
                    edge_weight = dag.graph[path[i]][path[i+1]]['weight']
                    path_effect *= edge_weight
                print(f"     Ruta: {path_str} (efecto total: {path_effect:.3f})")
    
    print()


if __name__ == "__main__":
    print("CLASIFICADOR DE APRENDIZAJE CAUSAL - DEMO")
    print("=" * 60)
    print()
    
    # Generar datos sintéticos
    print("Generando datos sintéticos del modelo médico...")
    X, y = generate_causal_data(n_samples=1000, noise_level=0.1)
    print(f"Generadas {len(X)} muestras con {len(X.columns)} variables")
    print()
    
    # Crear DAG causal
    print("Construyendo DAG causal...")
    dag = create_medical_dag()
    print(f"DAG creado con {len(dag.variables)} variables y {len(dag.graph.edges)} aristas causales")
    print()
    
    # Visualizar el DAG
    print("Visualizando estructura causal...")
    dag.visualize(title="Modelo Causal: Riesgo Cardiovascular")
    
    # Análisis causal
    analyze_causal_effects(dag, X, 'riesgo_cardiovascular')
    
    # Dividir datos para entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"División de datos:")
    print(f"   • Entrenamiento: {len(X_train)} muestras")
    print(f"   • Prueba: {len(X_test)} muestras")
    print()
    
    # Crear y entrenar clasificador causal
    print("Entrenando clasificador causal...")
    classifier = CausalClassifier(dag, 'riesgo_cardiovascular')
    classifier.fit(X_train, y_train)
    print("Clasificador entrenado exitosamente")
    print()
    
    # Realizar predicciones
    print("Realizando predicciones...")
    y_pred = classifier.predict(X_test)
    y_proba = classifier.predict_proba(X_test)
    
    # Evaluar rendimiento
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Precisión del clasificador causal: {accuracy:.3f}")
    print()
    
    # Mostrar reporte detallado
    print("REPORTE DE CLASIFICACIÓN:")
    print(classification_report(y_test, y_pred, 
                              target_names=['Bajo Riesgo', 'Alto Riesgo']))
    
    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Bajo Riesgo', 'Alto Riesgo'],
                yticklabels=['Bajo Riesgo', 'Alto Riesgo'])
    plt.title('Matriz de Confusión - Clasificador Causal')
    plt.ylabel('Etiqueta Real')
    plt.xlabel('Predicción')
    plt.show()
    
    # Importancia de características causales
    print("\nIMPORTANCIA DE CARACTERÍSTICAS CAUSALES:")
    importance = classifier.get_feature_importance()
    for feature, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
        print(f"   • {feature}: {imp:.3f}")
    
    print()
    print("Demo completada exitosamente!")
    print("\nEste clasificador demuestra cómo los modelos causales pueden")
    print("mejorar la interpretabilidad y robustez de las predicciones")
    print("al incorporar conocimiento del dominio sobre relaciones causales.")