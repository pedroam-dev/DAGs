"""
Tutorial Interactivo: Introducción a los Modelos Causales
========================================================

Este tutorial te guiará paso a paso a través de los conceptos fundamentales
de los modelos gráficos causales y cómo implementar un clasificador causal.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from causal_classifier import CausalDAG, CausalClassifier


def tutorial_conceptos_basicos():
    """
    Parte 1: Introducción a los conceptos básicos de modelos causales
    """
    print("🎓 TUTORIAL: CONCEPTOS BÁSICOS DE MODELOS CAUSALES")
    print("=" * 60)
    print()
    
    print("¿Qué es un Modelo Causal?")
    print("-" * 40)
    print("Un modelo causal es una representación matemática que describe")
    print("las relaciones de causa y efecto entre variables. A diferencia de")
    print("los modelos de correlación, los modelos causales nos permiten:")
    print()
    print("• Entender PORQUÉ ocurre algo, no solo CUÁNDO")
    print("• Predecir el efecto de intervenciones")
    print("• Hacer inferencias robustas bajo cambios en el entorno")
    print()
    
    print("DAGs (Directed Acyclic Graphs)")
    print("-" * 40)
    print("Un DAG es un grafo dirigido sin ciclos donde:")
    print("• Nodos = Variables")
    print("• Aristas dirigidas = Relaciones causales directas") 
    print("• X → Y significa 'X causa directamente a Y'")
    print()
    
    # Ejemplo simple
    print("Ejemplo Simple: Lluvia → Pavimento Mojado → Accidentes")
    print()
    
    # Crear DAG simple
    variables = ['lluvia', 'pavimento_mojado', 'accidentes']
    dag_simple = CausalDAG(variables)
    dag_simple.add_edge('lluvia', 'pavimento_mojado', strength=1.0)
    dag_simple.add_edge('pavimento_mojado', 'accidentes', strength=0.8)
    
    print("Variables del modelo:", variables)
    print("Relaciones causales:")
    print("  lluvia → pavimento_mojado (fuerza: 1.0)")
    print("  pavimento_mojado → accidentes (fuerza: 0.8)")
    print()
    
    # Visualizar
    dag_simple.visualize(title="Ejemplo Simple: Cadena Causal", figsize=(10, 6))
    
    input("Presiona Enter para continuar al siguiente concepto...")
    

def tutorial_d_separacion():
    """
    Parte 2: Concepto de D-Separación
    """
    print("\nD-SEPARACIÓN: INDEPENDENCIAS CONDICIONALES")
    print("=" * 60)
    print()
    
    print("¿Qué es la D-Separación?")
    print("-" * 40)
    print("La d-separación es un criterio gráfico para determinar si dos")
    print("variables son independientes dado un conjunto de variables observadas.")
    print()
    print("Regla básica: X e Y están d-separados por Z si todos los caminos")
    print("entre X e Y están 'bloqueados' por las variables en Z.")
    print()
    
    # Crear ejemplo más complejo
    variables = ['A', 'B', 'C', 'D', 'E']
    dag = CausalDAG(variables)
    
    # Estructura: A → B → D, A → C → D, B → E
    dag.add_edge('A', 'B', strength=1.0)
    dag.add_edge('A', 'C', strength=1.0) 
    dag.add_edge('B', 'D', strength=1.0)
    dag.add_edge('C', 'D', strength=1.0)
    dag.add_edge('B', 'E', strength=1.0)
    
    print("Ejemplo de D-Separación:")
    print("Estructura causal: A → B → D, A → C → D, B → E")
    print()
    
    # Visualizar
    dag.visualize(title="Ejemplo de D-Separación", figsize=(10, 6))
    
    print("Análisis de Independencias:")
    print("-" * 35)
    
    # Casos de independencia
    casos = [
        ('A', 'E', []),
        ('A', 'E', ['B']),
        ('C', 'E', []),
        ('C', 'E', ['A']),
        ('A', 'D', ['B', 'C'])
    ]
    
    for x, y, z in casos:
        is_separated = dag.is_d_separated(x, y, z)
        z_str = f" | {', '.join(z)}" if z else ""
        status = "INDEPENDIENTES" if is_separated else "DEPENDIENTES"
        print(f"• {x} ⊥ {y}{z_str}: {status}")
    
    print()
    print("Interpretación:")
    print("• A ⊥ E: No están directamente conectados → INDEPENDIENTES")
    print("• A ⊥ E | B: Condicionado en B bloquea el camino → INDEPENDIENTES")
    print("• C ⊥ E: Conectados vía A-B → DEPENDIENTES")
    print()
    
    input("Presiona Enter para continuar...")


def tutorial_clasificador_paso_a_paso():
    """
    Parte 3: Construcción paso a paso del clasificador causal
    """
    print("\nCONSTRUCCIÓN DEL CLASIFICADOR CAUSAL")
    print("=" * 60)
    print()
    
    print("Paso 1: Definir el Problema")
    print("-" * 35)
    print("Problema: Predecir si un estudiante aprobará un examen")
    print("Variables disponibles:")
    print("• horas_estudio: Horas dedicadas al estudio")
    print("• asistencia: Porcentaje de asistencia a clases") 
    print("• dificultad_materia: Dificultad percibida de la materia")
    print("• estres: Nivel de estrés del estudiante")
    print("• rendimiento: Rendimiento académico histórico")
    print("• aprobado: Variable objetivo (1=aprueba, 0=no aprueba)")
    print()
    
    print("Paso 2: Modelar Relaciones Causales")
    print("-" * 40)
    print("Basándonos en conocimiento del dominio educativo:")
    print("• dificultad_materia → estres (materias difíciles causan más estrés)")
    print("• rendimiento → horas_estudio (buen rendimiento motiva más estudio)")
    print("• horas_estudio → aprobado (más estudio mejora probabilidad)")
    print("• asistencia → aprobado (asistir a clases ayuda)")
    print("• estres → aprobado (estrés puede afectar negativamente)")
    print()
    
    # Construir DAG
    variables = ['horas_estudio', 'asistencia', 'dificultad_materia', 
                'estres', 'rendimiento', 'aprobado']
    
    dag_estudiante = CausalDAG(variables)
    
    # Relaciones causales
    dag_estudiante.add_edge('dificultad_materia', 'estres', strength=0.7)
    dag_estudiante.add_edge('rendimiento', 'horas_estudio', strength=0.6)
    dag_estudiante.add_edge('horas_estudio', 'aprobado', strength=1.2)
    dag_estudiante.add_edge('asistencia', 'aprobado', strength=1.0)
    dag_estudiante.add_edge('estres', 'aprobado', strength=-0.8)
    
    print("Paso 3: Visualizar el Modelo Causal")
    print("-" * 40)
    dag_estudiante.visualize(title="Modelo Causal: Aprobación de Estudiantes", 
                           figsize=(12, 8))
    
    print("Paso 4: Generar Datos Sintéticos")
    print("-" * 40)
    
    # Generar datos sintéticos basados en el modelo causal
    np.random.seed(42)
    n_samples = 500
    
    # Variables exógenas
    dificultad_materia = np.random.uniform(1, 10, n_samples)
    asistencia = np.random.uniform(0.5, 1.0, n_samples)
    rendimiento = np.random.normal(7, 2, n_samples)
    rendimiento = np.clip(rendimiento, 1, 10)
    
    # Variables endógenas (causadas por otras)
    estres = (0.7 * dificultad_materia + 
              np.random.normal(0, 1, n_samples))
    estres = np.clip(estres, 1, 10)
    
    horas_estudio = (0.6 * rendimiento + 
                    np.random.normal(0, 1, n_samples))
    horas_estudio = np.clip(horas_estudio, 0, 12)
    
    # Variable objetivo
    aprobado_score = (1.2 * horas_estudio + 
                     1.0 * asistencia * 10 + 
                     -0.8 * estres +
                     np.random.normal(0, 2, n_samples))
    
    aprobado = (aprobado_score > np.median(aprobado_score)).astype(int)
    
    # Crear dataset
    data_estudiantes = pd.DataFrame({
        'horas_estudio': horas_estudio,
        'asistencia': asistencia,
        'dificultad_materia': dificultad_materia,
        'estres': estres,
        'rendimiento': rendimiento
    })
    
    target_estudiantes = pd.Series(aprobado, name='aprobado')
    
    print(f"✓ Generados {len(data_estudiantes)} estudiantes sintéticos")
    print(f"✓ Tasa de aprobación: {target_estudiantes.mean():.1%}")
    print()
    print("Estadísticas del dataset:")
    print(data_estudiantes.describe().round(2))
    
    print("\nPaso 5: Entrenar Clasificador Causal")
    print("-" * 43)
    
    # Dividir datos
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        data_estudiantes, target_estudiantes, 
        test_size=0.3, random_state=42, stratify=target_estudiantes
    )
    
    # Crear y entrenar clasificador
    clasificador_estudiante = CausalClassifier(dag_estudiante, 'aprobado')
    clasificador_estudiante.fit(X_train, y_train)
    
    print("✓ Clasificador entrenado exitosamente")
    
    # Hacer predicciones
    predicciones = clasificador_estudiante.predict(X_test)
    probabilidades = clasificador_estudiante.predict_proba(X_test)
    
    # Evaluar
    from sklearn.metrics import accuracy_score, classification_report
    accuracy = accuracy_score(y_test, predicciones)
    print(f"✓ Precisión en test: {accuracy:.3f}")
    
    print("\nPaso 6: Analizar Importancia Causal")
    print("-" * 42)
    importancia = clasificador_estudiante.get_feature_importance()
    print("Importancia de variables causales:")
    for var, imp in sorted(importancia.items(), key=lambda x: x[1], reverse=True):
        print(f"• {var}: {imp:.3f}")
    
    print("\nPaso 7: Interpretar Resultados")
    print("-" * 37)
    print("Insights del modelo causal:")
    print("• Las horas de estudio tienen el mayor impacto causal")
    print("• La asistencia es importante pero secundaria")
    print("• El estrés tiene un efecto negativo significativo")
    print("• El rendimiento histórico influye indirectamente vía horas de estudio")
    print()
    print("Recomendaciones causales para mejorar aprobación:")
    print("• Aumentar horas de estudio (efecto directo positivo)")
    print("• Mejorar asistencia a clases (efecto directo positivo)")
    print("• Reducir estrés (efecto directo negativo)")
    print("• Considerar dificultad de materia al planificar estudio")
    
    return data_estudiantes, target_estudiantes, clasificador_estudiante


def tutorial_comparacion_metodos():
    """
    Parte 4: Comparación con métodos tradicionales
    """
    print("\nCOMPARACIÓN CON MÉTODOS TRADICIONALES")
    print("=" * 60)
    print()
    
    print("¿Por qué usar modelos causales en lugar de ML tradicional?")
    print("-" * 65)
    print()
    
    print("Ventajas de los Modelos Causales:")
    print("• INTERPRETABILIDAD: Explican el 'por qué' detrás de las predicciones")
    print("• ROBUSTEZ: Menos sensibles a cambios en la distribución de datos")
    print("• INTERVENCIONES: Permiten predecir efectos de acciones específicas")
    print("• CONOCIMIENTO DE DOMINIO: Incorporan expertise humano en el modelo")
    print("• TRANSFERIBILIDAD: Generalizan mejor a nuevos contextos")
    print()
    
    print("Limitaciones:")
    print("• CONOCIMIENTO REQUERIDO: Necesitan entendimiento causal del dominio")
    print("• COMPLEJIDAD DE MODELADO: Requieren definir estructura causal correcta")
    print("• DATOS LIMITADOS: Pueden necesitar más datos para relaciones causales")
    print()
    
    print("Cuándo usar cada enfoque:")
    print("-" * 35)
    print("Modelos Causales son mejores cuando:")
    print("  • Necesitas explicabilidad")
    print("  • Planeas hacer intervenciones")
    print("  • Los datos pueden cambiar de distribución")
    print("  • Tienes conocimiento del dominio")
    print()
    print("ML Tradicional es mejor cuando:")
    print("  • Solo necesitas precisión predictiva")
    print("  • No tienes conocimiento causal")
    print("  • Los datos son muy complejos/alta dimensionalidad")
    print("  • La interpretabilidad no es crítica")
    print()


def tutorial_casos_uso():
    """
    Parte 5: Casos de uso reales
    """
    print("\nCASOS DE USO REALES")
    print("=" * 60)
    print()
    
    casos = [
        {
            "dominio": "MEDICINA",
            "problema": "Diagnóstico de enfermedades",
            "variables": ["síntomas", "factores de riesgo", "historial", "pruebas"],
            "objetivo": "diagnóstico",
            "ventaja": "Los médicos pueden entender las relaciones causales y validar el modelo"
        },
        {
            "dominio": "FINANZAS", 
            "problema": "Evaluación de riesgo crediticio",
            "variables": ["ingresos", "historial", "empleo", "deudas"],
            "objetivo": "default",
            "ventaja": "Permite identificar intervenciones para reducir riesgo"
        },
        {
            "dominio": "MARKETING",
            "problema": "Optimización de conversiones",
            "variables": ["demografía", "canal", "interés", "exposición"],
            "objetivo": "conversión", 
            "ventaja": "Identifica qué acciones específicas aumentan conversiones"
        },
        {
            "dominio": "MANUFACTURA",
            "problema": "Control de calidad",
            "variables": ["temperatura", "presión", "velocidad", "materiales"],
            "objetivo": "defectos",
            "ventaja": "Permite optimizar procesos y reducir defectos específicos"
        }
    ]
    
    for i, caso in enumerate(casos, 1):
        print(f"{i}. {caso['dominio']}")
        print(f"   Problema: {caso['problema']}")
        print(f"   Variables: {', '.join(caso['variables'])}")
        print(f"   Objetivo: {caso['objetivo']}")
        print(f"   Ventaja clave: {caso['ventaja']}")
        print()


def main():
    """Ejecutar tutorial completo"""
    print("TUTORIAL INTERACTIVO: CLASIFICADORES CAUSALES")
    print("=" * 70)
    print()
    print("Este tutorial te enseñará paso a paso cómo implementar y usar")
    print("clasificadores basados en modelos gráficos causales.")
    print()
    
    input("Presiona Enter para comenzar...")
    
    # Parte 1: Conceptos básicos
    tutorial_conceptos_basicos()
    
    # Parte 2: D-separación
    tutorial_d_separacion()
    
    # Parte 3: Clasificador paso a paso
    datos, target, clasificador = tutorial_clasificador_paso_a_paso()
    
    # Parte 4: Comparación
    tutorial_comparacion_metodos()
    
    # Parte 5: Casos de uso
    tutorial_casos_uso()
    
    print("\n¡TUTORIAL COMPLETADO!")
    print("=" * 60)
    print()
    print("Lo que has aprendido:")
    print("• Conceptos fundamentales de modelos causales")
    print("• Cómo construir DAGs causales")
    print("• Implementación de clasificadores causales")
    print("• Ventajas sobre métodos tradicionales")
    print("• Casos de uso en diferentes dominios")
    print()
    print("Próximos pasos:")
    print("• Practica con tus propios datos")
    print("• Explora ejemplos más avanzados")
    print("• Lee sobre teoría de inferencia causal")
    print("• Experimenta con diferentes estructuras causales")
    print()
    print("Recursos recomendados:")
    print("• 'Causal Inference in Statistics' - Judea Pearl")
    print("• 'The Book of Why' - Judea Pearl")
    print("• 'Causal Inference: The Mixtape' - Scott Cunningham")


if __name__ == "__main__":
    main()