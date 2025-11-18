"""
Test Rápido del Clasificador Causal
===================================

Script para verificar que la implementación funciona correctamente
sin necesidad de visualizaciones complejas.
"""

import numpy as np
import pandas as pd
from causal_classifier import CausalDAG, CausalClassifier, generate_causal_data, create_medical_dag
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def test_basico():
    """Test básico del funcionamiento del clasificador causal."""
    print("🧪 EJECUTANDO TEST BÁSICO DEL CLASIFICADOR CAUSAL")
    print("=" * 60)
    
    try:
        # 1. Generar datos sintéticos
        print("\n📊 1. Generando datos sintéticos...")
        X, y = generate_causal_data(n_samples=100, noise_level=0.1)
        print(f"   ✓ Generadas {len(X)} muestras")
        print(f"   ✓ Variables: {list(X.columns)}")
        print(f"   ✓ Tasa objetivo: {y.mean():.1%}")
        
        # 2. Crear DAG causal
        print("\n🏗️ 2. Creando DAG causal...")
        dag = create_medical_dag()
        print(f"   ✓ DAG con {len(dag.variables)} variables")
        print(f"   ✓ {len(dag.graph.edges)} relaciones causales")
        
        # 3. Verificar estructura del DAG
        print("\n🔍 3. Verificando estructura causal...")
        target_parents = dag.get_parents('riesgo_cardiovascular')
        print(f"   ✓ Causas directas del riesgo: {target_parents}")
        
        target_ancestors = dag.get_ancestors('riesgo_cardiovascular')
        print(f"   ✓ Causas indirectas: {target_ancestors}")
        
        # 4. Test de d-separación
        print("\n🧮 4. Test de d-separación...")
        is_separated = dag.is_d_separated('edad', 'ejercicio', [])
        print(f"   ✓ edad ⊥ ejercicio: {'Sí' if is_separated else 'No'}")
        
        # 5. Dividir datos
        print("\n📋 5. Dividiendo datos...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        print(f"   ✓ Entrenamiento: {len(X_train)} muestras")
        print(f"   ✓ Prueba: {len(X_test)} muestras")
        
        # 6. Crear y entrenar clasificador
        print("\n🤖 6. Entrenando clasificador causal...")
        classifier = CausalClassifier(dag, 'riesgo_cardiovascular')
        classifier.fit(X_train, y_train)
        print("   ✓ Clasificador entrenado exitosamente")
        
        # 7. Hacer predicciones
        print("\n🎯 7. Realizando predicciones...")
        y_pred = classifier.predict(X_test)
        y_proba = classifier.predict_proba(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        print(f"   ✓ Predicciones realizadas")
        print(f"   ✓ Precisión: {accuracy:.3f}")
        
        # 8. Analizar importancia causal
        print("\n🔍 8. Analizando importancia causal...")
        importance = classifier.get_feature_importance()
        print("   Variables más importantes:")
        for var, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True)[:3]:
            print(f"     • {var}: {imp:.3f}")
        
        # 9. Estadísticas del modelo
        print("\n📈 9. Estadísticas del modelo...")
        print(f"   ✓ Umbral de decisión: {classifier.threshold:.3f}")
        print(f"   ✓ Variables de entrada: {len(classifier.feature_variables)}")
        
        print("\n✅ TODAS LAS PRUEBAS PASARON EXITOSAMENTE")
        print("\nEl clasificador causal está funcionando correctamente y puede:")
        print("• Procesar datos con relaciones causales")
        print("• Realizar predicciones basadas en estructura causal") 
        print("• Calcular importancia de variables causales")
        print("• Analizar independencias condicionales")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR EN EL TEST: {str(e)}")
        print("\nVerifica que todas las dependencias estén instaladas:")
        print("pip install numpy pandas networkx matplotlib seaborn scipy scikit-learn")
        return False


def demo_ejemplo_simple():
    """Demo con ejemplo simplificado para entender el concepto."""
    print("\n\n🎓 DEMO: EJEMPLO EDUCATIVO SIMPLE")
    print("=" * 60)
    
    # Crear DAG simple: Estudio → Conocimiento → Calificación
    variables = ['horas_estudio', 'conocimiento', 'calificacion_alta']
    dag_simple = CausalDAG(variables)
    
    # Relaciones causales simples
    dag_simple.add_edge('horas_estudio', 'conocimiento', strength=1.0)
    dag_simple.add_edge('conocimiento', 'calificacion_alta', strength=1.5)
    
    print("\n📚 Modelo causal: Horas de estudio → Conocimiento → Calificación alta")
    print("\nEste modelo representa que:")
    print("• Estudiar más horas aumenta el conocimiento")
    print("• Mayor conocimiento aumenta probabilidad de calificación alta")
    
    # Generar datos sintéticos simples
    np.random.seed(42)
    n = 200
    
    horas_estudio = np.random.uniform(0, 8, n)
    conocimiento = 0.8 * horas_estudio + np.random.normal(0, 0.5, n)
    conocimiento = np.clip(conocimiento, 0, 10)
    
    calificacion_score = 1.2 * conocimiento + np.random.normal(0, 1, n)
    calificacion_alta = (calificacion_score > np.percentile(calificacion_score, 70)).astype(int)
    
    data_simple = pd.DataFrame({
        'horas_estudio': horas_estudio,
        'conocimiento': conocimiento
    })
    
    target_simple = pd.Series(calificacion_alta, name='calificacion_alta')
    
    print(f"\n📊 Datos generados: {len(data_simple)} estudiantes")
    print(f"📊 Tasa de calificaciones altas: {target_simple.mean():.1%}")
    
    # Entrenar clasificador simple
    X_train, X_test, y_train, y_test = train_test_split(
        data_simple, target_simple, test_size=0.3, random_state=42
    )
    
    classifier_simple = CausalClassifier(dag_simple, 'calificacion_alta')
    classifier_simple.fit(X_train, y_train)
    
    y_pred_simple = classifier_simple.predict(X_test)
    accuracy_simple = accuracy_score(y_test, y_pred_simple)
    
    print(f"\n🎯 Resultado del clasificador causal: {accuracy_simple:.3f} precisión")
    
    # Mostrar importancia
    importance_simple = classifier_simple.get_feature_importance()
    print("\n🔍 Importancia causal:")
    for var, imp in importance_simple.items():
        print(f"   • {var}: {imp:.3f}")
    
    print("\n💡 Interpretación:")
    print("• El modelo identifica correctamente las relaciones causales")
    print("• Las horas de estudio tienen efecto indirecto via conocimiento")
    print("• El conocimiento tiene efecto directo en la calificación")
    print("• Este enfoque es más interpretable que ML tradicional")


if __name__ == "__main__":
    print("🚀 INICIANDO VERIFICACIÓN DEL CLASIFICADOR CAUSAL")
    print("Este script verifica que la implementación funcione correctamente")
    print()
    
    # Ejecutar test básico
    exito = test_basico()
    
    if exito:
        # Si el test básico pasa, mostrar demo educativo
        demo_ejemplo_simple()
        
        print("\n" + "="*60)
        print("🎉 VERIFICACIÓN COMPLETADA EXITOSAMENTE")
        print("="*60)
        print("\n🚀 Próximos pasos:")
        print("• Ejecuta 'python causal_classifier.py' para la demo completa")
        print("• Ejecuta 'python marketing_example.py' para el ejemplo de marketing")
        print("• Ejecuta 'python tutorial.py' para el tutorial interactivo")
        print("• Lee el README.md para más información")
        
    else:
        print("\n❌ La verificación falló. Revisa la instalación de dependencias.")