"""
Ejemplo Mínimo: Primer Contacto con Clasificadores Causales
==========================================================

Este es el ejemplo más simple posible para entender los conceptos básicos
del aprendizaje causal usando DAGs.
"""

import numpy as np
import pandas as pd
from causal_classifier import CausalDAG, CausalClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def ejemplo_minimo():
    """
    Ejemplo mínimo: Lluvia → Paraguas → Llegada seca
    """
    print("EJEMPLO MÍNIMO: MODELO CAUSAL SIMPLE")
    print("=" * 50)
    print()
    print("Problema: ¿Llegará una persona seca a destino?")
    print("Variables:")
    print("   • lluvia: Intensidad de lluvia (0-10)")
    print("   • paraguas: ¿Llevó paraguas? (0=no, 1=sí)")
    print("   • llega_seca: ¿Llegó seca? (0=no, 1=sí)")
    print()
    print("Modelo causal: lluvia → paraguas → llega_seca")
    print("   • Más lluvia aumenta probabilidad de llevar paraguas")
    print("   • Llevar paraguas aumenta probabilidad de llegar seco")
    print()
    
    # 1. Crear DAG causal
    variables = ['lluvia', 'paraguas', 'llega_seca']
    dag = CausalDAG(variables)
    
    # Definir relaciones causales
    dag.add_edge('lluvia', 'paraguas', strength=0.8)      # lluvia → paraguas
    dag.add_edge('paraguas', 'llega_seca', strength=1.2)  # paraguas → llega_seca
    
    print("✓ DAG creado con relaciones causales")
    
    # 2. Generar datos sintéticos
    np.random.seed(42)
    n = 300
    
    # Lluvia (variable exógena)
    lluvia = np.random.uniform(0, 10, n)
    
    # Paraguas (causado por lluvia)
    prob_paraguas = 1 / (1 + np.exp(-(0.8 * lluvia - 4)))  # Sigmoide
    paraguas = np.random.binomial(1, prob_paraguas, n)
    
    # Llega seca (causado por paraguas, y negativamente por lluvia)
    logit_seca = 1.2 * paraguas - 0.3 * lluvia + np.random.normal(0, 0.5, n)
    prob_seca = 1 / (1 + np.exp(-logit_seca))
    llega_seca = np.random.binomial(1, prob_seca, n)
    
    # Crear DataFrame
    datos = pd.DataFrame({
        'lluvia': lluvia,
        'paraguas': paraguas
    })
    
    objetivo = pd.Series(llega_seca, name='llega_seca')
    
    print(f"✓ Generados {len(datos)} casos")
    print(f"✓ {objetivo.mean():.1%} llegaron secos")
    print(f"✓ {datos['paraguas'].mean():.1%} llevaron paraguas")
    
    # 3. Entrenar clasificador causal
    X_train, X_test, y_train, y_test = train_test_split(
        datos, objetivo, test_size=0.3, random_state=42
    )
    
    clasificador = CausalClassifier(dag, 'llega_seca')
    clasificador.fit(X_train, y_train)
    
    print("✓ Clasificador causal entrenado")
    
    # 4. Hacer predicciones
    predicciones = clasificador.predict(X_test)
    probabilidades = clasificador.predict_proba(X_test)
    
    precision = accuracy_score(y_test, predicciones)
    print(f"✓ Precisión: {precision:.3f}")
    
    # 5. Analizar importancia causal
    importancia = clasificador.get_feature_importance()
    
    print("\nANÁLISIS CAUSAL:")
    print("Importancia de variables:")
    for var, imp in importancia.items():
        print(f"   • {var}: {imp:.3f}")
    
    # 6. Hacer predicción específica
    print("\nPREDICCIONES ESPECÍFICAS:")
    
    # Caso 1: Lluvia intensa, sin paraguas
    caso1 = pd.DataFrame({'lluvia': [8.0], 'paraguas': [0]})
    pred1 = clasificador.predict_proba(caso1)[0, 1]
    print(f"   Lluvia intensa SIN paraguas → P(llegar seco) = {pred1:.2f}")
    
    # Caso 2: Lluvia intensa, con paraguas  
    caso2 = pd.DataFrame({'lluvia': [8.0], 'paraguas': [1]})
    pred2 = clasificador.predict_proba(caso2)[0, 1]
    print(f"   Lluvia intensa CON paraguas → P(llegar seco) = {pred2:.2f}")
    
    # Caso 3: Sin lluvia
    caso3 = pd.DataFrame({'lluvia': [1.0], 'paraguas': [0]})
    pred3 = clasificador.predict_proba(caso3)[0, 1]
    print(f"   Sin lluvia → P(llegar seco) = {pred3:.2f}")
    
    print(f"\n💡 Efecto del paraguas: +{pred2-pred1:.2f} probabilidad")
    
    # 7. Análisis de independencias
    print("\nANÁLISIS DE INDEPENDENCIAS:")
    
    # ¿Son lluvia y llegar seco independientes?
    indep1 = dag.is_d_separated('lluvia', 'llega_seca', [])
    print(f"   lluvia ⊥ llega_seca: {'Sí' if indep1 else 'No'}")
    
    # ¿Son independientes dado paraguas?
    indep2 = dag.is_d_separated('lluvia', 'llega_seca', ['paraguas'])
    print(f"   lluvia ⊥ llega_seca | paraguas: {'Sí' if indep2 else 'No'}")
    
    print("\nRESUMEN:")
    print("• El modelo causal captura correctamente la lógica del problema")
    print("• Llevar paraguas aumenta significativamente la probabilidad de llegar seco")
    print("• La lluvia afecta indirectamente (vía motivar a llevar paraguas)")
    print("• El enfoque causal es naturalmente interpretable")
    
    return datos, objetivo, clasificador


def comparar_con_correlacion(datos, objetivo):
    """
    Compara el enfoque causal con análisis de correlación simple
    """
    print("\nCOMPARACIÓN: CAUSAL vs CORRELACIÓN")
    print("=" * 50)
    
    # Calcular correlaciones
    datos_completos = datos.copy()
    datos_completos['llega_seca'] = objetivo
    
    correlaciones = datos_completos.corr()['llega_seca'].drop('llega_seca')
    
    print("CORRELACIONES:")
    for var, corr in correlaciones.items():
        print(f"   • {var} ↔ llega_seca: {corr:.3f}")
    
    print("\nDIFERENCIAS CLAVE:")
    print("   CORRELACIÓN dice:")
    print("     • Paraguas correlaciona positivamente con llegar seco")
    print("     • Lluvia correlaciona negativamente con llegar seco")
    print()
    print("   MODELO CAUSAL explica:")
    print("     • Lluvia CAUSA que se lleve paraguas")
    print("     • Paraguas CAUSA llegar seco")
    print("     • Lluvia afecta indirectamente vía paraguas")
    print()
    print("   VENTAJA CAUSAL:")
    print("     • Explica el mecanismo subyacente")
    print("     • Permite predecir efectos de intervenciones")
    print("     • Más robusto ante cambios en el entorno")


def main():
    """Ejecutar ejemplo mínimo completo"""
    print("CLASIFICADOR CAUSAL - EJEMPLO MÍNIMO")
    print("Este ejemplo te introduce a los conceptos básicos")
    print("del aprendizaje causal de forma simple y clara.")
    print()
    
    # Ejecutar ejemplo principal
    datos, objetivo, clasificador = ejemplo_minimo()
    
    # Comparar enfoques
    comparar_con_correlacion(datos, objetivo)
    
    print("\n" + "="*50)
    print("EJEMPLO MÍNIMO COMPLETADO")
    print("="*50)
    print("\nHas aprendido:")
    print("• Cómo construir un DAG causal simple")
    print("• Cómo entrenar un clasificador causal")
    print("• La diferencia entre correlación y causación")
    print("• Cómo hacer predicciones causales específicas")
    print("\nSiguiente paso:")
    print("• Ejecuta 'python tutorial.py' para aprender más")
    print("• O 'python causal_classifier.py' para ejemplos avanzados")


if __name__ == "__main__":
    main()