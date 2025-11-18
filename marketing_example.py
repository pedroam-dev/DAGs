"""
Ejemplo Avanzado: Clasificador Causal para Análisis de Marketing
===============================================================

Este ejemplo demuestra el uso del clasificador causal en un contexto
de marketing digital, analizando factores que influyen en la conversión
de clientes potenciales.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from causal_classifier import CausalDAG, CausalClassifier, generate_causal_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def generate_marketing_data(n_samples: int = 1500) -> tuple:
    """
    Genera datos sintéticos para un modelo de conversión de marketing.
    
    Variables causales:
    - Demografía (edad, ingresos) → Interés en producto
    - Canal de marketing → Exposición y engagement
    - Interés + Exposición → Conversión
    """
    np.random.seed(123)
    
    # Variables demográficas
    edad = np.random.normal(35, 12, n_samples)
    ingresos = np.maximum(20000, 
                         30000 + 800 * edad + np.random.normal(0, 15000, n_samples))
    
    # Canal de marketing (0: email, 1: social media, 2: publicidad online)
    canal_marketing = np.random.choice([0, 1, 2], n_samples, p=[0.4, 0.35, 0.25])
    
    # Interés en producto (influenciado por demografía)
    interes_producto = (0.02 * edad + 0.00002 * ingresos - 0.5 +
                       np.random.normal(0, 0.3, n_samples))
    interes_producto = 1 / (1 + np.exp(-interes_producto))  # Sigmoide
    
    # Exposición al marketing (influenciado por canal)
    exposicion_base = [0.3, 0.6, 0.8]  # email, social, online ads
    exposicion = np.array([exposicion_base[canal] for canal in canal_marketing])
    exposicion += np.random.normal(0, 0.1, n_samples)
    exposicion = np.clip(exposicion, 0, 1)
    
    # Engagement (influenciado por interés y exposición)
    engagement = (0.7 * interes_producto + 0.5 * exposicion +
                 np.random.normal(0, 0.2, n_samples))
    engagement = np.clip(engagement, 0, 1)
    
    # Conversión (influenciado por interés, exposición y engagement)
    conversion_logit = (2 * interes_producto + 1.5 * exposicion + 
                       3 * engagement - 2.5 +
                       np.random.normal(0, 0.3, n_samples))
    
    conversion = (1 / (1 + np.exp(-conversion_logit)) > 0.5).astype(int)
    
    # Crear DataFrame
    data = pd.DataFrame({
        'edad': edad,
        'ingresos': ingresos,
        'canal_marketing': canal_marketing,
        'interes_producto': interes_producto,
        'exposicion_marketing': exposicion,
        'engagement': engagement
    })
    
    target = pd.Series(conversion, name='conversion')
    
    return data, target


def create_marketing_dag() -> CausalDAG:
    """Crea DAG causal para el modelo de marketing."""
    variables = ['edad', 'ingresos', 'canal_marketing', 
                'interes_producto', 'exposicion_marketing', 
                'engagement', 'conversion']
    
    dag = CausalDAG(variables)
    
    # Relaciones causales del modelo de marketing
    dag.add_edge('edad', 'interes_producto', strength=1.0)
    dag.add_edge('ingresos', 'interes_producto', strength=0.8)
    dag.add_edge('canal_marketing', 'exposicion_marketing', strength=1.2)
    dag.add_edge('interes_producto', 'engagement', strength=0.9)
    dag.add_edge('exposicion_marketing', 'engagement', strength=0.7)
    dag.add_edge('interes_producto', 'conversion', strength=1.5)
    dag.add_edge('exposicion_marketing', 'conversion', strength=1.0)
    dag.add_edge('engagement', 'conversion', strength=2.0)
    
    return dag


def compare_classifiers(X_train, X_test, y_train, y_test, dag):
    """
    Compara el clasificador causal con métodos tradicionales.
    """
    results = {}
    
    # 1. Clasificador Causal
    causal_clf = CausalClassifier(dag, 'conversion')
    causal_clf.fit(X_train, y_train)
    y_pred_causal = causal_clf.predict(X_test)
    y_proba_causal = causal_clf.predict_proba(X_test)[:, 1]
    
    results['Causal'] = {
        'accuracy': accuracy_score(y_test, y_pred_causal),
        'auc': roc_auc_score(y_test, y_proba_causal),
        'predictions': y_pred_causal,
        'probabilities': y_proba_causal
    }
    
    # 2. Regresión Logística
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    y_proba_lr = lr.predict_proba(X_test)[:, 1]
    
    results['Logistic Regression'] = {
        'accuracy': accuracy_score(y_test, y_pred_lr),
        'auc': roc_auc_score(y_test, y_proba_lr),
        'predictions': y_pred_lr,
        'probabilities': y_proba_lr
    }
    
    # 3. Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    y_proba_rf = rf.predict_proba(X_test)[:, 1]
    
    results['Random Forest'] = {
        'accuracy': accuracy_score(y_test, y_pred_rf),
        'auc': roc_auc_score(y_test, y_proba_rf),
        'predictions': y_pred_rf,
        'probabilities': y_proba_rf
    }
    
    return results, causal_clf


def visualize_results(results):
    """Visualiza comparación de resultados."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Comparación de métricas
    methods = list(results.keys())
    accuracies = [results[method]['accuracy'] for method in methods]
    aucs = [results[method]['auc'] for method in methods]
    
    x = np.arange(len(methods))
    width = 0.35
    
    ax1.bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.8)
    ax1.bar(x + width/2, aucs, width, label='AUC', alpha=0.8)
    ax1.set_xlabel('Método')
    ax1.set_ylabel('Score')
    ax1.set_title('Comparación de Rendimiento')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Curvas ROC
    for method in methods:
        fpr, tpr, _ = roc_curve(y_test, results[method]['probabilities'])
        auc = results[method]['auc']
        ax2.plot(fpr, tpr, label=f'{method} (AUC = {auc:.3f})', linewidth=2)
    
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax2.set_xlabel('Tasa de Falsos Positivos')
    ax2.set_ylabel('Tasa de Verdaderos Positivos')
    ax2.set_title('Curvas ROC')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Distribución de probabilidades por método
    for i, method in enumerate(methods):
        probs = results[method]['probabilities']
        ax3.hist(probs, bins=30, alpha=0.6, label=method, density=True)
    
    ax3.set_xlabel('Probabilidad Predicha')
    ax3.set_ylabel('Densidad')
    ax3.set_title('Distribución de Probabilidades Predichas')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Matriz de correlación de predicciones
    pred_df = pd.DataFrame({
        method: results[method]['probabilities'] 
        for method in methods
    })
    
    corr_matrix = pred_df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, ax=ax4)
    ax4.set_title('Correlación entre Predicciones')
    
    plt.tight_layout()
    plt.show()


def analyze_causal_insights(causal_clf, X_test, y_test):
    """Analiza insights causales del modelo."""
    print("\n🔍 ANÁLISIS DE INSIGHTS CAUSALES")
    print("=" * 50)
    
    # Importancia causal
    importance = causal_clf.get_feature_importance()
    print("\n📊 Importancia de Variables Causales:")
    for var, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
        print(f"   • {var}: {imp:.3f}")
    
    # Análisis de efectos causales por segmentos
    print("\n🎯 Análisis por Segmentos de Clientes:")
    
    # Segmento por edad
    X_test_copy = X_test.copy()
    young_mask = X_test_copy['edad'] < 30
    old_mask = X_test_copy['edad'] >= 50
    
    if young_mask.sum() > 0:
        young_proba = causal_clf.predict_proba(X_test_copy[young_mask])[:, 1]
        print(f"   • Clientes jóvenes (<30 años): {young_proba.mean():.3f} conv. promedio")
    
    if old_mask.sum() > 0:
        old_proba = causal_clf.predict_proba(X_test_copy[old_mask])[:, 1]
        print(f"   • Clientes mayores (≥50 años): {old_proba.mean():.3f} conv. promedio")
    
    # Segmento por canal de marketing
    for canal in [0, 1, 2]:
        canal_names = ['Email', 'Social Media', 'Publicidad Online']
        canal_mask = X_test_copy['canal_marketing'] == canal
        if canal_mask.sum() > 0:
            canal_proba = causal_clf.predict_proba(X_test_copy[canal_mask])[:, 1]
            print(f"   • Canal {canal_names[canal]}: {canal_proba.mean():.3f} conv. promedio")
    
    print("\n💡 Recomendaciones Causales:")
    print("   • Variables con mayor impacto causal en conversión:")
    top_vars = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:3]
    for var, imp in top_vars:
        print(f"     - {var} (importancia: {imp:.3f})")
    
    print("   • Para mejorar conversiones, enfocar esfuerzos en:")
    print("     - Aumentar engagement (mayor impacto causal)")
    print("     - Optimizar exposición por canal de marketing")
    print("     - Personalizar según perfil demográfico")


if __name__ == "__main__":
    print("📈 EJEMPLO AVANZADO: MARKETING DIGITAL")
    print("=" * 60)
    
    # Generar datos
    print("\n📊 Generando datos de marketing...")
    X, y = generate_marketing_data(n_samples=1500)
    print(f"✓ Generadas {len(X)} muestras")
    print(f"✓ Tasa de conversión: {y.mean():.1%}")
    
    # Mostrar estadísticas descriptivas
    print("\n📋 Estadísticas Descriptivas:")
    print(X.describe().round(2))
    
    # Crear DAG
    print("\n🏗️ Construyendo DAG causal de marketing...")
    dag = create_marketing_dag()
    print("✓ DAG creado exitosamente")
    
    # Visualizar DAG
    dag.visualize(title="Modelo Causal: Conversión de Marketing Digital")
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Comparar clasificadores
    print("\n🤖 Comparando clasificadores...")
    results, causal_clf = compare_classifiers(X_train, X_test, y_train, y_test, dag)
    
    # Mostrar resultados
    print("\n📈 RESULTADOS DE COMPARACIÓN:")
    print("-" * 40)
    for method, metrics in results.items():
        print(f"{method:20s}: Accuracy = {metrics['accuracy']:.3f}, AUC = {metrics['auc']:.3f}")
    
    # Visualizar resultados
    visualize_results(results)
    
    # Análisis causal
    analyze_causal_insights(causal_clf, X_test, y_test)
    
    print("\n" + "=" * 60)
    print("✅ EJEMPLO COMPLETADO")
    print("\nEste ejemplo demuestra cómo los modelos causales pueden")
    print("proporcionar insights accionables para estrategias de marketing,")
    print("superando a métodos tradicionales en interpretabilidad y")
    print("capacidad de generar recomendaciones causales específicas.")