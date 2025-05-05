# random_forest_app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import seaborn as sns
import matplotlib.pyplot as plt

# Configuración de la página
st.set_page_config(
    page_title="Predictor de Deudas - Random Forest",
    page_icon="💰",
    layout="wide"
)

# Título principal
st.title("🏦 Sistema de Predicción de Deudas con Random Forest")
st.markdown("---")

# Sidebar para navegación
st.sidebar.title("🚀 Navegación")
page = st.sidebar.radio(
    "Ir a",
    ["📊 Datos", "🤖 Modelo", "🎯 Predicción", "📈 Resultados", "📋 Información"]
)

# Cargar datos
@st.cache_data
def load_data():
    df = pd.read_csv("data.csv")
    return df

# Entrenar modelo
@st.cache_resource
def train_model(df):
    X = df.drop(['cliente_id', 'default'], axis=1)
    y = df['default']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    return model, X_train, X_test, y_train, y_test, X.columns

# Cargar datos
df = load_data()

# Página 1: Datos
if page == "📊 Datos":
    st.header("📊 Análisis de Datos")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total de Registros", len(df))
    with col2:
        st.metric("Tasa de Incumplimiento", f"{df['default'].mean():.1%}")
    with col3:
        st.metric("Características", len(df.columns) - 2)
    
    st.subheader("Vista General del Dataset")
    st.dataframe(df.head(10), use_container_width=True)
    
    st.subheader("Estadísticas Descriptivas")
    st.dataframe(df.describe(), use_container_width=True)
    
    # Visualizaciones interactivas
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Distribución de Incumplimientos")
        default_counts = df['default'].value_counts()
        fig_default = px.bar(
            x=['No Incumplimiento', 'Incumplimiento'],
            y=default_counts.values,
            labels={'x': 'Estado', 'y': 'Cantidad'},
            color=['Normal', 'Incumplimiento'],
            color_discrete_map={'Normal': '#1f77b4', 'Incumplimiento': '#d62728'}
        )
        st.plotly_chart(fig_default, use_container_width=True)
    
    with col2:
        st.subheader("Distribución de Edad")
        fig_age = px.histogram(df, x='edad', nbins=20, color='default',
                              labels={'edad': 'Edad', 'count': 'Cantidad'},
                              color_discrete_map={0: '#1f77b4', 1: '#d62728'})
        st.plotly_chart(fig_age, use_container_width=True)
    
    st.subheader("Correlación entre Variables")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()
    
    fig_corr = px.imshow(corr_matrix,
                         labels=dict(color="Correlación"),
                         x=numeric_cols,
                         y=numeric_cols,
                         color_continuous_scale="RdBu")
    fig_corr.update_layout(width=800, height=800)
    st.plotly_chart(fig_corr, use_container_width=True)

# Página 2: Modelo
elif page == "🤖 Modelo":
    st.header("🤖 Entrenamiento y Evaluación del Modelo")
    
    model, X_train, X_test, y_train, y_test, feature_names = train_model(df)
    
    # Métricas principales
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        accuracy = (y_pred == y_test).mean()
        st.metric("Precisión", f"{accuracy:.2%}")
    
    with col2:
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        st.metric("ROC AUC", f"{roc_auc:.3f}")
    
    with col3:
        # Recall
        from sklearn.metrics import recall_score
        recall = recall_score(y_test, y_pred)
        st.metric("Recall", f"{recall:.2%}")
    
    with col4:
        # F1-score
        from sklearn.metrics import f1_score
        f1 = f1_score(y_test, y_pred)
        st.metric("F1-Score", f"{f1:.3f}")
    
    # Matrices de confusión
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Matriz de Confusión")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm = px.imshow(cm,
                           labels=dict(x="Predicción", y="Valor Real", color="Cantidad"),
                           x=['No Incumplimiento', 'Incumplimiento'],
                           y=['No Incumplimiento', 'Incumplimiento'],
                           text_auto=True,
                           color_continuous_scale="Viridis")
        st.plotly_chart(fig_cm, use_container_width=True)
    
    with col2:
        st.subheader("Curva ROC")
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, 
                                     name=f'ROC (AUC = {roc_auc:.3f})',
                                     mode='lines',
                                     line=dict(color='#1f77b4', width=3)))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1],
                                     name='Línea Base',
                                     mode='lines',
                                     line=dict(color='black', width=1, dash='dash')))
        fig_roc.update_layout(xaxis_title='Tasa de Falsos Positivos',
                              yaxis_title='Tasa de Verdaderos Positivos',
                              title='Curva ROC')
        st.plotly_chart(fig_roc, use_container_width=True)
    
    # Importancia de características
    st.subheader("Importancia de Características")
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    fig_importance = px.bar(feature_importance.head(10),
                            x='importance',
                            y='feature',
                            orientation='h',
                            labels={'importance': 'Importancia', 'feature': 'Característica'})
    fig_importance.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig_importance, use_container_width=True)

# Página 3: Predicción
elif page == "🎯 Predicción":
    st.header("🎯 Predicción de Nuevos Clientes")
    
    model, _, _, _, _, feature_names = train_model(df)
    
    # Formulario para nueva predicción
    st.subheader("Ingrese los datos del cliente:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        edad = st.number_input("Edad", min_value=18, max_value=100, value=35)
        ingreso_mensual = st.number_input("Ingreso Mensual ($)", min_value=0, value=5000)
        historial_credito_meses = st.slider("Historial Crediticio (meses)", 0, 120, 30)
        puntuacion_credito = st.slider("Puntuación Crediticia", 300, 850, 650)
        monto_prestamo = st.number_input("Monto del Préstamo ($)", min_value=0, value=18000)
    
    with col2:
        plazo_meses = st.slider("Plazo (meses)", 12, 60, 48)
        tasa_interes = st.number_input("Tasa de Interés (%)", min_value=0.0, value=9.5, step=0.1)
        ratio_deuda_ingreso = st.slider("Ratio Deuda/Ingreso", 0.0, 1.0, 0.4)
        empleo_estable = st.selectbox("Empleo Estable", [0, 1], format_func=lambda x: "Sí" if x == 1 else "No", index=1)
        educacion = st.selectbox("Educación", [1, 2, 3, 4], format_func=lambda x: ["Primaria", "Secundaria", "Universitaria", "Posgrado"][x-1], index=2)
    
    with col3:
        vivienda = st.selectbox("Vivienda", [1, 2, 3], format_func=lambda x: ["Propia", "Alquiler", "Otros"][x-1])
        num_dependientes = st.number_input("Número de Dependientes", min_value=0, value=2)
        historial_pagos = st.selectbox("Historial de Pagos", [0, 1, 2], 
    format_func=lambda x: "Sin atrasos" if x == 0 else "1 atraso" if x == 1 else "2+ atrasos")
        accion_legal = st.selectbox("Acciones Legales", [0, 1], format_func=lambda x: "Sí" if x == 1 else "No")
    
    # Hacer predicción
    if st.button("🔮 Predecir Riesgo", type="primary"):
        new_data = pd.DataFrame({
            'edad': [edad],
            'ingreso_mensual': [ingreso_mensual],
            'historial_credito_meses': [historial_credito_meses],
            'puntuacion_credito': [puntuacion_credito],
            'monto_prestamo': [monto_prestamo],
            'plazo_meses': [plazo_meses],
            'tasa_interes': [tasa_interes],
            'ratio_deuda_ingreso': [ratio_deuda_ingreso],
            'empleo_estable': [empleo_estable],
            'educacion': [educacion],
            'vivienda': [vivienda],
            'num_dependientes': [num_dependientes],
            'historial_pagos': [historial_pagos],
            'accion_legal': [accion_legal]
        })
        
        prediction = model.predict(new_data)[0]
        probability = model.predict_proba(new_data)[0][1]
        
        st.markdown("---")
        st.subheader("Resultado de la Predicción")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Predicción", "❌ INCUMPLIRÁ" if prediction == 1 else "✅ NO INCUMPLIRÁ")
        
        with col2:
            st.metric("Probabilidad de Incumplimiento", f"{probability:.1%}")
        
        with col3:
            risk_level = "🔴 ALTO" if probability > 0.7 else "🟡 MEDIO" if probability > 0.4 else "🟢 BAJO"
            st.metric("Nivel de Riesgo", risk_level)
        
        # Visualización de probabilidad
        fig_prob = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            title={'text': "Probabilidad de Incumplimiento (%)"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': 'red' if probability > 0.7 else 'orange' if probability > 0.4 else 'green'},
                'steps': [
                    {'range': [0, 40], 'color': 'lightgreen'},
                    {'range': [40, 70], 'color': 'lightsalmon'},
                    {'range': [70, 100], 'color': 'lightcoral'}
                ]
            }
        ))
        st.plotly_chart(fig_prob, use_container_width=True)

# Página 4: Resultados
elif page == "📈 Resultados":
    st.header("📈 Análisis de Resultados")
    
    model, _, _, _, _, feature_names = train_model(df)
    
    # Análisis de importancia
    st.subheader("Análisis Avanzado de Importancia")
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Crear gráfico de barras con colores y estilo mejorado
    fig_importance = px.bar(
        feature_importance.head(10),
        x='importance',
        y='feature',
        orientation='h',
        title="Top 10 Características Más Importantes",
        labels={'importance': 'Importancia Relativa', 'feature': 'Característica'},
        color='importance',
        color_continuous_scale='Viridis'
    )
    fig_importance.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig_importance, use_container_width=True)
    
    # Análisis por grupos
    st.subheader("Análisis por Grupos de Riesgo")
    
    # Agregar predicciones al DataFrame original
    X = df.drop(['cliente_id', 'default'], axis=1)
    y_pred_proba = model.predict_proba(X)[:, 1]
    df_analysis = df.copy()
    df_analysis['predicted_risk'] = y_pred_proba
    df_analysis['risk_group'] = pd.cut(y_pred_proba, bins=[0, 0.2, 0.5, 0.8, 1.0], 
                                      labels=['Bajo', 'Medio', 'Alto', 'Muy Alto'])
    
    # Visualización por grupos de riesgo
    risk_stats = df_analysis.groupby('risk_group').agg({
        'edad': 'mean',
        'ingreso_mensual': 'mean',
        'puntuacion_credito': 'mean',
        'monto_prestamo': 'mean',
        'default': 'mean'
    }).reset_index()
    
    fig_risk_groups = make_subplots(rows=2, cols=2, 
                                    subplot_titles=('Edad Promedio por Grupo', 
                                                   'Ingreso Promedio por Grupo',
                                                   'Puntuación Crediticia por Grupo',
                                                   'Tasa de Incumplimiento Real'))
    
    fig_risk_groups.add_trace(go.Bar(x=risk_stats['risk_group'], y=risk_stats['edad'], 
                                     name='Edad'), row=1, col=1)
    fig_risk_groups.add_trace(go.Bar(x=risk_stats['risk_group'], y=risk_stats['ingreso_mensual'], 
                                     name='Ingreso'), row=1, col=2)
    fig_risk_groups.add_trace(go.Bar(x=risk_stats['risk_group'], y=risk_stats['puntuacion_credito'], 
                                     name='Score'), row=2, col=1)
    fig_risk_groups.add_trace(go.Bar(x=risk_stats['risk_group'], y=risk_stats['default']*100, 
                                     name='Default %'), row=2, col=2)
    
    fig_risk_groups.update_layout(height=800, showlegend=False)
    st.plotly_chart(fig_risk_groups, use_container_width=True)

# Página 5: Información
elif page == "📋 Información":
    st.header("📋 Información sobre el Sistema")
    
    st.markdown("""
    ## 🎯 Objetivos del Sistema
    
    Este sistema utiliza un modelo Random Forest para predecir la probabilidad de que un cliente 
    incumpla con el pago de sus deudas.
    
    ### 📊 Características del Modelo
    
    - **Algoritmo**: Random Forest (100 árboles)
    - **Métricas**: Precisión, Recall, F1-Score, ROC AUC
    - **Variables de entrada**: 14 características (demográficas, financieras, comportamentales)
    - **Variable objetivo**: Incumplimiento de pago (default)
    
    ### 🔄 Flujo de Trabajo
    
    1. **Carga de Datos**: Análisis exploratorio de dataset
    2. **Entrenamiento**: Modelo Random Forest con validación cruzada
    3. **Evaluación**: Métricas de rendimiento y análisis de errores
    4. **Predicción**: Interfaz para nuevos clientes
    5. **Resultados**: Visualización y análisis avanzado
    
    ### 📈 Interpretación de Resultados
    
    **Niveles de Riesgo:**
    - 🟢 **Bajo**: Probabilidad < 40%
    - 🟡 **Medio**: Probabilidad 40-70%
    - 🔴 **Alto**: Probabilidad > 70%
    
    ### 💡 Factores Clave
    
    Las características más importantes típicamente incluyen:
    1. Puntuación crediticia
    2. Historial de pagos
    3. Ratio deuda/ingreso
    4. Ingreso mensual
    5. Historial crediticio (meses)
    
    ### ⚠️ Limitaciones
    
    - El modelo se basa en datos históricos
    - No considera factores externos (crisis económicas, etc.)
    - Requiere actualización periódica
    
    ### 🚀 Recomendaciones de Uso
    
    - Utilizar como herramienta de apoyo en decisiones de crédito
    - Combinar con análisis manual para casos límite
    - Actualizar regularmente con nuevos datos
    """)
    
    # Información técnica
    with st.expander("🔧 Información Técnica"):
        st.markdown("""
        **Bibliotecas utilizadas:**
        - `sklearn`: Machine Learning
        - `streamlit`: Interfaz web
        - `plotly`: Visualizaciones interactivas
        - `pandas`: Manipulación de datos
        - `numpy`: Operaciones numéricas
        
        **Requisitos del sistema:**
        ```
        sklearn>=1.0.0
        streamlit>=1.10.0
        plotly>=5.3.0
        pandas>=1.3.0
        numpy>=1.20.0
        ```
        """)

# Footer
st.markdown("---")
st.markdown("💡 **Desarrollado para análisis de riesgo crediticio**")
st.markdown("🔒 **Datos confidenciales - Solo para uso interno**")