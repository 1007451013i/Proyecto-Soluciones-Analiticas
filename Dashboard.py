import dash
from dash import html, dcc, Input, Output, State, callback
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# ==============================
# CARGAR DATOS Y ENTRENAR MODELO
# ==============================

# Importar datos desde GitHub
url = "https://raw.github.com/1007451013i/Proyecto-Soluciones-Analiticas/main/heart.csv"
df = pd.read_csv(url)

# Definir columna objetivo
target_col = "condition"

# Definir variables predictoras y variable objetivo
X = df.drop(columns=[target_col])
y = df[target_col]

# Dividir datos de entrenamiento y prueba
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Escalar los datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Usar Regresion Logistica
model = LogisticRegression(
    max_iter=1000,
    solver="liblinear",
    random_state=42
)
model.fit(X_train_scaled, y_train)

# Calcular metricas del modelo
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
auc_roc = roc_auc_score(y_test, y_proba)
f1 = f1_score(y_test, y_pred)

print("Modelo: Regresion Logistica")
print(f"Precision: {accuracy:.2%}")
print(f"AUC-ROC: {auc_roc:.3f}")
print(f"F1-Score: {f1:.3f}")

# ==============================
# APLICACION DASH
# ==============================

app = dash.Dash(__name__)

app.layout = html.Div([
    # HEADER
    html.Div([
        html.H1("Predictores de Enfermedad Cardiaca", 
                style={'textAlign': 'center', 'color': 'white', 'marginBottom': 10}),
        html.P("Sistema de prediccion basado en Regresion Logistica", 
               style={'textAlign': 'center', 'color': 'white', 'fontSize': 16})
    ], style={'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
              'padding': '30px', 'borderRadius': '10px', 'marginBottom': '30px'}),
    
    # PRIMERA FILA: Informacion del paciente
    html.Div([
        # COLUMNA 1: Formulario del paciente
        html.Div([
            html.H3("INFORMACION DEL PACIENTE", 
                   style={'color': '#2c3e50', 'borderBottom': '2px solid #3498db', 'paddingBottom': '10px'}),
            
            html.Div([
                # Fila 1 de inputs
                html.Div([
                    html.Div([
                        html.Label("Edad", style={'fontWeight': 'bold', 'color': '#2c3e50'}),
                        dcc.Input(id='edad', type='number', value=55, min=20, max=100,
                                 style={'width': '100%', 'padding': '10px', 'border': '2px solid #bdc3c7', 
                                       'borderRadius': '5px', 'fontSize': '16px'})
                    ], style={'marginBottom': '20px'}),
                    
                    html.Div([
                        html.Label("Sexo", style={'fontWeight': 'bold', 'color': '#2c3e50'}),
                        dcc.Dropdown(id='sexo', 
                                    options=[{'label': 'Masculino', 'value': 1}, 
                                            {'label': 'Femenino', 'value': 0}],
                                    value=1,
                                    style={'border': '2px solid #bdc3c7', 'borderRadius': '5px'})
                    ], style={'marginBottom': '20px'}),
                ], style={'display': 'flex', 'gap': '20px'}),
                
                # Fila 2 de inputs
                html.Div([
                    html.Div([
                        html.Label("Presion arterial", style={'fontWeight': 'bold', 'color': '#2c3e50'}),
                        dcc.Input(id='presion', type='number', value=130, min=80, max=200,
                                 style={'width': '100%', 'padding': '10px', 'border': '2px solid #bdc3c7', 
                                       'borderRadius': '5px', 'fontSize': '16px'})
                    ], style={'marginBottom': '20px'}),
                    
                    html.Div([
                        html.Label("Colesterol serico", style={'fontWeight': 'bold', 'color': '#2c3e50'}),
                        dcc.Input(id='colesterol', type='number', value=240, min=100, max=400,
                                 style={'width': '100%', 'padding': '10px', 'border': '2px solid #bdc3c7', 
                                       'borderRadius': '5px', 'fontSize': '16px'})
                    ], style={'marginBottom': '20px'}),
                ], style={'display': 'flex', 'gap': '20px'}),
                
                # Fila 3 de inputs
                html.Div([
                    html.Div([
                        html.Label("Azucar en ayunas", style={'fontWeight': 'bold', 'color': '#2c3e50'}),
                        dcc.Dropdown(id='azucar', 
                                    options=[{'label': 'Si (>120 mg/dl)', 'value': 1}, 
                                            {'label': 'No', 'value': 0}],
                                    value=0,
                                    style={'border': '2px solid #bdc3c7', 'borderRadius': '5px'})
                    ], style={'marginBottom': '20px'}),
                    
                    html.Div([
                        html.Label("Frecuencia cardiaca maxima", style={'fontWeight': 'bold', 'color': '#2c3e50'}),
                        dcc.Input(id='frecuencia', type='number', value=150, min=60, max=220,
                                 style={'width': '100%', 'padding': '10px', 'border': '2px solid #bdc3c7', 
                                       'borderRadius': '5px', 'fontSize': '16px'})
                    ], style={'marginBottom': '20px'}),
                ], style={'display': 'flex', 'gap': '20px'}),
                
                # Boton de prediccion
                html.Div([
                    html.Button('Realizar Prediccion', id='predict-button', n_clicks=0,
                               style={'width': '100%', 'padding': '15px', 'backgroundColor': '#3498db', 
                                     'color': 'white', 'border': 'none', 'borderRadius': '8px', 
                                     'fontSize': '18px', 'fontWeight': 'bold', 'cursor': 'pointer'})
                ], style={'marginTop': '20px'})
                
            ], style={'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '10px'})
            
        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px'}),
        
        # COLUMNA 2: Resultados y graficos
        html.Div([
            html.H3("RESULTADOS DE PREDICCION", 
                   style={'color': '#2c3e50', 'borderBottom': '2px solid #3498db', 'paddingBottom': '10px'}),
            
            # Mensaje inicial
            html.Div([
                html.H4("Complete los datos y haga clic en 'Realizar Prediccion'", 
                       style={'textAlign': 'center', 'color': '#7f8c8d', 'fontStyle': 'italic'}),
            ], id='mensaje-inicial', style={'marginBottom': '20px'}),
            
            # Indicador de riesgo (oculto inicialmente)
            html.Div([
                html.H4("Nivel de Riesgo Cardiaco", style={'textAlign': 'center', 'color': '#7f8c8d'}),
                html.Div("", id='riesgo-indicador', 
                        style={'textAlign': 'center', 'fontSize': '32px', 'fontWeight': 'bold', 
                              'padding': '20px', 'margin': '10px', 'borderRadius': '10px',
                              'display': 'none'})
            ], style={'marginBottom': '20px'}),
            
            # Probabilidad (oculto inicialmente)
            html.Div([
                html.H4("Probabilidad de Enfermedad Cardiaca", style={'textAlign': 'center', 'color': '#7f8c8d'}),
                dcc.Graph(id='probabilidad-gauge', style={'height': '200px', 'display': 'none'})
            ]),
            
            # Factores de riesgo (oculto inicialmente)
            html.Div([
                html.H4("Factores de Riesgo Detectados", style={'color': '#e74c3c'}),
                html.Div("", id='factores-riesgo', 
                        style={'padding': '15px', 'backgroundColor': '#fff', 'border': '2px solid #e74c3c',
                              'borderRadius': '8px', 'minHeight': '60px', 'fontSize': '16px',
                              'display': 'none'})
            ], style={'marginTop': '20px'})
            
        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px'})
    ], style={'marginBottom': '30px'}),
    
    # SEGUNDA FILA: Metricas y comparacion (ocultas inicialmente)
    html.Div([
        html.Div([
            html.H3("COMPARACION CON VALORES DE REFERENCIA", 
                   style={'color': '#2c3e50', 'textAlign': 'center', 'marginBottom': '20px'}),
            dcc.Graph(id='comparacion-grafico', style={'display': 'none'})
        ], style={'width': '65%', 'display': 'inline-block', 'padding': '10px'}),
        
        html.Div([
            html.H3("METRICAS DEL MODELO", 
                   style={'color': '#2c3e50', 'textAlign': 'center', 'marginBottom': '20px'}),
            
            html.Div([
                html.Div([
                    html.H4("Precision", style={'color': '#27ae60', 'textAlign': 'center'}),
                    html.Div(f"{accuracy:.1%}", style={'fontSize': '28px', 'fontWeight': 'bold', 'textAlign': 'center'})
                ], style={'backgroundColor': '#f8f9fa', 'padding': '15px', 'margin': '5px', 'borderRadius': '8px'}),
                
                html.Div([
                    html.H4("AUC-ROC", style={'color': '#2980b9', 'textAlign': 'center'}),
                    html.Div(f"{auc_roc:.3f}", style={'fontSize': '28px', 'fontWeight': 'bold', 'textAlign': 'center'})
                ], style={'backgroundColor': '#f8f9fa', 'padding': '15px', 'margin': '5px', 'borderRadius': '8px'}),
                
                html.Div([
                    html.H4("F1 Score", style={'color': '#f39c12', 'textAlign': 'center'}),
                    html.Div(f"{f1:.3f}", style={'fontSize': '28px', 'fontWeight': 'bold', 'textAlign': 'center'})
                ], style={'backgroundColor': '#f8f9fa', 'padding': '15px', 'margin': '5px', 'borderRadius': '8px'}),
                
                html.Div([
                    html.H4("Tiempo respuesta", style={'color': '#9b59b6', 'textAlign': 'center'}),
                    html.Div("0.0 s", id='tiempo-respuesta', 
                            style={'fontSize': '28px', 'fontWeight': 'bold', 'textAlign': 'center'})
                ], style={'backgroundColor': '#f8f9fa', 'padding': '15px', 'margin': '5px', 'borderRadius': '8px'})
            ], style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '10px'})
            
        ], style={'width': '32%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px'})
    ], id='segunda-fila', style={'display': 'none'}),
    
], style={'padding': '20px', 'fontFamily': 'Arial, sans-serif', 'backgroundColor': '#ecf0f1', 'minHeight': '100vh'})

# ==============================
# CALLBACKS PARA INTERACTIVIDAD
# ==============================

@app.callback(
    [Output('riesgo-indicador', 'children'),
     Output('riesgo-indicador', 'style'),
     Output('probabilidad-gauge', 'figure'),
     Output('factores-riesgo', 'children'),
     Output('comparacion-grafico', 'figure'),
     Output('tiempo-respuesta', 'children'),
     Output('mensaje-inicial', 'style'),
     Output('probabilidad-gauge', 'style'),
     Output('factores-riesgo', 'style'),
     Output('comparacion-grafico', 'style'),
     Output('segunda-fila', 'style')],
    [Input('predict-button', 'n_clicks')],
    [State('edad', 'value'),
     State('sexo', 'value'),
     State('presion', 'value'),
     State('colesterol', 'value'),
     State('azucar', 'value'),
     State('frecuencia', 'value')]
)
def update_predictions(n_clicks, edad, sexo, presion, colesterol, azucar, frecuencia):
    # Si no se ha hecho clic en el boton, mostrar estado inicial
    if n_clicks == 0:
        return (
            "",  # riesgo-indicador children
            {'display': 'none'},  # riesgo-indicador style
            go.Figure(),  # probabilidad-gauge figure
            "",  # factores-riesgo children
            go.Figure(),  # comparacion-grafico figure
            "0.0 s",  # tiempo-respuesta children
            {'display': 'block'},  # mensaje-inicial style
            {'display': 'none'},  # probabilidad-gauge mostrar/ocultar
            {'display': 'none'},  # factores-riesgo mostrar/ocultar
            {'display': 'none'},  # comparacion-grafico mostrar/ocultar
            {'display': 'none'}  # segunda-fila mostrar/ocultar
        )
    
    import time
    start_time = time.time()
    
    # Validar que todos los campos tengan valores
    if any(v is None for v in [edad, sexo, presion, colesterol, azucar, frecuencia]):
        return (
            "ERROR: Complete todos los campos",
            {'textAlign': 'center', 'fontSize': '32px', 'fontWeight': 'bold', 
             'color': '#e74c3c', 'backgroundColor': '#fadbd8', 
             'padding': '20px', 'borderRadius': '10px', 'border': '3px solid #e74c3c', 'display': 'block'},
            go.Figure(),
            "Por favor complete todos los campos del formulario",
            go.Figure(),
            "0.0 s",
            {'display': 'none'},
            {'display': 'block'},
            {'display': 'block'},
            {'display': 'block'},
            {'display': 'block'}
        )
    
    # Valores por defecto para otras caracteristicas
    default_values = {
        'cp': 0,        # Tipo dolor pecho
        'restecg': 0,   # ECG reposo
        'exang': 0,     # Angina ejercicio
        'oldpeak': 1.0, # Depresion ST
        'slope': 1,     # Pendiente ST
        'ca': 0,        # Vasos principales
        'thal': 2       # Thal
    }
    
    # Crear array de entrada para el modelo
    input_data = np.array([[
        edad, sexo, default_values['cp'], presion, colesterol, azucar,
        default_values['restecg'], frecuencia, default_values['exang'],
        default_values['oldpeak'], default_values['slope'], default_values['ca'],
        default_values['thal']
    ]])
    
    # Escalar los datos
    input_scaled = scaler.transform(input_data)
    
    # Predecir con Regresion Logistica
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]
    
    # Determinar nivel de riesgo
    if probability >= 0.7:
        riesgo = "ALTO RIESGO"
        color_riesgo = '#e74c3c'
        background_color = '#fadbd8'
    elif probability >= 0.4:
        riesgo = "RIESGO MODERADO"
        color_riesgo = '#f39c12'
        background_color = '#fdebd0'
    else:
        riesgo = "BAJO RIESGO"
        color_riesgo = '#27ae60'
        background_color = '#d5f5e3'
    
    # Grafico gauge de probabilidad
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Probabilidad (%)"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': color_riesgo},
            'steps': [
                {'range': [0, 30], 'color': "#27ae60"},
                {'range': [30, 70], 'color': "#f39c12"},
                {'range': [70, 100], 'color': "#e74c3c"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig_gauge.update_layout(height=200, margin=dict(t=50, b=10))
    
    # Factores de riesgo
    factores = []
    if colesterol > 240:
        factores.append("Colesterol alto")
    if azucar == 1:
        factores.append("Azucar en ayunas elevada")
    if presion > 140:
        factores.append("Presion arterial alta")
    if edad > 60:
        factores.append("Edad avanzada")
    
    if factores:
        factores_text = html.Ul([html.Li(factor) for factor in factores])
    else:
        factores_text = "No se detectaron factores de riesgo significativos"
    
    # Grafico de comparacion
    valores_paciente = [edad, presion, colesterol, frecuencia]
    valores_promedio = [54, 131, 246, 149]  # Promedios del dataset
    
    fig_comparacion = go.Figure()
    
    fig_comparacion.add_trace(go.Bar(
        name='Paciente Actual',
        x=['Edad', 'Presion', 'Colesterol', 'Frecuencia Max'],
        y=valores_paciente,
        marker_color='#3498db',
        text=valores_paciente,
        textposition='auto'
    ))
    
    fig_comparacion.add_trace(go.Bar(
        name='Valores Promedio',
        x=['Edad', 'Presion', 'Colesterol', 'Frecuencia Max'],
        y=valores_promedio,
        marker_color='#95a5a6',
        text=valores_promedio,
        textposition='auto'
    ))
    
    fig_comparacion.update_layout(
        title="Comparacion con Valores de Referencia",
        barmode='group',
        showlegend=True,
        height=400
    )
    
    # Tiempo de respuesta
    tiempo_respuesta = f"{(time.time() - start_time):.2f} s"
    
    return (
        riesgo,
        {'textAlign': 'center', 'fontSize': '32px', 'fontWeight': 'bold', 
         'color': color_riesgo, 'backgroundColor': background_color, 
         'padding': '20px', 'borderRadius': '10px', 'border': f'3px solid {color_riesgo}', 'display': 'block'},
        fig_gauge,
        factores_text,
        fig_comparacion,
        tiempo_respuesta,
        {'display': 'none'},  # Ocultar mensaje inicial
        {'display': 'block'},  # Mostrar probabilidad-gauge
        {'display': 'block'},  # Mostrar factores-riesgo
        {'display': 'block'},  # Mostrar comparacion-grafico
        {'display': 'block'}   # Mostrar segunda-fila
    )

if __name__ == '__main__':
    app.run(debug=True, port=8050)