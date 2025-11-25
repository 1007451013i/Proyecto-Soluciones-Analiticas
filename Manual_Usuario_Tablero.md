# Manual de Usuario del Tablero

Este documento explica cómo usar el tablero de predicción de enfermedad cardíaca basado en Regresión Logística (archivo `Dashboard.py`). El tablero consume una API de salud empaquetada en `Health-api.zip` y el modelo empaquetado en `package-src.zip`.

## 1. Requisitos previos
- API de salud funcionando y accesible vía HTTP/HTTPS.
- Endpoint disponible: `POST /predict` bajo una URL base (ej.: `http://localhost:8001/api/v1`).
- Navegador web moderno (Chrome, Edge, Firefox).

## 2. Acceso al tablero
- Ejecute `python Dashboard.py`.
- Abra el navegador en `http://127.0.0.1:8050/`.

## 3. Estructura del tablero
- **Encabezado**: título y descripción del sistema.
- **Primera fila** (Formulario del paciente y configuración):
  - Configuración de API: campo “URL de la API”.
  - Formulario de paciente: campos clínicos y demográficos.
- **Segunda fila** (Resultados y métricas):
  - Indicador de riesgo, gráfico de probabilidad, factores de riesgo y comparación con valores de referencia.
  - Métricas del modelo (Precision, AUC-ROC, F1 Score) y tiempo de respuesta.

## 4. Configuración de la API
- Campo: `URL de la API` (ej.: `http://localhost:8001/api/v1`).
- Validación: el botón “Realizar Prediccion” se deshabilita si la URL no empieza por `http://` o `https://`.
- Si la URL es inválida, verá un mensaje de error pidiendo una URL válida.

## 5. Campos del formulario
Complete todos los campos antes de predecir:
- `Edad` (20–100)
- `Sexo` (0: Femenino, 1: Masculino)
- `Presión arterial` (`trestbps`)
- `Colesterol sérico` (`chol`)
- `Azúcar en ayunas` (`fbs`: 0/1)
- `Frecuencia cardíaca máxima` (`thalach`)
- Parámetros adicionales:
  - `cp` (tipo de dolor en el pecho: 0–3)
  - `restecg` (ECG en reposo: 0–2)
  - `exang` (angina inducida por ejercicio: 0/1)
  - `oldpeak` (depresión ST)
  - `slope` (pendiente ST: 0–2)
  - `ca` (vasos mayores coloreados: 0–4)
  - `thal` (0–3)

## 6. Realizar una predicción
1) Escriba la `URL de la API` (ej.: `http://localhost:8001/api/v1`).
2) Complete el formulario de paciente.
3) Haga clic en `Realizar Prediccion`.
4) El tablero enviará un `POST` a `BASE_URL/predict` con el payload:
```
{
  "inputs": [{
    "age": Edad,
    "sex": Sexo,
    "cp": cp,
    "trestbps": Presión,
    "chol": Colesterol,
    "fbs": Azúcar,
    "restecg": restecg,
    "thalach": Frecuencia,
    "exang": exang,
    "oldpeak": oldpeak,
    "slope": slope,
    "ca": ca,
    "thal": thal
  }]
}
```

## 7. Interpretación de resultados
- **Nivel de Riesgo Cardiaco**:
  - Bajo riesgo (< 40%)
  - Riesgo moderado (40%–70%)
  - Alto riesgo (≥ 70%)
- **Gráfico de probabilidad** (Gauge): porcentaje estimado.
- **Factores de riesgo**: lista simple basada en umbrales (
  p. ej., colesterol > 240, presión > 140, azúcar en ayunas = 1, edad > 60).
- **Comparación con valores de referencia**: barras comparativas de `Edad`, `Presión`, `Colesterol`, `Frecuencia Max` frente a promedios.
- **Tiempo de respuesta**: segundos entre la solicitud a la API y la respuesta.

## 8. Mensajes y estados comunes
- URL inválida: “ERROR: Ingrese una URL válida para la API”.
- Campos incompletos: “ERROR: Complete todos los campos”.
- Error de API (status != 200): se muestra el código y texto de la respuesta.
- Excepción de conexión: “No se pudo obtener respuesta de la API”.

## 9. Buenas prácticas
- Verifique que la API esté corriendo y accesible.
- Use datos realistas dentro de los rangos definidos.
- Mantenga la `URL de la API` sin `slashes` finales extra; el tablero normaliza internamente.
- Si el tiempo de respuesta es alto, revise la carga de la API y su hardware.

## 10. Soporte
- Para problemas de conexión, confirme la URL: `http://localhost:8001/api/v1` (o su entorno).
- Revise la consola donde ejecuta `Dashboard.py` para detalles.
- Asegúrese de que la API expose `POST /predict` y devuelva `{ "predictions": [...] }`.