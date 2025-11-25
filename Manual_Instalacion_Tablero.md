# Manual de Instalación del Tablero

Este documento describe cómo instalar y ejecutar el tablero (`Dashboard.py`) junto con la API empaquetada en `Health-api.zip` y el código del modelo en `package-src.zip`.

## 1. Prerrequisitos
- Sistema operativo: Windows 10/11.
- `Python` 3.9+ y `pip` instalados.
- Conexión a internet para instalar dependencias.
- Permisos para descomprimir archivos `.zip`.

## 2. Preparar entorno Python
Se recomienda usar un entorno virtual:
- Crear y activar entorno (PowerShell):
```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
- Instalar dependencias del tablero:
```
pip install -r requirements.txt
```

## 3. Desplegar la API (`Health-api.zip`)
1) Descomprima `Health-api.zip` en una carpeta, por ejemplo `c:\HealthAPI`.
2) Dentro de esa carpeta, instale sus dependencias:
```
pip install -r requirements.txt
```
3) Arranque de la API (comandos exactos):
```
# Instale el paquete de modelo incluido
pip install ./model-pkg/model_heart-0.1.0-py3-none-any.whl

# Ejecute pruebas y arranque del servidor (según tox.ini)
tox run -e test_app
tox run -e run
# Alternativa directa sin tox
python app/main.py
```
4) Validación rápida de salud:
```
curl -s http://localhost:8001/api/v1/health
```
5) El tablero espera:
- Una **URL base** `http://localhost:8001/api/v1`.
- Un endpoint `POST /predict` que reciba:
```
{
  "inputs": [{
    "age": <int>, "sex": <int>, "cp": <int>, "trestbps": <int>,
    "chol": <int>, "fbs": <int>, "restecg": <int>, "thalach": <int>,
    "exang": <int>, "oldpeak": <float>, "slope": <int>, "ca": <int>, "thal": <int>
  }]
}
```
  y devuelva `{ "predictions": [<float|int>] }`.

## 6. Validación de instalación
- Con la API corriendo, complete el formulario del paciente y haga clic en `Realizar Prediccion`.
- Debería ver el riesgo, el gauge de probabilidad, factores de riesgo y la comparación con valores de referencia.
- Revise el “Tiempo de respuesta” para confirmar la comunicación con la API.

## Checklist de verificación rápida
- Salud de la API:
```
curl -s http://localhost:8001/api/v1/health
```
  - La respuesta debe incluir `name`, `api_version` y `model_version`.
- Predicción de prueba directa a la API:
```
curl -s -X POST http://localhost:8001/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [
      {
        "age": 57,
        "sex": 1,
        "cp": 3,
        "trestbps": 110,
        "chol": 206,
        "fbs": 0,
        "restecg": 2,
        "thalach": 108,
        "exang": 1,
        "oldpeak": 0.0,
        "slope": 1,
        "ca": 1,
        "thal": 0
      }
    ]
  }'
```
  - Esperado: `status 200` y estructura `{ "predictions": [...], "errors": null, "version": "..." }`.
- Tablero operativo:
  - Ejecutar `python Dashboard.py`.
  - Configurar `URL de la API`: `http://localhost:8001/api/v1`.
  - Realizar una predicción y verificar que el “Tiempo de respuesta” aparece en pantalla.