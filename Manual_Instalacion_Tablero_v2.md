# Manual de Instalación del Tablero y la API

Este documento describe cómo instalar y ejecutar el tablero (`Dashboard.py`) junto con la API ubicada en la carpeta `Health-api/` y el paquete del modelo contenido en `package-src/`.

## 1. Prerrequisitos

Antes de iniciar, asegúrese de contar con:

- Sistema operativo: Windows 10/11.
- `Python` 3.9+ y `pip` instalados.
- Conexión a internet para instalar dependencias.
- Permisos para descomprimir archivos `.zip`.
- Permisos para instalar paquetes y ejecutar scripts.

## 2. Preparar entorno Python

Se recomienda usar un entorno virtual para evitar conflictos con otras librerías del sistema:

- Crear y activar entorno (PowerShell):
```
python -m venv .venv
```
- Activar entorno:
```
.\.venv\Scripts\Activate.ps1
```
- Instalar dependencias del tablero desde la raiz del repositorio:
```
pip install -r requirements.txt
```

## 3. Instalación y Ejecución de la API

La API del modelo cardíaco se encuentra en la carpeta `Health-api/`.

1) Instalar las dependencias de la API:

Desde la carpeta `Health-api/` ejecute:

```
pip install -r requirements.txt
```
2) Instalar el modelo empaquetado:

El modelo entrenado se encuentra en formato wheel dentro de `Health-api/model-pkg/`.
Instálelo con:

```
pip install ./model-pkg/model_heart-0.1.0-py3-none-any.whl
```
3) Ejecutar la API

Existen dos opciones:

```
#### Opción A: ejecución directa
python app/main.py

#### Opción B: usando tox
tox run -e test_app
tox run -e run
```
4) Validar que la API está activa:

En una terminal separada:

```
curl -s http://localhost:8001/api/v1/health
```

Debe responder con un JSON que incluya:

- `name`
- `api_version`
- `model_version`

## 4. Formato de entrada para el endpoint `POST /predict`

La API recibe un JSON con las variables clínicas del paciente.

```
{
  "inputs": [{
    "age": <int>, "sex": <int>, "cp": <int>, "trestbps": <int>,
    "chol": <int>, "fbs": <int>, "restecg": <int>, "thalach": <int>,
    "exang": <int>, "oldpeak": <float>, "slope": <int>, "ca": <int>, "thal": <int>
  }]
}
```
La API devuelve: `{ "predictions": [<float|int>] }`.

## 5. Ejecución del Tablero (`Dashboard.py`)

Con la API ejecutándose:

1) Asegúrese de estar en la raíz del repositorio.
2) Active su entorno virtual si aún no está activo.
3) Ejecute:

```
`python Dashboard.py`
```

El tablero estará disponible en:

**http://localhost:8050**

En el tablero, asegúrese de configurar la URL de la API:

```
`http://localhost:8001/api/v1`.
```

## 6. Validación de la instalación completa

1) Verifique que la API responde:

```
curl -s http://localhost:8001/api/v1/health
```

2) Abra el tablero en el navegador:  
   `http://localhost:8050`

3) Ingrese los valores del paciente y haga clic en **Realizar Predicción**.

4) Confirme que se muestran:
   - Nivel de riesgo
   - Gauge de probabilidad
   - Factores de riesgo
   - Comparación con valores de referencia
   - Tiempo de respuesta

## Checklist de verificación rápida

- Activación del entorno virtual  
- Instalación de dependencias del tablero  
- Instalación de dependencias de la API  
- Instalación del modelo wheel  
- API en funcionamiento en puerto **8001**  
- Tablero activo en puerto **8050**  
- Predicciones funcionando correctamente  
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
