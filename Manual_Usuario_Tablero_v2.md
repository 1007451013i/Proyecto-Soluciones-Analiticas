# Manual de Usuario del Tablero

Este documento explica cómo utilizar el tablero interactivo de predicción de enfermedad cardíaca construido en Dash (`Dashboard.py`), el cual se comunica con la API ubicada en la carpeta `Health-api/`. El tablero permite ingresar variables clínicas de un paciente y obtener una estimación inmediata del riesgo utilizando el modelo entrenado.

## 1. Requisitos previos

Para usar el tablero correctamente, asegúrese de que:

- La **API esté ejecutándose** en `http://localhost:8001/api/v1`.
- La API exponga los endpoints:
  - `GET /health`  
  - `POST /predict`
- El tablero haya sido instalado con las dependencias del archivo `requirements.txt`.
- Se cuente con un navegador moderno (Chrome, Edge o Firefox).

## 2. Acceso al tablero

1) Desde la raíz del repositorio, ejecute:
```
`python Dashboard.py`
```
2) Abra su navegador en:
```
`http://localhost:8050/`.
```

El tablero cargará automáticamente la interfaz gráfica con el formulario y los resultados.

## 3. Estructura del tablero

La interfaz se organiza en dos secciones principales:

### **Panel izquierdo – Configuración y datos del paciente**
Incluye:

- **URL de la API**: debe ingresar  
  `http://localhost:8001/api/v1`
- **Formulario clínico** con los campos:
  - `Edad`
  - `Sexo` (0 = Femenino, 1 = Masculino)
  - `Presión arterial en reposo (trestbps)`
  - `Colesterol sérico (chol)`
  - `Azúcar en ayunas (fbs)`
  - `Frecuencia cardíaca máxima (thalach)`

El tablero valida que todos los campos estén completos antes de permitir la predicción.

### **Panel derecho – Resultados**
Una vez enviada la información, se mostrarán:

- Nivel de riesgo estimado (bajo, moderado o alto)
- Gráfico tipo gauge con la probabilidad calculada
- Factores de riesgo detectados según los valores ingresados
- Comparación gráfica entre las variables del paciente y valores promedio
- Métricas del modelo (Precision, F1, AUC)
- Tiempo de respuesta de la API

## 4. Configuración de la API

El tablero requiere una URL de API válida. Debe ingresar:
```
`http://localhost:8001/api/v1`
```

Si la URL no inicia con `http://` o `https://`, el tablero mostrará un mensaje indicando:

> **ERROR: Ingrese una URL válida para la API**

El botón **Realizar Predicción** permanecerá deshabilitado hasta que la URL sea válida.

## 5. Campos del formulario

Debe completar todos los siguientes campos:

 **Edad**: valores típicos entre 20 y 100 años  
- **Sexo**:  
  - 0 = Femenino  
  - 1 = Masculino  
- **Presión arterial (trestbps)**  
- **Colesterol sérico (chol)**  
- **Azúcar en ayunas (fbs)**: 0/1  
- **Frecuencia cardíaca máxima (thalach)**
- Parámetros adicionales:
  - `cp` (tipo de dolor en el pecho: 0–3)
  - `restecg` (ECG en reposo: 0–2)
  - `exang` (angina inducida por ejercicio: 0/1)
  - `oldpeak` (depresión ST)
  - `slope` (pendiente ST: 0–2)
  - `ca` (vasos mayores coloreados: 0–4)
  - `thal` (0–3)

Los campos numéricos aceptan solo números válidos.

Si algún campo está vacío, aparecerá:

> **ERROR: Complete todos los campos**

## 6. Realizar una predicción

1) Introduzca la URL de la API.  
2) Llene todos los campos del formulario.  
3) Presione **Realizar Predicción**.

El tablero enviará automáticamente una solicitud `POST` a:
```
 `POST/predict`
```
Con un payload en este formato:
```
{
"inputs": [{
"age": 55,
"sex": 1,
"trestbps": 130,
"chol": 240,
"fbs": 0,
"thalach": 150
}]
}
```
La API devolverá una probabilidad y una predicción, que el tablero transformará en:

- Nivel de riesgo  
- Gráfico de probabilidad  
- Factores detectados  
- Comparaciones visuales  

## 7. Interpretación de resultados
El tablero clasifica el resultado en:

- **Bajo riesgo** (< 40%)
- **Riesgo moderado** (40%–70%)
- **Alto riesgo** (≥ 70%)

### **Gauge de probabilidad**
Indica visualmente el porcentaje estimado de riesgo.

### **Factores de riesgo**
El tablero analizará automáticamente:

- colesterol elevado  
- presión arterial alta  
- edad avanzada  
- frecuencia cardíaca baja  
- azúcar en ayunas positiva  

Dependiendo de los valores ingresados.

### **Comparación con valores de referencia**
Se muestra un gráfico con:

- edad  
- presión  
- colesterol  
- frecuencia cardíaca  

Comparados contra promedios clínicos típicos.

## 8. Mensajes y estados comunes
- **URL inválida**  
  > ERROR: Ingrese una URL válida para la API

- **Campos incompletos**  
  > ERROR: Complete todos los campos

- **Error en la API (status ≠ 200)**  
  Se mostrará el mensaje y código devuelto por la API.

- **Fallo de conexión**  
  > No se pudo obtener respuesta de la API  
  Verifique que la API esté ejecutándose.

## 9. Buenas prácticas
- Verifique que la API esté corriendo y accesible.
- Use datos realistas dentro de los rangos definidos.
- Mantenga la `URL de la API` sin `slashes` finales extra; el tablero normaliza internamente.
- Si el tiempo de respuesta es alto, revise la carga de la API y su hardware.

## 10. Soporte
Si el tablero no muestra resultados:

1) Verifique que la API esté activa:
```
curl -s http://localhost:8001/api/v1/health
```
2) Asegúrese de que el tablero tiene la URL correcta de la API.  
3) Revise la consola donde ejecutó `Dashboard.py` para posibles errores.  
4) Revise si su entorno virtual está correctamente activado e instalado.
