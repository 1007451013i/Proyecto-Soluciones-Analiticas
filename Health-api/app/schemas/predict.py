from typing import Any, List, Optional

from pydantic import BaseModel

# Define input schema locally to match model expected features
class DataInputSchema(BaseModel):
    age: Optional[int]
    sex: Optional[int]
    cp: Optional[int]
    trestbps: Optional[int]
    chol: Optional[int]
    fbs: Optional[int]
    restecg: Optional[int]
    thalach: Optional[int]
    exang: Optional[int]
    oldpeak: Optional[float]
    slope: Optional[int]
    ca: Optional[int]
    thal: Optional[int]

# Esquema de los resultados de predicción
class PredictionResults(BaseModel):
    errors: Optional[Any]
    version: str
    predictions: Optional[List[float]]

# Esquema para inputs múltiples
class MultipleDataInputs(BaseModel):
    inputs: List[DataInputSchema]

    class Config:
        schema_extra = {
            "example": {
                "inputs": [
                    {
                        "age": 57,
                        "sex": 1,
                        "cp": 3,
                        "trestbps": 110,
                        "chol": 206,
                        "fbs": 0,
                        "restecg": 2,
                        "thalach":108,
                        "exang":1,
                        "oldpeak":0.0,
                        "slope":1,
                        "ca":1,
                        "thal":0
                    }
                ]
            }
        }
