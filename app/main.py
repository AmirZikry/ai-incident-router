from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline

TRAINING_TEXTS = [
    "replica lag is increasing",
    "database connection pool is exhausted",
    "sql query latency is high",
    "postgres writes are failing",
    "database storage is full",
    "dns resolution is failing",
    "packet loss between services",
    "network request timed out",
    "load balancer cannot reach upstream",
    "gateway connection was refused",
    "deployment failed during release",
    "container image could not start",
    "rollback required after deploy",
    "new release caused errors",
    "build artifact is missing",
]

TRAINING_LABELS = [
    "database",
    "database",
    "database",
    "database",
    "database",
    "network",
    "network",
    "network",
    "network",
    "network",
    "deployment",
    "deployment",
    "deployment",
    "deployment",
    "deployment",
]

model=Pipeline(
    [
        ("vect",TfidfVectorizer()),
        ("clf",ComplementNB()),
    ]
)
model.fit(TRAINING_TEXTS,TRAINING_LABELS)

app=FastAPI(title="AI Incident Router", version="1.0.0")

class Incident(BaseModel):
    text: str


class Prediction(BaseModel):
    category: str

@app.post("/predict", response_model=Prediction)
def predict_incident(incident: Incident) -> Prediction:
    category=str(model.predict([incident.text])[0])
    return Prediction(category=category)