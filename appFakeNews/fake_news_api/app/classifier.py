import joblib
from app.utils import limpar_texto
import numpy as np

modelo = joblib.load('models/modelo_naive_bayes.pkl')
vetor = joblib.load('models/vectorizer.pkl')

historico = []

def classificar_noticia(texto):
    texto_limpo = limpar_texto(texto)
    X = vetor.transform([texto_limpo])
    proba = modelo.predict_proba(X)[0]
    classe = modelo.classes_[np.argmax(proba)]
    resultado = {
        "classe": classe,
        "probabilidade": round(max(proba), 2)
    }
    historico.append({"texto": texto, **resultado})
    return resultado

def obter_historico():
    return historico

def status_modelo():
    return {
        "modelo": "Naive Bayes",
        "vetorizador": "TF-IDF",
        "treinado": True
    }
