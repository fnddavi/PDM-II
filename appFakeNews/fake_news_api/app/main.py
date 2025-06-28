from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.classifier import classificar_noticia, obter_historico, status_modelo

app = FastAPI(title="Fake News Detector API")


class Noticia(BaseModel):
    texto: str


@app.get("/api/status")
def get_status():
    return status_modelo()


@app.get("/api/historico")
def get_historico():
    return obter_historico()


@app.post("/api/classificar-noticia")
def post_classificar_noticia(noticia: Noticia):
    if not noticia.texto.strip():
        raise HTTPException(status_code=400, detail="Texto não pode ser vazio.")
    return classificar_noticia(noticia.texto)
