from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Suas importações locais por último
from app.database import obter_historico
from app.classifier import classificar_noticia, status_modelo




app = FastAPI(title="Fake News Detector API")

# Lista de origens permitidas (seu frontend rodará em http://localhost:5173)
origins = [
    "http://localhost",
    "http://localhost:5173", # Adicione a URL do seu frontend aqui
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Permitir todos os métodos (GET, POST, etc.)
    allow_headers=["*"], # Permitir todos os cabeçalhos
)


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
