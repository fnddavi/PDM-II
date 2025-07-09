from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict
import joblib
import os
import numpy as np
from datetime import datetime
import logging
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
from sklearn.model_selection import train_test_split
from preprocessing import preprocessar
from data_loader import carregar_dados, explorar_dataset, preparar_dados_ml

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EntradaFrase(BaseModel):
    frase: str = Field(
        ..., min_length=1, max_length=500, description="Frase para análise de emoção"
    )


class EntradaMultiplas(BaseModel):
    frases: List[str] = Field(
        ..., description="Lista de frases para análise (min: 1, max: 100)"
    )

    @validator("frases")
    def validar_lista_frases(cls, v):
        if not isinstance(v, list):
            raise ValueError("Deve ser uma lista")
        if len(v) < 1:
            raise ValueError("A lista deve conter pelo menos 1 frase")
        if len(v) > 100:
            raise ValueError("A lista não pode conter mais de 100 frases")
        if not all(isinstance(frase, str) for frase in v):
            raise ValueError("Todas as frases devem ser strings")
        return v


class SaidaEmocao(BaseModel):
    emocao_predita: str
    confianca: float
    probabilidades: Dict[str, float]
    texto_processado: str
    timestamp: datetime


class SaidaMultiplas(BaseModel):
    resultados: List[SaidaEmocao]
    total_processado: int
    tempo_total: float


class StatusModelo(BaseModel):
    modelo_carregado: bool
    caminho_modelo: str
    classes_disponiveis: List[str]
    ultima_atualizacao: str


class EstatisticasAPI(BaseModel):
    total_predicoes: int
    predicoes_por_emocao: Dict[str, int]
    tempo_medio_processamento: float
    uptime: str


class ResultadosModelo(BaseModel):
    """
    Modelo para retornar métricas de desempenho do modelo
    """

    acuracia: float
    precisao_macro: float
    precisao_micro: float
    recall_macro: float
    recall_micro: float
    f1_score_macro: float
    f1_score_micro: float
    total_amostras_teste: int
    classes: List[str]
    detalhes_por_classe: Dict[str, Dict[str, float]]
    data_avaliacao: datetime
    nome_modelo: str


# Variáveis globais
app = FastAPI(
    title="API de Detecção de Emoções",
    description="API para classificação de emoções em textos em português",
    version="2.0.0",
)

modelo = None
modelo_path = "modelo_emocao.pkl"
metricas_modelo = None  # Armazenar métricas do modelo
inicio_api = datetime.now()

estatisticas = {
    "total_predicoes": 0,
    "predicoes_por_emocao": {},
    "tempos_processamento": [],
}


def carregar_modelo():
    """
    Carrega o modelo treinado e calcula métricas se necessário.
    """
    global modelo, metricas_modelo

    try:
        if os.path.exists(modelo_path):
            modelo = joblib.load(modelo_path)
            logger.info(f"✅ Modelo carregado: {modelo_path}")

            # Calcular métricas se ainda não foram calculadas
            if metricas_modelo is None:
                calcular_metricas_modelo()

            return True
        else:
            logger.warning(f"⚠️ Modelo não encontrado: {modelo_path}")
            return False
    except Exception as e:
        logger.error(f"❌ Erro ao carregar modelo: {e}")
        return False


def calcular_metricas_modelo():
    """
    Calcula métricas de desempenho do modelo usando dados de teste.
    """
    global metricas_modelo

    try:
        logger.info("🔄 Calculando métricas do modelo...")

        # Carregar dados de teste
        df = carregar_dados("../dataset_emocoes_sintetico.csv")
        X_train, X_test, y_train, y_test = preparar_dados_ml(
            df, aplicar_preprocessamento=True
        )

        # Fazer predições
        y_pred = modelo.predict(X_test)
        y_pred_proba = modelo.predict_proba(X_test)

        # Calcular métricas
        acuracia = accuracy_score(y_test, y_pred)
        precisao_macro = precision_score(y_test, y_pred, average="macro")
        precisao_micro = precision_score(y_test, y_pred, average="micro")
        recall_macro = recall_score(y_test, y_pred, average="macro")
        recall_micro = recall_score(y_test, y_pred, average="micro")
        f1_macro = f1_score(y_test, y_pred, average="macro")
        f1_micro = f1_score(y_test, y_pred, average="micro")

        # Relatório detalhado por classe
        relatorio = classification_report(y_test, y_pred, output_dict=True)
        classes = list(modelo.classes_)

        # Detalhes por classe
        detalhes_por_classe = {}
        for classe in classes:
            if classe in relatorio:
                detalhes_por_classe[classe] = {
                    "precisao": relatorio[classe]["precision"],
                    "recall": relatorio[classe]["recall"],
                    "f1_score": relatorio[classe]["f1-score"],
                    "suporte": relatorio[classe]["support"],
                }

        # Armazenar métricas
        metricas_modelo = {
            "acuracia": acuracia,
            "precisao_macro": precisao_macro,
            "precisao_micro": precisao_micro,
            "recall_macro": recall_macro,
            "recall_micro": recall_micro,
            "f1_score_macro": f1_macro,
            "f1_score_micro": f1_micro,
            "total_amostras_teste": len(y_test),
            "classes": classes,
            "detalhes_por_classe": detalhes_por_classe,
            "data_avaliacao": datetime.now(),
            "nome_modelo": modelo_path,
        }

        logger.info(f"✅ Métricas calculadas - Acurácia: {acuracia:.4f}")

    except Exception as e:
        logger.error(f"❌ Erro ao calcular métricas: {e}")
        metricas_modelo = None


def atualizar_estatisticas(emocao: str, tempo_processamento: float):
    """
    Atualiza estatísticas da API.
    """
    estatisticas["total_predicoes"] += 1

    if emocao not in estatisticas["predicoes_por_emocao"]:
        estatisticas["predicoes_por_emocao"][emocao] = 0
    estatisticas["predicoes_por_emocao"][emocao] += 1

    estatisticas["tempos_processamento"].append(tempo_processamento)

    # Manter apenas últimos 1000 tempos
    if len(estatisticas["tempos_processamento"]) > 1000:
        estatisticas["tempos_processamento"] = estatisticas["tempos_processamento"][
            -1000:
        ]


@app.on_event("startup")
async def startup_event():
    """
    Evento executado na inicialização da API.
    """
    logger.info("🚀 Iniciando API de Detecção de Emoções")
    carregar_modelo()


@app.get("/", summary="Página inicial")
async def root():
    """
    Endpoint raiz da API.
    """
    return {
        "message": "API de Detecção de Emoções",
        "version": "2.0.0",
        "endpoints": {
            "detectar": "/api/detectar-emocao",
            "multiplas": "/api/detectar-multiplas",
            "resultados": "/api/resultados",
            "status": "/api/status",
            "estatisticas": "/api/estatisticas",
        },
    }


@app.post(
    "/api/detectar-emocao",
    response_model=SaidaEmocao,
    summary="Detectar emoção em frase",
)
async def detectar_emocao(entrada: EntradaFrase):
    """
    Detecta a emoção predominante em uma frase.

    Args:
        entrada: Objeto contendo a frase para análise

    Returns:
        SaidaEmocao: Emoção detectada com probabilidades
    """
    if modelo is None:
        raise HTTPException(status_code=503, detail="Modelo não carregado")

    try:
        inicio = datetime.now()

        # Preprocessar texto
        texto_processado = preprocessar(entrada.frase)

        if not texto_processado.strip():
            raise HTTPException(
                status_code=400, detail="Texto vazio após preprocessamento"
            )

        # Predição
        predicao = modelo.predict([texto_processado])[0]
        probabilidades = modelo.predict_proba([texto_processado])[0]

        # Criar dicionário de probabilidades
        classes = modelo.classes_
        prob_dict = {
            classe: float(prob) for classe, prob in zip(classes, probabilidades)
        }

        # Confiança (probabilidade máxima)
        confianca = float(max(probabilidades))

        # Calcular tempo de processamento
        tempo_processamento = (datetime.now() - inicio).total_seconds()

        # Atualizar estatísticas
        atualizar_estatisticas(predicao, tempo_processamento)

        return SaidaEmocao(
            emocao_predita=predicao,
            confianca=confianca,
            probabilidades=prob_dict,
            texto_processado=texto_processado,
            timestamp=datetime.now(),
        )

    except Exception as e:
        logger.error(f"Erro na detecção: {e}")
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")


@app.post(
    "/api/detectar-multiplas",
    response_model=SaidaMultiplas,
    summary="Detectar emoções em múltiplas frases",
)
async def detectar_multiplas(entrada: EntradaMultiplas):
    """
    Detecta emoções em múltiplas frases.

    Args:
        entrada: Lista de frases para análise

    Returns:
        SaidaMultiplas: Resultados para todas as frases
    """
    if modelo is None:
        raise HTTPException(status_code=503, detail="Modelo não carregado")

    try:
        inicio_total = datetime.now()
        resultados = []

        for frase in entrada.frases:
            try:
                inicio = datetime.now()

                # Preprocessar texto
                texto_processado = preprocessar(frase)

                if not texto_processado.strip():
                    continue  # Pular frases vazias

                # Predição
                predicao = modelo.predict([texto_processado])[0]
                probabilidades = modelo.predict_proba([texto_processado])[0]

                # Criar dicionário de probabilidades
                classes = modelo.classes_
                prob_dict = {
                    classe: float(prob) for classe, prob in zip(classes, probabilidades)
                }

                # Confiança
                confianca = float(max(probabilidades))

                # Tempo de processamento
                tempo_processamento = (datetime.now() - inicio).total_seconds()

                # Atualizar estatísticas
                atualizar_estatisticas(predicao, tempo_processamento)

                resultados.append(
                    SaidaEmocao(
                        emocao_predita=predicao,
                        confianca=confianca,
                        probabilidades=prob_dict,
                        texto_processado=texto_processado,
                        timestamp=datetime.now(),
                    )
                )

            except Exception as e:
                logger.warning(f"Erro ao processar frase '{frase}': {e}")
                continue

        tempo_total = (datetime.now() - inicio_total).total_seconds()

        return SaidaMultiplas(
            resultados=resultados,
            total_processado=len(resultados),
            tempo_total=tempo_total,
        )

    except Exception as e:
        logger.error(f"Erro no processamento múltiplo: {e}")
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")


@app.get(
    "/api/resultados",
    response_model=ResultadosModelo,
    summary="Métricas de desempenho do modelo",
)
async def obter_resultados():
    """
    Retorna métricas de desempenho do modelo (acurácia, precisão, etc.).

    Returns:
        ResultadosModelo: Métricas detalhadas do modelo
    """
    if modelo is None:
        raise HTTPException(status_code=503, detail="Modelo não carregado")

    if metricas_modelo is None:
        # Tentar recalcular métricas
        calcular_metricas_modelo()

        if metricas_modelo is None:
            raise HTTPException(status_code=503, detail="Métricas não disponíveis")

    try:
        return ResultadosModelo(
            acuracia=metricas_modelo["acuracia"],
            precisao_macro=metricas_modelo["precisao_macro"],
            precisao_micro=metricas_modelo["precisao_micro"],
            recall_macro=metricas_modelo["recall_macro"],
            recall_micro=metricas_modelo["recall_micro"],
            f1_score_macro=metricas_modelo["f1_score_macro"],
            f1_score_micro=metricas_modelo["f1_score_micro"],
            total_amostras_teste=metricas_modelo["total_amostras_teste"],
            classes=metricas_modelo["classes"],
            detalhes_por_classe=metricas_modelo["detalhes_por_classe"],
            data_avaliacao=metricas_modelo["data_avaliacao"],
            nome_modelo=metricas_modelo["nome_modelo"],
        )

    except Exception as e:
        logger.error(f"Erro ao obter resultados: {e}")
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")


@app.get("/api/status", response_model=StatusModelo, summary="Status do modelo")
async def obter_status():
    """
    Retorna status do modelo carregado.

    Returns:
        StatusModelo: Informações sobre o modelo
    """
    try:
        if modelo is None:
            return StatusModelo(
                modelo_carregado=False,
                caminho_modelo=modelo_path,
                classes_disponiveis=[],
                ultima_atualizacao="N/A",
            )

        # Obter informações do arquivo
        stat_info = os.stat(modelo_path)
        ultima_modificacao = datetime.fromtimestamp(stat_info.st_mtime)

        return StatusModelo(
            modelo_carregado=True,
            caminho_modelo=modelo_path,
            classes_disponiveis=list(modelo.classes_),
            ultima_atualizacao=ultima_modificacao.strftime("%Y-%m-%d %H:%M:%S"),
        )

    except Exception as e:
        logger.error(f"Erro ao obter status: {e}")
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")


@app.get(
    "/api/estatisticas", response_model=EstatisticasAPI, summary="Estatísticas da API"
)
async def obter_estatisticas():
    """
    Retorna estatísticas de uso da API.

    Returns:
        EstatisticasAPI: Estatísticas de uso
    """
    try:
        # Calcular tempo médio de processamento
        tempo_medio = 0
        if estatisticas["tempos_processamento"]:
            tempo_medio = sum(estatisticas["tempos_processamento"]) / len(
                estatisticas["tempos_processamento"]
            )

        # Calcular uptime
        uptime = datetime.now() - inicio_api
        uptime_str = f"{uptime.days} dias, {uptime.seconds // 3600} horas, {(uptime.seconds % 3600) // 60} minutos"

        return EstatisticasAPI(
            total_predicoes=estatisticas["total_predicoes"],
            predicoes_por_emocao=estatisticas["predicoes_por_emocao"],
            tempo_medio_processamento=tempo_medio,
            uptime=uptime_str,
        )

    except Exception as e:
        logger.error(f"Erro ao obter estatísticas: {e}")
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")


@app.get("/api/health", summary="Health check")
async def health_check():
    """
    Verifica se a API está funcionando.

    Returns:
        dict: Status da API
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "modelo_carregado": modelo is not None,
        "metricas_disponiveis": metricas_modelo is not None,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
