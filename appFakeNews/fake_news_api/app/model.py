# app/model.py
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from app.utils import limpar_texto, criar_vetorizador
import os
import kagglehub
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)  # Importar métricas


def treinar_modelo(df_or_path=None):
    if df_or_path is None:
        print("Baixando dataset do Kaggle Hub...")
        dataset_name = "clmentbisaillon/fake-and-real-news-dataset"
        dataset_path = kagglehub.dataset_download(dataset_name)
        print(f"Dataset baixado em: {dataset_path}")

        true_news_path = os.path.join(dataset_path, "True.csv")
        fake_news_path = os.path.join(dataset_path, "Fake.csv")

        if not os.path.exists(true_news_path):
            raise FileNotFoundError(
                f"Arquivo True.csv não encontrado em {true_news_path}. Verifique a estrutura do dataset baixado."
            )
        if not os.path.exists(fake_news_path):
            raise FileNotFoundError(
                f"Arquivo Fake.csv não encontrado em {fake_news_path}. Verifique a estrutura do dataset baixado."
            )

        df_true = pd.read_csv(true_news_path)
        df_fake = pd.read_csv(fake_news_path)

        df_true["rotulo"] = "true"
        df_fake["rotulo"] = "fake"

        df_true["texto"] = (
            df_true["title"].fillna("") + " " + df_true["text"].fillna("")
        )
        df_fake["texto"] = (
            df_fake["title"].fillna("") + " " + df_fake["text"].fillna("")
        )

        df = pd.concat(
            [df_true[["texto", "rotulo"]], df_fake[["texto", "rotulo"]]],
            ignore_index=True,
        )

    elif isinstance(df_or_path, pd.DataFrame):
        df = df_or_path
    elif isinstance(df_or_path, str):
        df = pd.read_csv(df_or_path)
    else:
        raise ValueError(
            "treinar_modelo espera um DataFrame, um caminho de arquivo ou None para baixar do Kaggle."
        )

    df["texto_limpo"] = df["texto"].apply(limpar_texto)

    vetor, X = criar_vetorizador(df["texto_limpo"])
    y = df["rotulo"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    modelo = MultinomialNB()
    modelo.fit(X_train, y_train)

    # --- Nova seção de avaliação do modelo ---
    y_pred = modelo.predict(X_test)

    print("\n--- Avaliação do Modelo no Conjunto de Teste ---")
    print(f"Acurácia: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precisão (Fake): {precision_score(y_test, y_pred, pos_label='fake'):.4f}")
    print(f"Recall (Fake): {recall_score(y_test, y_pred, pos_label='fake'):.4f}")
    print(f"F1-Score (Fake): {f1_score(y_test, y_pred, pos_label='fake'):.4f}")
    print(f"Precisão (True): {precision_score(y_test, y_pred, pos_label='true'):.4f}")
    print(f"Recall (True): {recall_score(y_test, y_pred, pos_label='true'):.4f}")
    print(f"F1-Score (True): {f1_score(y_test, y_pred, pos_label='true'):.4f}")
    print("------------------------------------------\n")
    # --- Fim da nova seção ---

    os.makedirs("models", exist_ok=True)
    joblib.dump(modelo, "models/modelo_naive_bayes.pkl")
    joblib.dump(vetor, "models/vectorizer.pkl")
    return modelo, vetor
