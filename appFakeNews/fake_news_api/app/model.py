import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from app.utils import limpar_texto, criar_vetorizador


def treinar_modelo(caminho_dataset="data/dataset.csv"):
    df = pd.read_csv(caminho_dataset)
    df["texto_limpo"] = df["texto"].apply(limpar_texto)

    vetor, X = criar_vetorizador(df["texto_limpo"])
    y = df["rotulo"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    modelo = MultinomialNB()
    modelo.fit(X_train, y_train)

    joblib.dump(modelo, "models/modelo_naive_bayes.pkl")
    joblib.dump(vetor, "models/vectorizer.pkl")
    return modelo, vetor
