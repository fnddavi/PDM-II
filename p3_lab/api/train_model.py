# train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os
from preprocessing import preprocessar
from data_loader import carregar_dados, explorar_dataset, preparar_dados_ml


def treinar_modelo_basico(
    caminho_csv: str = "../dataset_emocoes_sintetico.csv",
    salvar_modelo=True,
    nome_modelo="modelo_emocao.pkl",
):
    """
    Função otimizada que treina um modelo de classificação de emoções.

    Args:
        caminho_csv (str): Caminho para o arquivo CSV
        salvar_modelo (bool): Se deve salvar o modelo treinado
        nome_modelo (str): Nome do arquivo do modelo

    Returns:
        Pipeline: Modelo treinado
    """
    print("🚀 INICIANDO TREINAMENTO DO MODELO")
    print("=" * 50)

    # Carregar e preparar dados
    df = carregar_dados(caminho_csv)
    explorar_dataset(df)
    X_train, X_test, y_train, y_test = preparar_dados_ml(
        df, aplicar_preprocessamento=True
    )

    # Criar pipeline otimizado
    print("\n🔧 Criando pipeline de ML...")
    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=5000,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    stop_words=None,  # Já removemos no preprocessing
                ),
            ),
            ("clf", MultinomialNB(alpha=0.1)),
        ]
    )

    # Treinar modelo
    print("🎯 Treinando modelo...")
    pipeline.fit(X_train, y_train)

    # Avaliar modelo
    print("\n📊 AVALIAÇÃO DO MODELO")
    print("-" * 30)
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Acurácia: {accuracy:.4f}")
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred))

    # Matriz de confusão
    print("\n📈 Matriz de Confusão:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # Validação cruzada
    print("\n🔄 Validação Cruzada:")
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5)
    print(f"Scores CV: {cv_scores}")
    print(f"Média CV: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    # Salvar modelo
    if salvar_modelo:
        joblib.dump(pipeline, nome_modelo)
        print(f"\n💾 Modelo salvo como: {nome_modelo}")

    print("✅ Treinamento concluído!")
    return pipeline


def comparar_modelos(caminho_csv: str = "../dataset_emocoes_sintetico.csv"):
    """
    Compara diferentes algoritmos de ML para classificação de emoções.

    Args:
        caminho_csv (str): Caminho para o arquivo CSV

    Returns:
        dict: Resultados dos diferentes modelos
    """
    print("🏆 COMPARANDO MODELOS DE ML")
    print("=" * 50)

    # Carregar dados
    df = carregar_dados(caminho_csv)
    X_train, X_test, y_train, y_test = preparar_dados_ml(
        df, aplicar_preprocessamento=True
    )

    # Definir modelos para comparar
    modelos = {
        "Naive Bayes": MultinomialNB(alpha=0.1),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(kernel="linear", random_state=42),
    }

    # Vectorizer comum
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    resultados = {}

    for nome, modelo in modelos.items():
        print(f"\n🔧 Treinando {nome}...")

        # Treinar modelo
        modelo.fit(X_train_vec, y_train)

        # Predizer
        y_pred = modelo.predict(X_test_vec)

        # Calcular métricas
        accuracy = accuracy_score(y_test, y_pred)

        # Validação cruzada
        cv_scores = cross_val_score(modelo, X_train_vec, y_train, cv=5)

        resultados[nome] = {
            "accuracy": accuracy,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "modelo": modelo,
        }

        print(f"  Acurácia: {accuracy:.4f}")
        print(f"  CV: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    # Mostrar ranking
    print("\n🥇 RANKING DOS MODELOS:")
    print("-" * 30)
    ranking = sorted(resultados.items(), key=lambda x: x[1]["accuracy"], reverse=True)
    for i, (nome, metrics) in enumerate(ranking, 1):
        print(f"{i}. {nome}: {metrics['accuracy']:.4f}")

    # Salvar melhor modelo
    melhor_modelo = ranking[0][0]
    melhor_pipeline = Pipeline(
        [("tfidf", vectorizer), ("clf", resultados[melhor_modelo]["modelo"])]
    )

    joblib.dump(
        melhor_pipeline, f"melhor_modelo_{melhor_modelo.lower().replace(' ', '_')}.pkl"
    )
    print(f"\n💾 Melhor modelo ({melhor_modelo}) salvo!")

    return resultados


def otimizar_hiperparametros(caminho_csv: str = "../dataset_emocoes_sintetico.csv"):
    """
    Otimiza hiperparâmetros usando GridSearchCV.

    Args:
        caminho_csv (str): Caminho para o arquivo CSV

    Returns:
        Pipeline: Modelo otimizado
    """
    print("⚡ OTIMIZANDO HIPERPARÂMETROS")
    print("=" * 50)

    # Carregar dados
    df = carregar_dados(caminho_csv)
    X_train, X_test, y_train, y_test = preparar_dados_ml(
        df, aplicar_preprocessamento=True
    )

    # Pipeline base
    pipeline = Pipeline([("tfidf", TfidfVectorizer()), ("clf", MultinomialNB())])

    # Parâmetros para otimizar
    param_grid = {
        "tfidf__max_features": [3000, 5000, 7000],
        "tfidf__ngram_range": [(1, 1), (1, 2), (1, 3)],
        "tfidf__min_df": [1, 2, 3],
        "clf__alpha": [0.01, 0.1, 0.5, 1.0],
    }

    print("🔍 Executando Grid Search...")
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=5, scoring="accuracy", n_jobs=-1, verbose=1
    )

    grid_search.fit(X_train, y_train)

    # Resultados
    print(f"\n🎯 Melhores parâmetros: {grid_search.best_params_}")
    print(f"🎯 Melhor score CV: {grid_search.best_score_:.4f}")

    # Avaliar no conjunto de teste
    y_pred = grid_search.best_estimator_.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"🎯 Acurácia no teste: {accuracy:.4f}")

    # Salvar modelo otimizado
    joblib.dump(grid_search.best_estimator_, "modelo_otimizado.pkl")
    print("\n💾 Modelo otimizado salvo!")

    return grid_search.best_estimator_


def testar_modelo_detalhado(modelo_path="modelo_emocao.pkl", textos_teste=None):
    """
    Testa um modelo já treinado com análise detalhada.

    Args:
        modelo_path (str): Caminho para o modelo salvo
        textos_teste (list): Lista de textos para testar

    Returns:
        dict: Resultados detalhados dos testes
    """
    if not os.path.exists(modelo_path):
        raise FileNotFoundError(f"Modelo {modelo_path} não encontrado")

    modelo = joblib.load(modelo_path)

    if textos_teste is None:
        textos_teste = [
            "Estou muito feliz hoje! Que dia maravilhoso!",
            "Tenho medo do que vai acontecer amanhã",
            "Estou com muita raiva desta situação injusta",
            "Que surpresa incrível e inesperada!",
            "Estou muito triste com essa notícia terrível",
            "Isso me dá nojo, que coisa repugnante",
        ]

    print("🧪 TESTE DETALHADO DO MODELO")
    print("=" * 50)

    resultados = {}
    for i, texto in enumerate(textos_teste, 1):
        texto_limpo = preprocessar(texto)
        predicao = modelo.predict([texto_limpo])[0]
        probabilidades = modelo.predict_proba([texto_limpo])[0]
        classes = modelo.classes_

        # Encontrar top 3 probabilidades
        top_indices = np.argsort(probabilidades)[-3:][::-1]

        print(f"\n📝 Teste {i}:")
        print(f"Texto: '{texto}'")
        print(f"Emoção predita: {predicao}")
        print(f"Confiança: {probabilidades[np.where(classes == predicao)[0][0]]:.3f}")
        print("Top 3 probabilidades:")
        for idx in top_indices:
            print(f"  {classes[idx]}: {probabilidades[idx]:.3f}")

        resultados[texto] = {
            "predicao": predicao,
            "confianca": float(probabilidades[np.where(classes == predicao)[0][0]]),
            "probabilidades": dict(zip(classes, probabilidades.astype(float))),
        }

    return resultados


# Função principal mantida para compatibilidade
def treinar():
    """
    Função principal que executa treinamento básico.
    """
    return treinar_modelo_basico()


if __name__ == "__main__":
    print("Escolha uma opção:")
    print("1. Treinamento básico")
    print("2. Comparar modelos")
    print("3. Otimizar hiperparâmetros")
    print("4. Testar modelo")

    opcao = input("Digite o número da opção: ")

    if opcao == "1":
        treinar_modelo_basico()
    elif opcao == "2":
        comparar_modelos()
    elif opcao == "3":
        otimizar_hiperparametros()
    elif opcao == "4":
        testar_modelo_detalhado()
    else:
        print("Opção inválida!")
