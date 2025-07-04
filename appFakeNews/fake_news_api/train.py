# train.py
import kagglehub
import os
import pandas as pd
from app.model import (
    treinar_modelo,
)  # A função treinar_modelo já foi atualizada para aceitar o DataFrame

if __name__ == "__main__":
    print("Baixando dataset do Kaggle Hub...")

    # Download do dataset
    # O caminho retornado por dataset_download é o diretório raiz do dataset
    dataset_name = "clmentbisaillon/fake-and-real-news-dataset"
    dataset_path = kagglehub.dataset_download(dataset_name)
    print(f"Dataset baixado em: {dataset_path}")

    # Definindo os caminhos completos para os arquivos CSV dentro do diretório baixado
    true_news_path = os.path.join(dataset_path, "True.csv")
    fake_news_path = os.path.join(dataset_path, "Fake.csv")

    # Verificando se os arquivos existem
    if not os.path.exists(true_news_path):
        print(
            f"Erro: Arquivo True.csv não encontrado em {true_news_path}. Verifique a estrutura do dataset baixado."
        )
        # Opcional: listar o conteúdo do diretório para depuração
        # print("Conteúdo do diretório do dataset:")
        # for root, dirs, files in os.walk(dataset_path):
        #     for name in files:
        #         print(os.path.join(root, name))
        exit()
    if not os.path.exists(fake_news_path):
        print(
            f"Erro: Arquivo Fake.csv não encontrado em {fake_news_path}. Verifique a estrutura do dataset baixado."
        )
        exit()

    # Carregar os datasets
    df_true = pd.read_csv(true_news_path)
    df_fake = pd.read_csv(fake_news_path)

    # Adicionar rótulos
    df_true["rotulo"] = "true"
    df_fake["rotulo"] = "fake"

    # Concatenar 'title' e 'text' para criar a coluna 'texto' conforme usado no seu modelo
    df_true["texto"] = df_true["title"].fillna("") + " " + df_true["text"].fillna("")
    df_fake["texto"] = df_fake["title"].fillna("") + " " + df_fake["text"].fillna("")

    # Selecionar apenas as colunas 'texto' e 'rotulo'
    df_true = df_true[["texto", "rotulo"]]
    df_fake = df_fake[["texto", "rotulo"]]

    # Concatenar os dataframes
    df_combined = pd.concat([df_true, df_fake], ignore_index=True)

    print("Iniciando treinamento do modelo com o dataset do Kaggle...")
    # Passar o DataFrame combinado diretamente para a função treinar_modelo
    # que agora aceita um DataFrame como argumento.
    modelo, vetor = treinar_modelo(df_combined)
    print("Modelo treinado e salvo com sucesso!")
