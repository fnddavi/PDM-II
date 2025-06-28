# appFakeNews

Este projeto `appFakeNews` é uma aplicação que visa detectar notícias falsas. Ele inclui uma API para interação, modelos de machine learning e dados para treinamento e classificação.

## Estrutura do Projeto

A estrutura principal do projeto é organizada da seguinte forma:

    appFakeNews/
    ├── fake_news_api/
    │   └── app/
    │       ├── classifier.py
    │       ├── database.py
    │       ├── main.py        # Ponto de entrada da API
    │       ├── model.py
    │       └── utils.py
    ├── data/
    │   └── dataset.csv        # Conjunto de dados utilizado para treinamento
    ├── models/
    │   └── modelo.pkl         # Modelo de machine learning serializado
    ├── venv/                  # Ambiente virtual do Python
    ├── .gitignore
    ├── README.md
    └── requirements.txt


# Dependências do projeto


## Configuração do Ambiente

Para configurar e executar o projeto, siga os passos abaixo:

1.  **Clone o repositório (se ainda não o fez):**

    ```bash
    git clone (https://github.com/fnddavi/PDM-II/tree/main/appFakeNews
    cd appFakeNews
    ```

2.  **Crie e ative o ambiente virtual:**

    É altamente recomendável usar um ambiente virtual para gerenciar as dependências do projeto.

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
