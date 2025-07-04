# Detector de Fake News

Este projeto consiste em uma API (FastAPI) para classificar notícias como "verdadeiras" ou "falsas" e um frontend (React com TypeScript e Material UI) para interagir com essa API.

---

## Estrutura do Projeto

        .
        ├── fake_news_api/ # Backend da API (FastAPI)
        │   ├── app/
        │   │   ├── classifier.py
        │   │   ├── database.py
        │   │   ├── main.py
        │   │   ├── model.py
        │   │   └── utils.py
        │   ├── models/ # (será criado após o treinamento)
        │   ├── venv/
        │   └── requirements.txt
        ├── fake-news-frontend/
        │   ├── public/
        │   ├── src/
        │   │   ├── api.ts
        │   │   ├── App.css
        │   │   ├── App.tsx
        │   │   ├── components/
        │   │   │   ├── ApiStatus.tsx
        │   │   │   ├── HistoryList.tsx
        │   │   │   └── NewsClassifier.tsx
        │   │   ├── index.css
        │   │   └── main.tsx
        │   ├── venv/
        │   ├── package.json
        │   ├── tsconfig.json
        │   └── vite.config.ts
        └── README.md

---

## Como Configurar e Rodar o Projeto

Siga os passos abaixo para configurar e executar tanto o backend quanto o frontend.

### Pré-requisitos

- **Python 3.7+:** Certifique-se de ter o Python instalado.
- **Node.js e npm (ou Yarn):** Necessário para o frontend.
- **Credenciais do Kaggle API:** Para que o backend possa baixar o dataset de treinamento.
  - Vá para sua conta do Kaggle ([kaggle.com/your-username/account](https://www.kaggle.com/your-username/account)).
  - Gere um novo token de API (`kaggle.json`).
  - Mova `kaggle.json` para a pasta `~/.kaggle/` (Linux/macOS) ou `C:\Users\<SeuUsuario>\.kaggle\` (Windows). Crie a pasta `.kaggle` se ela não existir.

### 1. Configurar e Rodar o Backend (API)

1.  **Navegue até a pasta do backend:**

    ```bash
    cd fake_news_api
    ```

2.  **Crie um ambiente virtual (recomendado):**

    ```bash
    python -m venv venv
    ```

3.  **Ative o ambiente virtual:**

    - **Windows:**
      ```bash
      .\venv\Scripts\activate
      ```
    - **macOS/Linux:**
      ```bash
      source venv/bin/activate
      ```

4.  **Instale as dependências:**

    ```bash
    pip install -r requirements.txt
    ```

5.  **Baixe os dados do NLTK (para processamento de texto):**

    ```bash
    python -c "import nltk; nltk.download('stopwords')"
    ```

6.  **Treine o modelo de classificação:**
    Este passo fará o download do dataset do Kaggle, treinará o modelo e o salvará.

    ```bash
    python train.py
    ```

    Você verá algumas métricas de avaliação do modelo no final do treinamento.

7.  **Inicie o servidor da API:**
    ```bash
    uvicorn app.main:app --reload
    ```
    A API estará acessível em `http://127.0.0.1:8000`. Você pode testar os endpoints e ver a documentação interativa em `http://127.0.0.1:8000/docs`.

### 2. Configurar e Rodar o Frontend (React)

**Abra um NOVO terminal** e siga estes passos:

1.  **Navegue até a pasta do frontend:**

    ```bash
    cd fake-news-frontend
    ```

2.  **Instale as dependências:**

    ```bash
    npm install
    ```

    _(Se encontrar problemas, verifique se Node.js e npm estão instalados corretamente)_

3.  **Inicie o servidor de desenvolvimento do frontend:**
    ```bash
    npm run dev
    ```
    O frontend estará acessível em `http://localhost:5173/`.

### 3. Usar a Aplicação

Com ambos os servidores (API e Frontend) rodando:

- Abra seu navegador e acesse `http://localhost:5173/`.
- Você verá a interface do "Detector de Fake News".
- Verifique o "Status da API" para confirmar a conexão.
- Utilize a caixa de texto para inserir notícias e classificá-las.
- O histórico de classificações será atualizado automaticamente.

---

## Resolução de Problemas Comuns

- **Problemas de CORS:** Se o frontend não conseguir se comunicar com a API, verifique se o middleware CORS em `fake_news_api/app/main.py` está configurado corretamente para permitir `http://localhost:5173`.
- **"Tela em branco" no frontend:** Verifique o console do navegador (`F12`) e o terminal onde o `npm run dev` está rodando em busca de erros.
- **"Modelo não treinado" ou erros na API:** Verifique o terminal onde a API está rodando e certifique-se de que `python train.py` foi executado com sucesso.
- **Kaggle API:** Certifique-se de que seu arquivo `kaggle.json` está na pasta correta (`~/.kaggle/` ou `C:\Users\<SeuUsuario>\.kaggle\`).
