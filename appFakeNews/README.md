# Detector de Fake News

Projeto com uma API (FastAPI) para classificar notícias como verdadeiras ou falsas, e um frontend (React + TypeScript + Material UI) para interação com a API.

---

## Estrutura do Projeto

```
.
├── fake_news_api/           # Backend (FastAPI)
│   ├── app/                 # Lógica da API
│   ├── models/              # Modelos treinados
│   ├── venv/                # Ambiente virtual (opcional)
│   └── requirements.txt     # Dependências do backend
├── fake-news-frontend/      # Frontend (React)
│   ├── public/
│   ├── src/
│   ├── venv/                # Ambiente Node (opcional)
│   ├── package.json
│   └── vite.config.ts
└── README.md
```

---

## Requisitos

- Python 3.7+
- Node.js e npm (ou Yarn)
- Conta no Kaggle com o arquivo `kaggle.json` (credenciais da API)

Para gerar o `kaggle.json`, acesse sua conta Kaggle:  
https://www.kaggle.com/your-username/account  
Depois, mova o arquivo para:

- Linux/macOS: `~/.kaggle/kaggle.json`
- Windows: `C:\Users\<SeuUsuario>\.kaggle\kaggle.json`

---

## Como Rodar

### 1. Backend (API)

```bash
cd fake_news_api
python -m venv .venv
source .venv/bin/activate     # ou .\.venv\Scripts\activate no Windows
pip install -r requirements.txt
python -c "import nltk; nltk.download('stopwords')"
python train.py               # Treina o modelo e salva os arquivos
uvicorn app.main:app --reload
```

Acesse a API em: http://127.0.0.1:8000  
Documentação Swagger: http://127.0.0.1:8000/docs

---

### 2. Frontend (React)

Em um novo terminal:

```bash
cd fake-news-frontend
npm install
npm run dev
```

Acesse a aplicação em: http://localhost:5173

---

## Como Usar

- Acesse o frontend em `http://localhost:5173`
- Verifique o status da API
- Insira uma notícia e veja a classificação
- O histórico de classificações aparecerá automaticamente

---

## Problemas Comuns

- **Erro de CORS:** Verifique se a API permite requisições do frontend (`http://localhost:5173`)
- **Tela branca:** Verifique erros no console do navegador ou terminal do frontend
- **Modelo não encontrado:** Certifique-se de que `train.py` foi executado
- **Erro com Kaggle:** Confirme que `kaggle.json` está no local correto

---

## Uso com Docker

```bash
# Navegue até a raiz do projeto
cd /caminho/para/appFakeNews

# Subir o projeto
docker compose up --build

# Em background
docker compose up -d --build

# Parar
docker compose down

# Parar e remover volumes
docker compose down -v

# Logs
docker compose logs -f
docker compose logs -f api
docker compose logs -f frontend

# Subir ou reconstruir serviços específicos
docker compose up api
docker compose build frontend

# Acessar container da API
docker compose exec api bash

# Status dos containers
docker compose ps

# Reiniciar serviços
docker compose restart api
docker compose restart frontend
```
