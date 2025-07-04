# Imagem base com Node.js
FROM node:20.10.0-slim

# Diretório de trabalho dentro do contêiner
WORKDIR /server

# Copiar o package.json e o package-lock.json para dentro do contêiner
COPY package*.json ./

# Instalar as dependências do projeto
RUN npm install

# Copy to root
COPY package.json .
COPY package-lock.json .
COPY tsconfig.json .

# Instalar as dependências do projeto
RUN npm install

# Copiar o restante dos arquivos do projeto para o contêiner
COPY src ./src

# Expor a porta que a aplicação vai utilizar
EXPOSE 5002

# Comando para rodar a aplicação
CMD ["npm", "start"]