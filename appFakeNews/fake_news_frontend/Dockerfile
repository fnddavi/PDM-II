# Dockerfile para o Frontend React
FROM node:18-alpine as builder

# Define o diretório de trabalho
WORKDIR /app

# Copia os arquivos de dependências
COPY package*.json ./

# Instala as dependências
RUN npm ci

# Copia o código fonte
COPY . .

# Faz o build da aplicação
RUN npm run build

# Estágio de produção com nginx
FROM nginx:alpine

# Copia os arquivos buildados para o nginx
COPY --from=builder /app/dist /usr/share/nginx/html

# Copia configuração personalizada do nginx (opcional)
# COPY nginx.conf /etc/nginx/conf.d/default.conf

# Expõe a porta 80
EXPOSE 80

# O nginx já roda por padrão
CMD ["nginx", "-g", "daemon off;"]
