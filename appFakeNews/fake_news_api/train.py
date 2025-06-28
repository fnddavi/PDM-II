from app.model import treinar_modelo

if __name__ == "__main__":
    print("Iniciando treinamento do modelo...")
    modelo, vetor = treinar_modelo()
    print("Modelo treinado e salvo com sucesso.")
