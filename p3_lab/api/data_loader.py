# data_loader.py
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from preprocessing import preprocessar


def carregar_dados(caminho_csv: str = "dataset_emocoes_sintetico.csv"):
    """
    Carrega dados do CSV de emoções sintéticas com formato problemático.

    Args:
        caminho_csv (str): Caminho para o arquivo CSV

    Returns:
        pd.DataFrame: DataFrame com colunas 'texto' e 'emocao'
    """
    # Verificar se o arquivo existe
    if not os.path.exists(caminho_csv):
        # Tentar encontrar na pasta pai
        parent_path = os.path.join("..", caminho_csv)
        if os.path.exists(parent_path):
            caminho_csv = parent_path
        else:
            raise FileNotFoundError(f"Arquivo {caminho_csv} não encontrado")

    print(f"🔄 Carregando CSV: {caminho_csv}")

    # Carregar CSV linha por linha devido ao formato problemático
    dados = []
    linhas_problematicas = 0

    with open(caminho_csv, "r", encoding="utf-8") as arquivo:
        linhas = arquivo.readlines()

        print(f"📄 Total de linhas no arquivo: {len(linhas)}")

        for i, linha in enumerate(linhas):
            linha = linha.strip()

            # Pular primeira linha se for header problemático
            if i == 0 and "texto,emocao" in linha:
                continue

            # Pular linhas vazias
            if not linha:
                continue

            try:
                # Limpar a linha removendo ;;;
                linha_limpa = linha.replace(";;;", "")

                # Se ainda há vírgula, dividir
                if "," in linha_limpa:
                    partes = linha_limpa.split(",")
                    if len(partes) >= 2:
                        texto = partes[0].strip()
                        emocao = partes[1].strip()

                        # Validar se não são vazios
                        if texto and emocao:
                            dados.append({"texto": texto, "emocao": emocao})
                        else:
                            linhas_problematicas += 1
                    else:
                        linhas_problematicas += 1
                else:
                    linhas_problematicas += 1

            except Exception as e:
                linhas_problematicas += 1
                if linhas_problematicas <= 5:  # Mostrar apenas primeiros 5 erros
                    print(f"⚠️ Erro na linha {i+1}: {linha[:50]}...")

    # Criar DataFrame
    if not dados:
        raise ValueError("Nenhum dado válido encontrado no CSV")

    df = pd.DataFrame(dados)

    # Remover duplicatas
    df = df.drop_duplicates()

    print(f"✅ Dados carregados:")
    print(f"   📊 {len(df)} linhas válidas")
    print(f"   ⚠️ {linhas_problematicas} linhas ignoradas")
    print(f"   🏷️ Emoções encontradas: {sorted(df['emocao'].unique())}")

    return df


def explorar_dataset(df):
    """
    Explora e exibe informações sobre o dataset.

    Args:
        df (pd.DataFrame): DataFrame com os dados
    """
    print("=" * 50)
    print("📊 ANÁLISE DO DATASET")
    print("=" * 50)
    print(f"Total de linhas: {len(df)}")
    print(f"Colunas: {df.columns.tolist()}")
    print(f"\nDistribuição das emoções:")
    distribuicao = df["emocao"].value_counts()
    print(distribuicao)

    # Mostrar percentuais
    print(f"\nPercentuais:")
    percentuais = df["emocao"].value_counts(normalize=True) * 100
    for emocao, perc in percentuais.items():
        print(f"  {emocao}: {perc:.1f}%")

    # Exemplos por emoção
    print("\n📝 EXEMPLOS DE TEXTOS POR EMOÇÃO:")
    print("-" * 40)
    for emocao in df["emocao"].unique():
        print(f"\n{emocao.upper()}:")
        exemplos = df[df["emocao"] == emocao]["texto"].head(3)
        for i, texto in enumerate(exemplos, 1):
            print(f"  {i}. {texto}")
    print("=" * 50)


def preparar_dados_ml(
    df, test_size=0.2, random_state=42, aplicar_preprocessamento=True
):
    """
    Prepara os dados para machine learning dividindo em treino e teste.

    Args:
        df (pd.DataFrame): DataFrame com os dados
        test_size (float): Proporção dos dados para teste
        random_state (int): Seed para reprodutibilidade
        aplicar_preprocessamento (bool): Se deve aplicar pré-processamento

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    # Aplicar pré-processamento se solicitado
    if aplicar_preprocessamento:
        print("🔄 Aplicando pré-processamento...")
        df_copy = df.copy()
        df_copy["texto_limpo"] = df_copy["texto"].apply(preprocessar)
        X = df_copy["texto_limpo"]
    else:
        X = df["texto"]

    y = df["emocao"]

    # Dividir em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"✅ Dados de treino: {len(X_train)}")
    print(f"✅ Dados de teste: {len(X_test)}")
    print(f"📊 Distribuição de classes no treino:")
    print(y_train.value_counts())

    return X_train, X_test, y_train, y_test


def carregar_e_preparar_completo(
    caminho_csv: str = "dataset_emocoes_sintetico.csv", explorar=True, preparar_ml=True
):
    """
    Função completa que carrega, explora e prepara os dados.

    Args:
        caminho_csv (str): Caminho para o arquivo CSV
        explorar (bool): Se deve explorar o dataset
        preparar_ml (bool): Se deve preparar para ML

    Returns:
        tuple: (df, X_train, X_test, y_train, y_test) ou apenas df
    """
    # Carregar dados
    df = carregar_dados(caminho_csv)

    # Explorar dataset
    if explorar:
        explorar_dataset(df)

    # Preparar para ML
    if preparar_ml:
        X_train, X_test, y_train, y_test = preparar_dados_ml(df)
        return df, X_train, X_test, y_train, y_test

    return df
