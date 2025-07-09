# preprocessing.py
import re
import string
from collections import Counter
import pandas as pd

# Tentar importar NLTK, mas sempre usar fallback se houver problema
try:
    import nltk

    # Tentar baixar recursos necessários
    try:
        nltk.download("punkt_tab", quiet=True)
        nltk.download("stopwords", quiet=True)
        nltk.download("rslp", quiet=True)
    except:
        pass

    # Verificar se recursos estão disponíveis
    from nltk.corpus import stopwords
    from nltk.stem import RSLPStemmer
    from nltk.tokenize import word_tokenize

    # Teste básico para verificar se funciona
    word_tokenize("teste")
    stopwords.words("portuguese")

    print("✅ NLTK funcionando corretamente")
    NLTK_AVAILABLE = True
    stopwords_pt = set(stopwords.words("portuguese"))
    stemmer = RSLPStemmer()

except Exception as e:
    print(f"⚠️ NLTK não disponível ({e}), usando versão básica")
    NLTK_AVAILABLE = False

    # Stopwords manuais básicas em português
    stopwords_pt = {
        "a",
        "ao",
        "aos",
        "aquela",
        "aquelas",
        "aquele",
        "aqueles",
        "aquilo",
        "as",
        "até",
        "com",
        "como",
        "da",
        "das",
        "de",
        "dela",
        "delas",
        "dele",
        "deles",
        "depois",
        "do",
        "dos",
        "e",
        "ela",
        "elas",
        "ele",
        "eles",
        "em",
        "entre",
        "era",
        "eram",
        "essa",
        "essas",
        "esse",
        "esses",
        "esta",
        "estão",
        "estar",
        "estas",
        "estava",
        "estavam",
        "este",
        "esteja",
        "estejam",
        "estejamos",
        "estes",
        "esteve",
        "estive",
        "estivemos",
        "estiver",
        "estivera",
        "estiveram",
        "estiverem",
        "estivermos",
        "estivesse",
        "estivessem",
        "estivéramos",
        "estivéssemos",
        "estou",
        "está",
        "estávamos",
        "estão",
        "eu",
        "foi",
        "fomos",
        "for",
        "fora",
        "foram",
        "forem",
        "formos",
        "fosse",
        "fossem",
        "fui",
        "fôramos",
        "fôssemos",
        "haja",
        "hajam",
        "hajamos",
        "havemos",
        "haver",
        "havia",
        "hei",
        "houve",
        "houvemos",
        "houver",
        "houvera",
        "houveram",
        "houverei",
        "houverem",
        "houveremos",
        "houveria",
        "houveriam",
        "houvermos",
        "houveríamos",
        "houvesse",
        "houvessem",
        "houvéramos",
        "houvéssemos",
        "há",
        "hão",
        "isso",
        "isto",
        "já",
        "lhe",
        "lhes",
        "mais",
        "mas",
        "me",
        "mesmo",
        "meu",
        "meus",
        "minha",
        "minhas",
        "muito",
        "na",
        "nas",
        "nem",
        "no",
        "nos",
        "nossa",
        "nossas",
        "nosso",
        "nossos",
        "num",
        "numa",
        "não",
        "nós",
        "o",
        "os",
        "ou",
        "para",
        "pela",
        "pelas",
        "pelo",
        "pelos",
        "por",
        "qual",
        "quando",
        "que",
        "quem",
        "se",
        "seja",
        "sejam",
        "sejamos",
        "sem",
        "ser",
        "será",
        "serão",
        "serei",
        "seremos",
        "seria",
        "seriam",
        "seríamos",
        "sou",
        "sua",
        "suas",
        "são",
        "só",
        "também",
        "te",
        "tem",
        "temos",
        "tenha",
        "tenham",
        "tenhamos",
        "tenho",
        "ter",
        "terei",
        "teremos",
        "teria",
        "teriam",
        "teríamos",
        "teu",
        "teus",
        "teve",
        "tinha",
        "tinham",
        "tínhamos",
        "tive",
        "tivemos",
        "tiver",
        "tivera",
        "tiveram",
        "tiverem",
        "tivermos",
        "tivesse",
        "tivessem",
        "tivéramos",
        "tivéssemos",
        "tu",
        "tua",
        "tuas",
        "tém",
        "tínhamos",
        "um",
        "uma",
        "você",
        "vocês",
        "vos",
        "à",
        "às",
        "éramos",
        "é",
    }

# Adicionar stopwords personalizadas
stopwords_customizadas = {
    "muito",
    "bem",
    "mais",
    "menos",
    "assim",
    "aqui",
    "ali",
    "lá",
    "ontem",
    "hoje",
    "amanhã",
    "depois",
    "antes",
    "sempre",
    "nunca",
    "agora",
    "ainda",
    "então",
    "apenas",
    "sobre",
    "contra",
    "durante",
    "dentro",
    "fora",
    "perto",
    "longe",
    "cima",
    "baixo",
    "meio",
    "vez",
    "vezes",
    "dia",
    "dias",
    "ano",
    "anos",
    "hora",
    "horas",
    "minuto",
    "minutos",
    "segundo",
    "segundos",
    "coisa",
    "coisas",
    "pessoa",
    "pessoas",
    "lugar",
    "lugares",
    "tempo",
    "tempos",
    "forma",
    "formas",
    "modo",
    "modos",
    "vida",
    "mundo",
    "casa",
    "trabalho",
    "parte",
    "partes",
    "lado",
    "lados",
    "vez",
    "vezes",
}
stopwords_pt.update(stopwords_customizadas)


def tokenizar_basico(texto):
    """
    Tokenização básica sem NLTK.

    Args:
        texto (str): Texto para tokenizar

    Returns:
        list: Lista de tokens
    """
    # Converter para minúsculas e remover pontuação
    texto_limpo = re.sub(r"[^\w\s]", " ", texto.lower())
    # Extrair palavras (letras e números)
    tokens = re.findall(r"\b\w+\b", texto_limpo)
    return tokens


def stemmer_basico(palavra):
    """
    Stemming básico manual para português.

    Args:
        palavra (str): Palavra para aplicar stemming

    Returns:
        str: Palavra com stemming aplicado
    """
    # Sufixos comuns em português (ordenados por tamanho)
    sufixos = [
        "mente",
        "ação",
        "ções",
        "ador",
        "adora",
        "adores",
        "adoras",
        "izar",
        "ável",
        "ível",
        "oso",
        "osa",
        "osos",
        "osas",
        "ado",
        "ada",
        "ados",
        "adas",
        "ando",
        "endo",
        "indo",
        "ar",
        "er",
        "ir",
        "or",
        "ez",
        "eza",
        "al",
        "ais",
        "ão",
        "ões",
        "ão",
        "ões",
        "ão",
        "ões",
    ]

    palavra_lower = palavra.lower()

    # Aplicar stemming apenas se palavra for longa o suficiente
    if len(palavra_lower) <= 3:
        return palavra_lower

    for sufixo in sufixos:
        if palavra_lower.endswith(sufixo) and len(palavra_lower) > len(sufixo) + 2:
            return palavra_lower[: -len(sufixo)]

    return palavra_lower


def preprocessar(texto: str, aplicar_stemming=False, manter_emojis=False) -> str:
    """
    Preprocessa texto para análise de emoções.

    Args:
        texto (str): Texto para preprocessar
        aplicar_stemming (bool): Se deve aplicar stemming
        manter_emojis (bool): Se deve manter emojis

    Returns:
        str: Texto preprocessado
    """
    if not isinstance(texto, str) or not texto.strip():
        return ""

    # Converter para minúsculas
    texto = texto.lower()

    # Manter ou remover emojis
    if not manter_emojis:
        # Remover emojis (padrão Unicode)
        texto = re.sub(
            r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U000024C2-\U0001F251]+",
            "",
            texto,
        )

    # Remover pontuação, mas manter espaços
    texto = re.sub(r"[^\w\s]", " ", texto)

    # Remover números
    texto = re.sub(r"\d+", "", texto)

    # Remover espaços extras
    texto = re.sub(r"\s+", " ", texto).strip()

    # Tokenização
    if NLTK_AVAILABLE:
        try:
            tokens = word_tokenize(texto, language="portuguese")
        except:
            # Se NLTK falhar, usar tokenização básica
            tokens = tokenizar_basico(texto)
    else:
        tokens = tokenizar_basico(texto)

    # Remover stopwords e tokens muito curtos
    tokens = [token for token in tokens if token not in stopwords_pt and len(token) > 2]

    # Aplicar stemming se solicitado
    if aplicar_stemming:
        if NLTK_AVAILABLE:
            try:
                tokens = [stemmer.stem(token) for token in tokens]
            except:
                # Se NLTK falhar, usar stemming básico
                tokens = [stemmer_basico(token) for token in tokens]
        else:
            tokens = [stemmer_basico(token) for token in tokens]

    return " ".join(tokens)


def preprocessar_basico(texto: str) -> str:
    """
    Versão básica do preprocessamento para compatibilidade.

    Args:
        texto (str): Texto para preprocessar

    Returns:
        str: Texto preprocessado
    """
    return preprocessar(texto, aplicar_stemming=False, manter_emojis=False)


def analisar_vocabulario(textos, top_n=20):
    """
    Analisa o vocabulário de uma lista de textos.

    Args:
        textos: Lista de textos
        top_n: Número de palavras mais frequentes

    Returns:
        Counter: Contador de palavras
    """
    print(f"📊 ANÁLISE DE VOCABULÁRIO (Top {top_n})")
    print("=" * 40)

    todas_palavras = []
    for texto in textos:
        texto_limpo = preprocessar(texto)
        palavras = texto_limpo.split()
        todas_palavras.extend(palavras)

    contador = Counter(todas_palavras)

    print(f"Total de palavras únicas: {len(contador)}")
    print(f"Total de palavras: {sum(contador.values())}")
    print(f"\nPalavras mais frequentes:")

    for palavra, freq in contador.most_common(top_n):
        print(f"  {palavra}: {freq}")

    return contador


def analisar_por_emocao(df):
    """
    Analisa vocabulário por emoção.

    Args:
        df: DataFrame com colunas 'texto' e 'emocao'
    """
    print("📊 ANÁLISE POR EMOÇÃO")
    print("=" * 40)

    for emocao in df["emocao"].unique():
        textos_emocao = df[df["emocao"] == emocao]["texto"].tolist()
        print(f"\n🏷️ {emocao.upper()} ({len(textos_emocao)} textos)")
        print("-" * 20)

        analisar_vocabulario(textos_emocao, top_n=10)


def encontrar_palavras_discriminativas(df, min_freq=3):
    """
    Encontra palavras que são discriminativas para cada emoção.

    Args:
        df: DataFrame com dados
        min_freq: Frequência mínima para considerar palavra

    Returns:
        dict: Palavras discriminativas por emoção
    """
    print("🔍 PALAVRAS DISCRIMINATIVAS")
    print("=" * 40)

    # Análise por emoção
    vocabulario_por_emocao = {}

    for emocao in df["emocao"].unique():
        textos_emocao = df[df["emocao"] == emocao]["texto"].tolist()

        todas_palavras = []
        for texto in textos_emocao:
            texto_limpo = preprocessar(texto)
            todas_palavras.extend(texto_limpo.split())

        contador = Counter(todas_palavras)
        # Filtrar palavras com frequência mínima
        palavras_freq = {
            palavra: freq for palavra, freq in contador.items() if freq >= min_freq
        }

        vocabulario_por_emocao[emocao] = palavras_freq

    # Encontrar palavras discriminativas
    palavras_discriminativas = {}

    for emocao in vocabulario_por_emocao:
        palavras_emocao = set(vocabulario_por_emocao[emocao].keys())

        # Palavras que aparecem em outras emoções
        palavras_outras = set()
        for outra_emocao in vocabulario_por_emocao:
            if outra_emocao != emocao:
                palavras_outras.update(vocabulario_por_emocao[outra_emocao].keys())

        # Palavras exclusivas ou muito mais frequentes
        discriminativas = []
        for palavra in palavras_emocao:
            freq_atual = vocabulario_por_emocao[emocao][palavra]

            # Verificar se é exclusiva ou muito mais frequente
            if palavra not in palavras_outras:
                discriminativas.append((palavra, freq_atual, "exclusiva"))
            else:
                # Calcular ratio com outras emoções
                max_freq_outras = max(
                    [
                        vocabulario_por_emocao[outra].get(palavra, 0)
                        for outra in vocabulario_por_emocao
                        if outra != emocao
                    ]
                )

                if max_freq_outras > 0:
                    ratio = freq_atual / max_freq_outras
                    if ratio > 2.0:  # Pelo menos 2x mais frequente
                        discriminativas.append(
                            (palavra, freq_atual, f"ratio:{ratio:.1f}")
                        )

        # Ordenar por frequência
        discriminativas.sort(key=lambda x: x[1], reverse=True)
        palavras_discriminativas[emocao] = discriminativas[:10]

        print(f"\n🏷️ {emocao.upper()}:")
        for palavra, freq, tipo in discriminativas[:5]:
            print(f"  {palavra}: {freq} ({tipo})")

    return palavras_discriminativas


def preprocessar_dataset_completo(df, salvar_cache=True):
    """
    Preprocessa dataset completo com opção de cache.

    Args:
        df: DataFrame com dados
        salvar_cache: Se deve salvar cache

    Returns:
        pd.DataFrame: DataFrame preprocessado
    """
    print("🔄 Preprocessando dataset completo...")

    df_processado = df.copy()
    df_processado["texto_limpo"] = df_processado["texto"].apply(preprocessar_basico)

    # Remover linhas com texto vazio após preprocessamento
    df_processado = df_processado[df_processado["texto_limpo"].str.strip() != ""]

    print(f"✅ Dataset preprocessado: {len(df_processado)} linhas")

    if salvar_cache:
        try:
            df_processado.to_csv("dataset_preprocessado.csv", index=False)
            print("💾 Cache salvo em 'dataset_preprocessado.csv'")
        except Exception as e:
            print(f"⚠️ Erro ao salvar cache: {e}")

    return df_processado
