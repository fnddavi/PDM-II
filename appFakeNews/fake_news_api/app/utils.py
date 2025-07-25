import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download("stopwords")
stop_words = set(stopwords.words("portuguese")) # Essa variável já contém as stopwords em português


def limpar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r"[^a-záéíóúãõâêîôûç ]", "", texto)
    palavras = texto.split()
    palavras = [p for p in palavras if p not in stop_words]
    return " ".join(palavras)


def criar_vetorizador(textos, max_features=3000):
    # Passe a variável 'stop_words' (que é um set de palavras) diretamente aqui
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words=list(stop_words)) #
    X = vectorizer.fit_transform(textos)
    return vectorizer, X
