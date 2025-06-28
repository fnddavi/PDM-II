# Armazena o histórico das análises
historico = []


def adicionar_entrada(texto, resultado):
    historico.append({"texto": texto, **resultado})


def obter_historico():
    return historico
