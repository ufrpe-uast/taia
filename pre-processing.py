# -*- coding: utf-8 -*-

import csv
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

def opinioes(arquivo):
    opinioes = []
    with open(arquivo) as corpus:
        ops = csv.reader(corpus)
        for l in ops:
            opinioes.append(l[1])

    return opinioes

def tokenizar(arquivo):
    opinioes_tokenizadas = []
    with open(arquivo) as corpus:
        ops = csv.reader(corpus)
        for l in ops:
            opiniao = l[1]
            opinioes_tokenizadas.append(word_tokenize(opiniao))

    return opinioes_tokenizadas

# tokenizado = tokenizar('corpus-classificado.csv')

def remover_stopwords(arquivo):
    removidos = []
    pt_br_stop_words = set(stopwords.words('portuguese'))
    pt_br_stop_words.update('@', '#','...','.',':','!','?',',')
    # Adicionar tbm os nomes dos candidatos

    tokens = tokenizar(arquivo)
    for t in tokens:
        filtrado = []
        for p in t:
            if p.lower() == 'não' or p not in pt_br_stop_words:
                filtrado.append(p)

        removidos.append(filtrado)

    return removidos

def stemming(arquivo):
    stemizados = []
    stemmer = RSLPStemmer()
    tokens = tokenizar(arquivo)
    for t in tokens:
        s = []
        for p in t:
            s.append(stemmer.stem(p))

        stemizados.append(s)

    return stemizados

def tfidf(arquivo):
    vec = TfidfVectorizer(min_df=1)
    x = vec.fit_transform(opinioes(arquivo))
    idf = vec.idf_
    return dict(zip(vec.get_feature_names(), idf))

def tokens_diferentes(tokens):
    ps = []
    for t in tokens:
        for p in t:
            if p not in ps:
                ps.append(p)

    return len(ps)

def media_tokens_por_opiniao(tokens):
    opinioes = len(tokens)
    ts = []
    for t in tokens:
        for p in t:
            ts.append(p)

    return len(ts)/opinioes

# Total
# Antes pre-processamento
print(tokens_diferentes(tokenizar('corpus-classificado.csv')))

# Pos remocao stopwords
print(tokens_diferentes(remover_stopwords('corpus-classificado.csv')))

# Pos stemming
print(tokens_diferentes(stemming('corpus-classificado.csv')))

# Media
# Antes pre-processamento
print(media_tokens_por_opiniao(tokenizar('corpus-classificado.csv')))

# Pos remocao stopwords
print(media_tokens_por_opiniao(remover_stopwords('corpus-classificado.csv')))

# Pos stemming
print(media_tokens_por_opiniao(stemming('corpus-classificado.csv')))
