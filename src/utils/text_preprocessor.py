import re
import spacy
import pandas as pd

# chargement des modèles de langues
nlp_fr = spacy.load("fr_core_news_sm")
nlp_en = spacy.load("en_core_web_sm")

ponctuation = {'.', ',', '!', '?', ';', ':', '...', '(', ')', '[', ']', '{', '}'}


def nettoyer_texte(texte):
    texte = re.sub(r'\n', ' ', str(texte))
    texte = re.sub(r'\s+', ' ', texte)
    return texte.strip()

def tokenizer_lemmatizer(texte, langue):
    if langue == 'fr':
        doc = nlp_fr(texte)
    else:
        doc = nlp_en(texte)
    
    tokens = [token.text for token in doc]
    lemmes = [token.lemma_ for token in doc]
    
    return tokens, lemmes

def appliquer_traitement_ligne(row):
    tokens, lemmes = tokenizer_lemmatizer(row['avis_nettoye'], row['langue'])
    return pd.Series([tokens, lemmes])

def filtrer_stopwords(tokens, lemmes, langue):
    """
    Filtre les tokens et lemmes pour retirer les stop words.
    
    Args:
        tokens (list): Liste des tokens
        lemmes (list): Liste des lemmes correspondants
        langue (str): 'fr' ou 'en'
    
    Returns:
        tuple: (tokens_filtres, lemmes_filtres)
    """
    if langue == 'fr':
        stop_words = nlp_fr.Defaults.stop_words
    else:
        stop_words = nlp_en.Defaults.stop_words
    
    tokens_filtres = []
    lemmes_filtres = []
    
    for token, lemme in zip(tokens, lemmes):
        if token.lower() not in stop_words:
            tokens_filtres.append(token)
            lemmes_filtres.append(lemme)
    
    return tokens_filtres, lemmes_filtres

def appliquer_filtrage(row):
    tokens_filtres, lemmes_filtres = filtrer_stopwords(
        row['tokens'], row['lemmes'], row['langue']
    )
    return pd.Series([tokens_filtres, lemmes_filtres])

def filtrer_ponctuation(tokens, lemmes):
    tokens_filtres = [t for t in tokens if t not in ponctuation]
    lemmes_filtres = [l for l in lemmes if l not in ponctuation]
    return tokens_filtres, lemmes_filtres