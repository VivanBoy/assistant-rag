# fonction d'analyse des sentiments bilingue (français/anglais)
from textblob import TextBlob
from textblob_fr import PatternTagger, PatternAnalyzer

def analyser_sentiment(texte, langue):
    """
    Analyse le sentiment d'un texte en français ou en anglais.
    
    Args:
        texte (str): Le texte à analyser
        langue (str): 'fr' ou 'en'
        
    Returns:
        float: Score de polarité entre -1 (négatif) et 1 (positif)
    """
    if langue == 'fr':
        blob = TextBlob(texte, pos_tagger=PatternTagger(), analyzer=PatternAnalyzer())
        return blob.sentiment[0]  # PatternAnalyzer retourne un tuple (polarity, subjectivity)
    else:
        blob = TextBlob(texte)
        return blob.sentiment.polarity