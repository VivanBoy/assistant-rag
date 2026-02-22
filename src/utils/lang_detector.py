from langdetect import detect, LangDetectException

def detecter_langue(texte):
    try:
        return detect(texte)
    except LangDetectException:
        return "inconnu"