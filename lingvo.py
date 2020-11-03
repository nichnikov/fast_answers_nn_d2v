import time, re
from pymystem3 import Mystem


class TextH():
    def __init__(self):
        self.m = Mystem()

    # функция лемматизации одного текста
    def text_lemmatize(self, text: str):
        try:
            lemm_txt = self.m.lemmatize(text)
            lemm_txt = [w for w in lemm_txt if w not in [' ', '\n', ' \n']]
            return lemm_txt
        except:
            return ['']


txtlm = TextH()

# функция, проводящая предобработку текста
def text_hangling(text: str):
    try:
        txt = re.sub('[^a-zа-я\d]', ' ', text.lower())
        txt = re.sub('\s+', ' ', txt)
        # сюда можно будет вложить самую разную обработку, в том числе и вариационную
        return txt
    except:
        return ""


# функция лемматизации списка текстов текста
def texts_lemmatize(texts_list):
    return [' '.join(txtlm.text_lemmatize(text_hangling(tx))) for tx in texts_list]


def ngram_apply(asc_dsc_list, texts_list):
    texts = '\n'.join(texts_list)
    for asc, dsc in asc_dsc_list:
        texts = re.sub(asc, dsc, texts)
    return texts.split('\n')