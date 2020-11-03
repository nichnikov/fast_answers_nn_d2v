# здесь будут функции, которые нужны для других объектов
import os, pickle, difflib, logging
from abc import ABC, abstractmethod
from itertools import groupby


# функция для создания модели
def models_assembly(model_type="simple_rules", **kwargs):
    # загрузка и проверка правил:

    if "tokenizer" in kwargs:
        tokenizer = kwargs["tokenizer"]
    else:
        tokenizer = "SimpleTokenizer"

    if 'lingv_rules' in kwargs:

        if 'rules' in kwargs['lingv_rules']:
            rules = kwargs['lingv_rules']['rules']
        else:
            rules = []

        if 'words' in kwargs['lingv_rules']:
            texts = kwargs['lingv_rules']['words']
        else:
            texts = []

        if 'tags' in kwargs['lingv_rules']:
            tags = kwargs['lingv_rules']['tags']
        else:
            tags = []

        if 'coeff' in kwargs['lingv_rules']:
            coeff = kwargs['lingv_rules']['coeff']
        else:
            coeff = []
    else:
        rules, texts, tags, coeff, tknz_texts = [], [], [], [], []

    assert len(rules) == len(texts), "количество правил не соответсвует количеству текстов"
    assert len(rules) == len(tags), "количество правил не соответсвует количеству тегов"
    assert len(rules) == len(coeff), "количество правил не соответсвует количеству коэффициентов"

    # загрузка прочих параметров модели:
    """добавить модуль Lingvo"""

    if "lingvo" in kwargs:
        lingvo = kwargs["lingvo"]
    else:
        lingvo = []

    if "classificator_algorithms" in kwargs:
        classificator_algorithms = kwargs["classificator_algorithms"]
    else:
        classificator_algorithms = {}

    if "texts_algorithms" in kwargs:
        texts_algorithms = kwargs["texts_algorithms"]
    else:
        texts_algorithms = {}

    if "is_etalons_lemmatize" in kwargs:
        is_etalons_lemmatize = kwargs["is_etalons_lemmatize"]
    else:
        is_etalons_lemmatize = False

    if "is_lingvo_lemmatize" in kwargs:
        is_lingvo_lemmatize = kwargs["is_lingvo_lemmatize"]
    else:
        is_lingvo_lemmatize = False

    model_dict = {"model_type": model_type,
                  # лемматизированные или нет тексты в эталонах для классификации:
                  # (предпочтительно создавать модели сразу с лемматизированными эталонами)
                  # значения: True или False
                  "is_etalons_lemmatize": is_etalons_lemmatize,
                  # лемматизированные или нет тексты в словарях лингвистических моделей:
                  # (предпочтительно создавать модели сразу с лемматизированными словарями)
                  # значения: True или False
                  "is_lingvo_lemmatize": is_lingvo_lemmatize,
                  "etalons": {
                      "rules": rules,
                      "texts": texts,
                      "tags": tags,
                      "coeff": coeff},
                  "lingvo": lingvo,
                  "classificator_algorithms": classificator_algorithms,
                  "texts_algorithms": texts_algorithms,
                  "tokenizer": tokenizer}
    return model_dict


# загрузчик из бинарника:
class Loader():
    def __init__(self, model_dict):
        # возвращает ключ модели (имя модели) по ключу остальные объекты "понимают" что за функции им нужно запускать
        self.model_type = model_dict["model_type"]
        assert self.model_type in ["siamese_lstm_d2v", "simple_rules", "lsi"], \
            "тип модели не соответствует ожиданию класса Loader"

        # отвечает на вопрос, нужно ли лемматизировать словари (или словари в модели уже лемматизированные)
        self.is_lingvo_lemmatize = model_dict["is_lingvo_lemmatize"]
        # отвечает на вопрос, нужно ли лемматизировать эталоны (или эталоны в модели уже лемматизированные)
        self.is_etalons_lemmatize = model_dict["is_etalons_lemmatize"]
        # возвращает модели для правил
        self.classificator_algorithms = model_dict["classificator_algorithms"]
        # возвращает модели для обработки текстов (например, Word2Vec - модели векторизации)
        self.texts_algorithms = model_dict["texts_algorithms"]
        # возвращает словари для обработки текста
        self.dictionaries = model_dict["lingvo"]
        # возвращает тип токенезации входящего текста
        self.tokenizer_type = model_dict["tokenizer"]
        assert self.tokenizer_type in ["SimpleTokenizer", "Doc2VecTokenizer", "LsiTokenizer"], \
            "tokenizer_type не соответствует ожиданиям класса Loader"

        # возвращает правила
        self.application_field = model_dict["etalons"]
        for nm in self.application_field:
            assert nm in ["rules", "texts", "coeff", "tags"], \
                "имена словаря etalons не соответствуют ожиданиям класса Loader"


class AbstractRules(ABC):
    def __init__(self):
        # перменная, описывающая, какие модели входят в класс
        self.model_types = []

    # должен возвращать структуру типа: [(num, [(tag, True), ...]), ...]
    # num - номер текста
    # tag - номер текста
    # True / False - результат для данного тега и данного текста
    @abstractmethod
    def rules_apply(self, text: []):
        pass


def models_chain(texts, rules: []):
    results = []
    for Class_with_model in rules:
        cls_results = Class_with_model.rules_apply(texts)
        for tx_result in cls_results:
            results.append(tx_result)
    # grouping results with the same texts
    results_grouped = [(x, [z[1] for z in y]) for x, y in
                       groupby(sorted(results, key=lambda x: x[0]), key=lambda x: x[0])]

    tags_result = []
    # если "последовательность правил" состоит только из одного правила:
    if len(rules) == 1:
        for tx_num, txt_tags_group in results_grouped:
            if txt_tags_group != [[]]:
                tags_result.append(tuple((tx_num, txt_tags_group[0][0])))
            else:
                tags_result.append(tuple((tx_num, None)))
    # если "последовательность правил" состоит больше, чем из одного правила:
    else:
        for tx_num, txt_tags_group in results_grouped:
            tags_result.append(tuple((tx_num, decision_choice(txt_tags_group))))
    return tags_result


# Основное время тратится на загрузку лемматизатора и на лемматизацию эталонов
# classes - классы, которые используются в цепочке
# функция, применяющая набор моделей (цепочку моделей) к входящему тексту
# допущение - модели должны содержать одинаковые эталоны с одинаковыми тегами
# models :[] -> [loader_obj, ...]
# true_tags = classes_models[0][1].application_field["tags"]
# приоритет моделей соответствует последовательности загрузки моделей в класс


class ModelsChain(AbstractRules):
    def __init__(self, model_list):
        self.model_list = model_list
        # цикл нужен для "инициализации моделей класса, если его не делать
        # при первом запуске методов класса, время выполнения включает в себя инициализации моделей
        # в районе 1 сек
        for model_dict in self.model_list:
            if "function_type" in model_dict:
                model_dict["classificator"].rules_apply(["test"], function_type=model_dict["function_type"])
            else:
                model_dict["classificator"].rules_apply(["test"])

    def rules_apply(self, texts):  # выбор классов для полученных моделей:
        results = []
        for model_dict in self.model_list:
            if "function_type" in model_dict:
                cls_results = model_dict["classificator"].rules_apply(texts, function_type=model_dict["function_type"])
            else:
                cls_results = model_dict["classificator"].rules_apply(texts)
            for tx_result in cls_results:
                results.append(tx_result)
        # grouping results with the same texts
        results_grouped = [(x, [z[1] for z in y]) for x, y in
                           groupby(sorted(results, key=lambda x: x[0]), key=lambda x: x[0])]

        tags_result = []
        # если "последовательность правил" состоит только из одного правила:
        if len(self.model_list) == 1:
            for tx_num, txt_tags_group in results_grouped:
                if txt_tags_group != [[]]:
                    tags_result.append(tuple((tx_num, txt_tags_group[0][0])))
                    # tags_result.append(tuple((tx_num, txt_tags_group[0])))
                else:
                    tags_result.append(tuple((tx_num, None)))
        # если "последовательность правил" состоит больше, чем из одного правила:
        else:
            for tx_num, txt_tags_group in results_grouped:
                tags_result.append(tuple((tx_num, decision_choice(txt_tags_group))))
        return tags_result


# совсем утилитарная функция для класса ModelsChain
def decision_choice(ls: []):
    for i in ls[0]:
        decision = False
        for l in ls[1:]:
            if i in l:
                decision = True
            else:
                decision = False
        if decision:
            return i
    return None


""" Утилитарные функции над массивами """
""" Оставляет только базовые элементы сложной (иерархической) структуры """


def flatten_list(iterable):
    for elem in iterable:
        if not isinstance(elem, list):
            yield elem
        else:
            for x in flatten_list(elem):
                yield x


def flatten_tuple(iterable):
    for elem in iterable:
        if not isinstance(elem, tuple):
            yield elem
        else:
            for x in flatten_tuple(elem):
                yield x


"""Нарезает массив окном размера len c шагом stride"""
def sliceArray(src: [], length: int = 1, stride: int = 1):
    return [src[i:i + length] for i in range(0, len(src), stride) if len(src[i:i + length]) == length]


'''Нарезает массив окном размера len с шагом в размер окна'''
def splitArray(src: [], length: int):
    return sliceArray(src, length, length)


# Преобразует массив токенов в мешок (каждый токен представлен кортежем
# -- (токен, сколько раз встречается в массиве))
def arr2bag(src: []):
    return [(x, src.count(x)) for x in set(src)]


"""Возвращает массив токенов src за исключением rem"""


def removeTokens(src: [], rem: []):
    return [t for t in src if t not in rem]


"""Заменяет в массиве src множество токенов аскриптора asc дексрипторами токена (синонимия)"""


def replaceAscriptor(src: [], asc: [], desc: []):
    src_repl = []
    length = len(asc)
    src_ = [src[i:i + length] for i in range(0, len(src), 1)]
    i = 0
    while i < len(src_):
        if src_[i] == asc:
            src_repl = src_repl + desc
            i += length
        else:
            src_repl.append(src_[i][0])
            i += 1
    return src_repl


""" Прочие функции """
""" функция оценивающая похожесть строк (возвращает оценку похожести) """


def strings_similarities(str1: str, str2: str):
    return difflib.SequenceMatcher(None, str1, str2).ratio()


# сравнение двух списков (возвращает токены, принадлежащие обоим спискам)
def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))


# функция для одного текста (модели могут отрабатывать на пакете (списке) текстов)
"""
def search(pubs_models, pbid, q: str):
    for pbs, model in pubs_models:
        if pbid in pbs:
            true_decisions = model.rules_apply([q])
            if true_decisions and true_decisions[0][1] is not None:
                return {"mod": 85, "id": true_decisions[0][1]}
    return {"mod": 85, "id": 9999}

"""
if __name__ == "__main__":
    import time

    data_rout = r'./data'
    models_rout = r'./models'

    from classificators import LsiClassifier, SimpleRules, SimpleRulesRang

    # with open(os.path.join(models_rout, "fast_answrs", "kosgu_intersec_share_model.pickle"), "br") as f:
    #    model0 = pickle.load(f)

    # with open(os.path.join(models_rout, "fast_answrs", "kosgu_lsi_model.pickle"), "br") as f:
    #    model1 = pickle.load(f)

    with open(os.path.join(models_rout, "fast_answrs", "zavuch_include_and_model.pickle"), "br") as f:
        model = pickle.load(f)

    smplm = ModelsChain([(SimpleRules, model)])
    # lsim = ModelsChain([(LsiClassifier, model1)])
    # lsim = LsiClassifier(Loader(model1))

    txs = ["положение о педсовете"]

    print(smplm.rules_apply(txs))
