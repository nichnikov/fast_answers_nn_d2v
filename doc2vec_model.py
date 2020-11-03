import os, pickle, logging
from collections import deque
from texts_processors import TokenizerApply, SimpleTokenizer
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from utility import splitArray, Loader
from clickhouse_connect import questions_from_clickhouse
from random import shuffle
from lingv_functions import texts_lemmatize, ngram_apply


def create_doc2vec_model(**kwargs):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    # внимание! очень важно, как разбиты токены в split_txt
    # вид должен быть следующий: [['word1', 'word2', ...], ['word1', 'word5', ...], ... ]
    tagged_data = [TaggedDocument(doc, [i]) for i, doc in enumerate(kwargs["split_txt"])]
    print("tagged_data made, example:", tagged_data[0])
    model = Doc2Vec(tagged_data, vector_size=300, window=3, min_count=1, workers=8)
    model.build_vocab(documents=tagged_data, update=True)
    model.train(tagged_data, total_examples=model.corpus_count, epochs=25)
    model.save(kwargs["model_rout"])
    print("model made and saved")
    return 0


def update_doc2vec_model(**kwargs):
    # внимание! очень важно, как разбиты токены в tokens_texts
    # вид должен быть следующий: [['word1', 'word2', ...], ['word1', 'word5', ...], ... ]
    # split_txts дополнительно разбивается на указанное количество кусков, из которых образуется очередь (если текстов дял обучения модели слишком много)
    # logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    model = Doc2Vec.load(kwargs["initial_model_rout"])
    model.save(kwargs["updated_model_rout"])
    print("model loaded and saved in new path")

    # разбиваем входящие тексты из токенов на куски для формирования очереди (указываем размер)
    split_txts = splitArray(kwargs["tokens_texts"], kwargs["chunk_size"])

    q = deque(split_txts)
    # счетчик
    k = 1
    n = len(split_txts)
    print("deque length is:", n)
    while q:
        print(k, '/', n)

        # дообучение модели Doc2Vec по оставшимся текстам:
        documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(q.pop())]
        model = Doc2Vec.load(kwargs["updated_model_rout"])
        print("loop:", k, "model loaded")
        model.build_vocab(documents=documents, update=True)
        print("loop:", k, "build_vocab done")
        try:
            model.train(documents, total_examples=model.corpus_count, epochs=25)
            print(documents[:10])
        except Exception:
            print("ModeLearningErr")
        model.save(kwargs["updated_model_rout"])
        k += 1
    return 0



def doc2vec_model_maker(**kwargs):
    key_words = ['%%']
    questions = []
    for word in key_words:
        res = questions_from_clickhouse(clickhose_host="srv02.ml.dev.msk3.sl.amedia.tech", user='nichnikov',
                                        password='CGMKRW5rNHzZAWmvyNd6C8EfR3jZDwbV', date_in='2020-04-01',
                                        date_out='2020-08-31', limit=1000000, pubids_tuple=kwargs["pubids_tuple"],
                                        key_word=word)

        qs, dts = zip(*res)
        questions = questions + list(qs)

    print(len(questions))
    shuffle(questions)
    questions = questions # [:1000]
    # data_for_models = list(questions[:1000000])

    # модель для токенизатора (используем простую модель, которая предполагается в наличие у каждой системы):
    if "simple_model_path" in kwargs:
        with open(kwargs["simple_model_path"], "rb") as f:
            model_for_tokenizer = pickle.load(f)

        tokenizer = SimpleTokenizer(Loader(model_for_tokenizer))
        tz_txs = tokenizer.texts_processing(questions)

    # надо сделать отдельную функцию для лемматизации
    if "lingvo_data" in kwargs:
        asc_dsc_syn = []
        asc_dsc_ngrm = []
        if "synonyms" in kwargs["lingvo_data"]:
            for fn in kwargs["lingvo_data"]["synonyms"]:
                temp_syn_df = pd.read_csv(fn)
                syn_asc_temp = [" " + tx + " " for tx in texts_lemmatize(temp_syn_df['words'])]
                syn_dsc_temp = [" " + tx + " " for tx in texts_lemmatize(temp_syn_df['initial_forms'])]
                asc_dsc_syn += list(zip(syn_asc_temp, syn_dsc_temp))

        if "ngrams" in kwargs["lingvo_data"]:
            asc_dsc_ngrm = []
            for fn in kwargs["lingvo_data"]["ngrams"]:
                temp_ngrms_df = pd.read_csv(fn)
                temp_ngrms = [(' '.join([w1, w2]), bgr) for w1, w2, bgr in
                              zip(temp_ngrms_df['w1'], temp_ngrms_df['w2'], temp_ngrms_df['bigrams'])]
                asc_dsc_ngrm += temp_ngrms


        asc_dsc_list = asc_dsc_syn + asc_dsc_ngrm
        tz_txs = ngram_apply(asc_dsc_list, texts_lemmatize(questions))
        tz_txs_split = [tx.split() for tx in tz_txs if tx.split() != []]

    # соберем LSI модель на основании коллекции из 100 тысяч вопросов:
    model_parameters = {"split_txt": tz_txs_split, "model_rout": kwargs["doc2vec_model_path"]}
    create_doc2vec_model(**model_parameters)

    return 0


if __name__ == "__main__":
    data_path = r'./data'
    models_path = r'./models'
    lingv_path = r'./lingvo'

    # создание модели
    # использование измененной лингвистики:
    bss_d2v_parameters = {
        "lingvo_data": {"synonyms": [os.path.join(lingv_path, "bss_synonyms01.csv"),
                                     os.path.join(lingv_path, "bss_synonyms02.csv"),
                                     os.path.join(lingv_path, "bss_synonyms03.csv")],
                        "ngrams": [os.path.join(lingv_path, "bss_ngrams.csv")]},
        "pubids_tuple": (6, 8, 9),
        "doc2vec_model_path": os.path.join(models_path, "bss_doc2vec_model_no_stpwds")}

    # использоваоние лингвистики быстрых ответов:
    bss_d2v_parameters_bo = {
        "simple_model_path": os.path.join(models_path, "bss_model_simple.pickle"),
                        "ngrams": [os.path.join(lingv_path, "bss_ngrams.csv")],
        "pubids_tuple": (6, 8, 9),
        "doc2vec_model_path": os.path.join(models_path, "bss_doc2vec_model")}

    # создание модели:
    # doc2vec_model_maker(**bss_d2v_parameters)

    # тестирование модели
    model = Doc2Vec.load(os.path.join(models_path, 'bss_doc2vec_model_no_stpwds'))
    d2v_voc = [w for w in model.wv.vocab]
    for w in d2v_voc:
        print(w)

    test = ['срок', 'камеральныйпроверка', 'декларация', 'налогприбыль']  # [['срок', 'камеральныйпроверка', '3ндфл']]
    print(d2v_voc[:10])
    test_data = ['камеральныйпроверка']
    v = model.infer_vector(test_data)
    print(v)

    test_data = ['камеральныйпроверка']
    print(model.most_similar(positive=test_data, topn=10))
    print(len(model.wv.vocab))
    """
    # тест на словаре из слов вопросов, на которых предполагается обучаться:
    """
    """
    test_df = pd.read_csv(os.path.join(data_path, "lemm_dataset_01.csv"))
    print(test_df)

    test_voc = []
    for x in test_df["text"]:
        test_voc = test_voc + x.split()

    test_voc_unique = set(test_voc)
    print(test_voc_unique)
    print(len(test_voc_unique))

    out_words = [w for w in test_voc_unique if w not in d2v_voc]
    print(out_words)
    print(len(out_words))"""