from __future__ import absolute_import
from __future__ import print_function

# нужно добавить проверку на наличие слов входящих запросов в словаре (например, в словах из правил)
import os, pickle
from keras.models import load_model
import keras.losses
from keras_functions import contrastive_loss, siamese_data_prepare, accuracy
from gensim.models.doc2vec import Doc2Vec
from texts_processors import TokenizerApply
import pandas as pd
import numpy as np
from lingv_functions import lingvo_application
import keras.losses
from keras.utils.generic_utils import get_custom_objects


def binarizator(x, coeff):
    if x > coeff:
        return 1
    else:
        return 0


def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))


def intersec_score(lst1, lst2):
    inters = intersection(lst1, lst2)
    score = len(inters) / len(set(lst1))
    return score


def sliceArray(src: [], length: int = 1, stride: int = 1):
    return [src[i:i + length] for i in range(0, len(src), stride)]


def tester(**kwargs):
    scores = kwargs["model"].predict([kwargs["et_vecs"].reshape(kwargs["et_vecs"].shape[0], 300, 1),
                                      kwargs["tst_vecs"].reshape(kwargs["tst_vecs"].shape[0], 300, 1)])
    texts_df = pd.DataFrame(kwargs["ets_tsts_txs"], columns=["etalon", "etalon_lm", "question_tx", "question_lm"])
    scores_df = pd.DataFrame([score for score in scores], columns=['score'])
    result_df = pd.concat([texts_df, scores_df], axis=1)
    result_df.to_csv(os.path.join(results_path, kwargs["result_file_name"] + "_" + str(kwargs["num"]) + ".csv"))
    result_df[result_df['score'] < 0.2].to_csv(
        os.path.join(results_path, result_file_name + "_" + str(kwargs["num"]) + "_02.csv"))
    return 0


data_path = r"./data"
models_path = r"./models"
lingv_path = r"./lingvo"
results_path = r"./results"
nn_model_file = r"bss_d2v_nn_siamese_gr55_1021.h5"
doc2vec_file = r'bss_doc2vec_model'
etalons_file_name = "bss_test_etalons_gr55.csv"
test_file_name = "bss_test_one_day15273.csv"
result_file_name = "test_results_gr55_one_day"


#  как бороться с отсутствием пользовательской функции без грубого вмешательства в  keras (которое к тому же не получилось)
# https://github.com/keras-team/keras/issues/5916
keras.losses.contrastive_loss = contrastive_loss
# вот это рабочий вариант:
get_custom_objects().update({"contrastive_loss": contrastive_loss})

# load models:
d2v_model = Doc2Vec.load(os.path.join(models_path, doc2vec_file))
print("d2v_model load Done")

nn_model = load_model(os.path.join(models_path, nn_model_file))
print("lstm_model load Done")

test_df = pd.read_csv(os.path.join(data_path, test_file_name))
etalons_df = pd.read_csv(os.path.join(data_path, etalons_file_name))

bss_lingvo_parameters = {
    "synonyms": [os.path.join(lingv_path, "bss_synonyms01.csv"),
                 os.path.join(lingv_path, "bss_synonyms02.csv"),
                 os.path.join(lingv_path, "bss_synonyms03.csv")],
    "ngrams": [os.path.join(lingv_path, "bss_ngrams.csv")]}

lemm_test_texts = lingvo_application(test_df['text'], **bss_lingvo_parameters)
lemm_etalons_texts = lingvo_application(etalons_df['text'], **bss_lingvo_parameters)

et_vecs = [d2v_model.infer_vector(et_lm) for et_lm in lemm_etalons_texts]
tst_vecs = [d2v_model.infer_vector(et_lm) for et_lm in lemm_test_texts]
ets_tsts_vects = [(et_v, ts_v) for et_v in et_vecs for ts_v in tst_vecs]
ets_tsts_texts = [(et_tx, et_lm, ts_tx, ts_lm) for et_tx, et_lm in
                  zip(etalons_df['text'], [" ".join(tx) for tx in lemm_etalons_texts])
                  for ts_tx, ts_lm in zip(test_df['text'], [" ".join(tx) for tx in lemm_test_texts])]

del d2v_model, test_df, etalons_df

et_vs, ts_vs = zip(*ets_tsts_vects)
et_vecs_arr = np.array(et_vs)
tst_vecs_arr = np.array(ts_vs)

print("vectors done")
print(et_vecs_arr.shape)
print(tst_vecs_arr.shape)

exmpls_quant = 1000000
et_vecs_arr_spl = sliceArray(et_vecs_arr, length=exmpls_quant, stride=exmpls_quant)
tst_vecs_arr_spl = sliceArray(tst_vecs_arr, length=exmpls_quant, stride=exmpls_quant)
ets_tsts_texts_spl = sliceArray(ets_tsts_texts, length=exmpls_quant, stride=exmpls_quant)

del et_vecs_arr, tst_vecs_arr, ets_tsts_texts

num = 1
for et_vecs, tst_vecs, ets_tsts_txs in zip(et_vecs_arr_spl, tst_vecs_arr_spl, ets_tsts_texts_spl):
    parameters = {"model": nn_model, "et_vecs": et_vecs, "tst_vecs": tst_vecs, "ets_tsts_txs": ets_tsts_txs,
                  "result_file_name": result_file_name, "num": num}
    print(num)
    tester(**parameters)
    num += 1
