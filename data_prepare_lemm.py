import os, pickle
import pandas as pd
import random
from texts_processors import TokenizerApply
from utility import Loader
from lingv_functions import lingvo_application

data_path = r'./data'
models_path = r'./models'
lingv_path = r'./lingvo'

dataset_file = "bss_dataset_gr55_ex2132.csv"
lem_dataset_file = "bss_lemm_dataset.csv"
paraphrase_file = "bss_paraphrase_gr55.csv"
no_paraphrase_file = "bss_no_paraphrase_gr55.csv"
dataset_train = "dataset_train_gr55.csv"
dataset_validate = "dataset_validate_gr55.csv"
lem_add_datasets = ["bss_add_dataset_1023.csv"]


data_df = pd.read_csv(os.path.join(data_path, dataset_file))
print(data_df)
bss_lingvo_parameters = {
        "synonyms": [os.path.join(lingv_path, "bss_synonyms01.csv"),
                     os.path.join(lingv_path, "bss_synonyms02.csv"),
                     os.path.join(lingv_path, "bss_synonyms03.csv")],
        "ngrams": [os.path.join(lingv_path, "bss_ngrams.csv")]}

lemm_txts_l = lingvo_application(data_df["text"], **bss_lingvo_parameters)

lemm_txts_df = pd.DataFrame(list(zip([" ".join(x) for x in lemm_txts_l], data_df['group'])), columns=['text', 'group'])
lemm_txts_df.to_csv(os.path.join(data_path, lem_dataset_file), index=False)

df = pd.read_csv(os.path.join(data_path, lem_dataset_file))
print(df)

# герерация пар семантически одинаковых вопросов
lbs = set(df['group'])
print(lbs)
results_tuples = []
for lb in lbs:
    work_list = list(df['text'][df['group'] == lb])
    for tx1 in work_list:
        for tx2 in work_list:
            results_tuples.append((tx1, tx2, 1))


# из этих групп сформировать обучающуюся выборку
print(lbs)
print(len(lbs))

results_tuples_diff_lbs = []
for lb in lbs:
    txts_list_1 = list(df['text'][df['group'] == lb])
    txts_list_2 = list(df['text'][df['group'] != lb])
    for tx1 in txts_list_1:
        for tx2 in txts_list_2:
            results_tuples_diff_lbs.append((tx1, tx2, 0))

result_pair_df = pd.DataFrame(results_tuples, columns=["question1", "question2", "is_duplicate"])
result_no_pairs_df = pd.DataFrame(results_tuples_diff_lbs, columns=["question1", "question2", "is_duplicate"])

print(result_pair_df)
print(result_no_pairs_df)

result_pair_df.to_csv(os.path.join(data_path, paraphrase_file))
result_no_pairs_df.to_csv(os.path.join(data_path, no_paraphrase_file))

# подготовка базового (канонического датасета, зависящего от лингвистики)
# выберем не идентичные пары (выберем случайным образом в 3 раза больше пар, чем идентичных):
no_pairs_df = result_no_pairs_df.sample(result_pair_df.shape[0]*3)
pairs_df = pd.concat([result_pair_df, no_pairs_df], axis=0, ignore_index=True)
pairs_df = pairs_df.sample(frac=1)
pairs_df.to_csv(os.path.join(data_path, lem_dataset_file), index=False)

# добавим дополнительные данные (если есть)
df = pd.read_csv(os.path.join(data_path, lem_dataset_file))
for lem_add_dataset_file in lem_add_datasets:
    df_add = pd.read_csv(os.path.join(data_path, lem_add_dataset_file))
    df = pd.concat([df, df_add])

# удалим дубликаты:
df.drop_duplicates(inplace=True)
train_num = 9 * df.shape[0] // 10
df = df.sample(frac=1)
df[:train_num].to_csv(os.path.join(data_path, dataset_train))
df[train_num:].to_csv(os.path.join(data_path, dataset_validate))

print(df)
print(train_num)
