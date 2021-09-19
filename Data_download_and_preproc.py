import GEOparse
import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join

###############################################################
# GSE130036 RNA-seq данные 28 HCM patients and 9 healthy donors (для test)
###############################################################

PATH = './GSE130036_RAW/'
files = [f for f in listdir(PATH) if isfile(join(PATH, f))]

seq_data = pd.DataFrame()
colnames = []

# В цикле пробегаемся по всем файлам и объединяем их в один фрейм

for file in files:
    # пропускаем системный файл
    if file == '.DS_Store':
        continue

    df = pd.read_csv('./GSE130036_RAW/' + file,
                     compression='gzip', header=0, sep='\t', quotechar='"', error_bad_lines=False)

    df.index = df['target_id']
    colnames.append(file.split('_')[1])  # чтобы потом присвоить имена файлов столбцам
    seq_data = pd.concat([seq_data, df['tpm']], axis=1, sort=False)

seq_data.columns = colnames
seq_data['ID'] = seq_data.index

# аннотация
annotation = pd.read_csv('./ENSG.ENST.ENSP.Symbol.hg19.bed.txt', sep='\t', header=None)

annotation = pd.DataFrame(annotation.iloc[:, [3, 5]])  # оставляем только ENST и Gene Symbol. Остальное нам не нужно
annotation.columns = ['Gene Symbol', 'ID']

seq_data_annot = seq_data.merge(annotation, left_on='ID', right_on='ID')  # мерджим дату с аннотацией по ENST

# небольшой препроцессинг
# remove ENST without Gene Symbol
seq_data_annot = seq_data_annot.dropna(subset=['Gene Symbol'])
# for each Gene Symbol average TPM over ENST
seq_data_annot = seq_data_annot.groupby('Gene Symbol').median()

# формируем y и сохраняем
rnaseq_y = np.array(list((map(lambda x: 'HCM' in x, seq_data_annot.columns)))) + 0

pd.DataFrame(rnaseq_y).to_csv('./test_rna_seq_target.csv')
seq_data_annot.to_csv('./test_rna_seq_data.csv')

###############################################################
# чиповые данные
# GSE36961 Illumina array 106 больных HCM и 39 контролей, Mayo Clinic (для train)
###############################################################

dataset_name = "GSE36961"

gse = GEOparse.get_GEO(dataset_name)
data_train = gse.pivot_samples('VALUE')

# вытаскиваем лейблы семплов

experiments = {}
for i, (idx, row) in enumerate(gse.phenotype_data.iterrows()):
    tmp = {}
    tmp["Experiment"] = idx
    tmp['platform_id'] = row['platform_id']
    tmp['status'] = 'HCM' if 'HCM' in row['source_name_ch1'] else 'CTRL'
    experiments[i] = tmp
experiments = pd.DataFrame(experiments).T

gsm_to_stat = dict(zip(experiments['Experiment'], experiments['status']))
data_train.columns = [gsm_to_stat[i] for i in data_train.columns]

# сохраняем в файл
data_train.to_csv('./train_GSE36961.csv')
y_train = np.array(list((map(lambda x: 'HCM' in x, data_train.columns)))) + 0
pd.DataFrame(y_train).to_csv('./train_GSE36961_target.csv')


###############################################################
# чиповые данные
# GSE1145 Affimetrix array 16 неадекватов + стенозники и прочее, (старый тест)
###############################################################

def annotation(data_test, platform):

    data_test['index'] = data_test.index

    # annotate with GPL
    data_test = data_test.reset_index().merge(gse.gpls[platform].table[["ID", "Gene Symbol"]],
                                              left_on='index', right_on="ID").set_index('index')
    del data_test["ID"]
    # remove probes without ENTREZ
    data_test = data_test.dropna(subset=['Gene Symbol'])
    # remove probes with more than one gene assigned
    data_test = data_test[~data_test['Gene Symbol'].str.contains("///")]
    # for each gene average LFC over probes
    data_test = data_test.groupby('Gene Symbol').median()
    return data_test


dataset_name = "GSE1145"

gse = GEOparse.get_GEO(dataset_name)
data_test = gse.pivot_samples('VALUE')
data_test.head()


# вытаскиваем лейблы семплов

def status_extr(row):
    # print(row['title'])
    # print(row['description'].split('Keywords = ')[1], row['platform_id'])
    if 'N' in row['title']:
        return 'CTRL'
    if 'hypertrophic cardiomyopathy' in row['description'].split('Keywords = ')[1]:
        return 'HCM'
    elif 'congestive cardiomyopathy' in row['description'].split('Keywords = ')[1]:
        return 'CTRL'
    else:
        return 9999


experiments = {}
for i, (idx, row) in enumerate(gse.phenotype_data.iterrows()):
    tmp = {}
    tmp['title'] = row['title'][3:]
    tmp["Experiment"] = idx
    tmp['platform_id'] = row['platform_id']
    tmp['description'] = row['description'].split('Keywords = ')[1]
    tmp['status'] = status_extr(row)
    experiments[i] = tmp
experiments = pd.DataFrame(experiments).T

tmp_exp = experiments[experiments['status'] != 9999]  # [experiments['platform_id'] == 'GPL570']

data_test = data_test.filter(
    tmp_exp['Experiment'])  # отбираем ту часть эксперимента, которая нам нужна, сделанная на GPL570.
# Другая часть - исследование экспрессии микроРНК
gsm_to_stat = dict(zip(tmp_exp['Experiment'], tmp_exp['status'] + '_' + tmp_exp['platform_id']))
data_test.columns = [gsm_to_stat[i] for i in data_test.columns]

# этот блок появился для расширения тестового набора стенозниками
# аннотируем отдельно данные которые были получена на чипе GPL570 и на GPL8300

# GPL570
mask_570 = np.array(list(map(lambda x: 'GPL570' in x, data_test.columns)))
data_570 = data_test.loc[:, mask_570]
data_570_annot = annotation(data_570, 'GPL570')

# GPL8300
data_8300 = data_test.loc[:, mask_570 == False]
data_8300_annot = annotation(data_8300, 'GPL8300')

# merge
data_8300_annot['index'] = data_8300_annot.index
data_570_annot['index'] = data_570_annot.index

data_8300_570 = data_8300_annot.reset_index().merge(data_570_annot,
                                                    left_on='index', right_on="index").set_index('index')

data_8300_570 = data_8300_570.iloc[:, 1:]  # выкидываем индекс
data_570_annot = data_570_annot.iloc[:, :-1]  # выкидываем индекс

# сохраняем в файлики
data_8300_570.to_csv('./test_GSE1145_extension.csv')
y_test = np.array(list((map(lambda x: 'HCM' in x, data_8300_570.columns)))) + 0
pd.DataFrame(y_test).to_csv('./test_GSE1145_extension_target.csv')

data_570_annot.to_csv('./test_GSE1145.csv')
y_test = np.array(list((map(lambda x: 'HCM' in x, data_570_annot.columns)))) + 0
pd.DataFrame(y_test).to_csv('./test_GSE1145_target.csv')
