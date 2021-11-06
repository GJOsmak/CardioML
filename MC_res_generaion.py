import pandas as pd
import numpy as np
from sklearn.manifold import MDS
from sklearn import metrics, linear_model
from random import choices
from itertools import compress
from sklearn.preprocessing import StandardScaler, Normalizer
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy import stats
from multiprocessing import Manager, Pool, cpu_count

NUM_ITERATIONS = 200
NUM_PER_FILE = 5
NUM_PROCESSES = cpu_count()

class Data(object): 
    def __init__(self):
        pass
    
Data.X = Data()
Data.X.train = pd.read_csv('./Data/Sets/train_GSE36961.csv', index_col=0)
Data.y = Data()
Data.y.train = list(pd.read_csv('./Data/Sets/train_GSE36961_target.csv').iloc[:,1])

X_shared = pd.DataFrame(Data.X.train.transpose())
y_shared = Data.y.train

def MC_model(X, y, C=0.03):
#        X = X_shared, 
#        y = y_shared

        mask = np.array(y) == 0

        # размер выборки бутстрепа, берем размер минимальной группы HCM или CTRL
        k_len = min(len(mask) - sum(mask), sum(mask))  

        CTRL_idx = list(compress(range(0, len(mask)), mask))
        HCM_idx = list(compress(range(0, len(mask)), mask == False))

        train_idx = choices(HCM_idx, k=k_len) + choices(CTRL_idx, k=k_len) # бутстрепим номера строк из группы больных и здоровых
        test_idx = list(set([i for i in range(len(mask))]) - set(train_idx))

        # объединяем это всё дело обратно

        _X_train = pd.DataFrame(X).iloc[train_idx, :]
        _y_train = np.array(y)[train_idx]

        # обучаем лог.рег. с ранее отобранным коэффициентом регуляризации

        linear_regressor = linear_model.LogisticRegression(penalty='l1', C=C, solver='liblinear',
                                                           random_state=42)
        linear_regressor.fit(_X_train, _y_train)

        # тестим

        roc_auc = metrics.roc_auc_score(y_score=linear_regressor.predict(pd.DataFrame(X).iloc[test_idx, :]), y_true=np.array(y)[test_idx])

        # далее отбираем фичи из моделей, которые хоть как-то работают (roc_auc > 0.7)

        if roc_auc > 0.7:
            # отбираем смысловые фичи
            mask = linear_regressor.coef_ != 0
            mask = np.append(mask[0], roc_auc)
        
        return mask

def run_iteration(seed):
    np.random.seed(seed)
    val = MC_model(ns.X, ns.y)
    return val

mgr = Manager()
ns = mgr.Namespace()
ns.X = X_shared
ns.y = y_shared

for i in range(NUM_ITERATIONS // NUM_PER_FILE):
    print("Iteration:", i, "/", NUM_ITERATIONS // NUM_PER_FILE)
    
    with Pool(NUM_PROCESSES) as p:
        res = p.map(run_iteration, [seed for seed in range(i*NUM_PER_FILE, (i+1)*NUM_PER_FILE)])
        
    out_df = pd.DataFrame.from_records(res)
    if i == 0:
        # create the initial file
        # write the data in a form of pandas data frame
        out_df.to_csv('./MC_res.csv', header=False, index = False)
    else:
        # append it to the file
        out_df.to_csv('./MC_res.csv', mode='a', header=False, index = False)