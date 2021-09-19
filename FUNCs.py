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


def plot_mds(X, target, colors=['green', 'red'], labels=['CTRL', 'HCM']):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    mds = MDS(2, random_state=42)
    X_2d = mds.fit_transform(X_scaled)

    plt.rcParams['figure.figsize'] = [7, 7]
    plt.rc('font', size=14)
    for i in np.unique(target):
        subset = X_2d[target == i]
        x = [row[0] for row in subset]
        y = [row[1] for row in subset]
        plt.scatter(x, y, c=colors[i], label=labels[i])
    plt.legend()
    plt.show()


class FeatureExtraction(object):
    """
    Класс для экстракции фичей. Главная идея не фиксировать случайность, а оседлать её :)

    """

    def __init__(self):
        pass

    def fit_inner_loop(self, n_iter, X_train, X_test, y_train, y_test, C=0.03):

        """
        Будем n_iter раз бутстрепить сбалансированную train выборку из X_train.
        Обучаем лог.рег. с L1-решуляризацией, с коэффициентом как мы отобрали выше.
        Тестим на X_test, значение добавляем в roc_auc_list

        Если на X_test модель работает круче 0.7, то: 
            1) ненулевые фичи модели добавляем в словарик отобранных фичей feature_dict
            2) обновляем число фичей в перменной len_best_feature = len(feature_dict.keys())
        Если нет, то:
            3) дублируем последнее значение в len_best_feature, т.к. число фичей не изменилось
        """

        len_best_feature = [0]  # заводим лист, в котором будем отслеживать изменение количества фичей
        len_best_more_one = [
            0]  # заводим лист, в котором будем отслеживать изменение количества числа включений уже включенных фичей
        roc_auc_list = list()  # аналогично, отслеживаем как меняется roc-auc, так для интереса
        feature_dict = dict()  # # словарь "ген: log.reg.coef"

        for i in range(0, n_iter):

            if not i % 100:
                print(i, 'inner loop fitting...', sep=' ')

            scaler = StandardScaler()
            # чтобы получить сбаланнсированную выборку, бутстрепим отдельно семплы из контроля и из опыта

            mask = np.array(y_train) == 0

            k_len = min(len(mask) - sum(mask),
                        sum(mask))  # размер выборки бутстрепа, берем размер минимальной группы HCM или CTRL

            CTRL_rows = list(compress(range(0, len(mask)), mask))
            HCM_rows = list(compress(range(0, len(mask)), mask == False))

            _HCM_rows = choices(HCM_rows, k=k_len)  # бутстрепим номера строк из группы больных
            _CTRL_rows = choices(CTRL_rows, k=k_len)  # бутстрепим номера строк из группы здоровых

            # объединяем это всё дело обратно

            _X_train = pd.DataFrame(X_train).iloc[_HCM_rows + _CTRL_rows, :]
            _y_train = np.array(y_train)[_HCM_rows + _CTRL_rows]

            # обучаем лог.рег. с ранее отобранным коэффициентом регуляризации

            linear_regressor = linear_model.LogisticRegression(penalty='l1', C=C, solver='liblinear',
                                                               random_state=42)
            linear_regressor.fit(_X_train, _y_train)

            # тестим

            roc_auc = metrics.roc_auc_score(y_score=linear_regressor.predict(X_test), y_true=y_test)
            roc_auc_list.append(roc_auc)

            # далее отбираем фичи из моделей, которые хоть как-то работают (roc_auc > 0.7)

            if roc_auc > 0.7:
                # отбираем смысловые фичи
                mask = linear_regressor.coef_ != 0
                genes = X_train.columns[mask[0]]
                values = linear_regressor.coef_[mask]

                _feature_dict = dict(zip(genes, abs(values) * roc_auc))  # делаем временный словарь "ген: его ценность"

                # обнавляем глобальный словарь фичей
                for gene, values in _feature_dict.items():
                    if gene in feature_dict:
                        feature_dict[gene].append(values)
                    else:
                        feature_dict[gene] = [values]

                len_best_feature.append(len(feature_dict.keys()))

                feature_distr = np.array(list(map(lambda x: len(feature_dict[x]), feature_dict)))
                len_best_more_one.append(sum(feature_distr[feature_distr > 1]))

            else:
                # если модель была говёной, то просто дублируем предыдущее значение. Ну, число фичей то не изменилось :)
                len_best_feature.append(len_best_feature[-1])
                len_best_more_one.append(len_best_more_one[-1])

        self.len_best_more_one = np.array(len_best_more_one)
        self.len_best_feature = np.array(len_best_feature)
        self.feature_dict = feature_dict
        self.roc_auc_list = roc_auc_list

    def fit_all_loops(self, X, y, inner_itter, extr_itter):

        """
        Идея: повторим экстракцию фичей n раз,
        отсортируем каждый из получившихся наборов по важности фичей, найдем размер окна для отбора n-топ фичей,
        в котором состав фичей минимально изменяется от набора к набору. А потом отберем те фичи, которые всегда встречаются в окне этого размера.
        """

        list_feature_dicts = list()

        for i in range(0, extr_itter):
            print(i, 'external loop fitting...', sep=' ')
            # train_test split and transformation
            X_train, X_test, y_train, y_test = train_test_split(X,
                                                                y,
                                                                test_size=0.2,
                                                                random_state=i)

            normalizer = Normalizer()

            X_train = pd.DataFrame(normalizer.fit_transform(X_train), columns=X_train.columns)
            X_test = pd.DataFrame(normalizer.fit_transform(X_test), columns=X_test.columns)

            scaler = StandardScaler()
            X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
            X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

            # отбор фичей
            fe = FeatureExtraction()
            fe.fit_inner_loop(inner_itter,
                              X_train, X_test, y_train, y_test)
            list_feature_dicts.append(fe.feature_dict)

        self.list_feature_dicts = list_feature_dicts

    def fold_size_finder(self):
        """
        берем list_feature_dicts и ищем окно размера n,
        в котором множетства первых n генов во всех словарях максимально пересекаются
        :return: вектор значений = долям пересекающихся генов при окнках размера i (1<i<max(длина словаря))
        """

        lfd_sort = list()
        for d in self.list_feature_dicts:
            lfd_sort.append(dict(sorted(d.items(), key=lambda item: len(item[1]), reverse=True)))

        self.list_feature_dicts_sort = lfd_sort

        # поиск первых n-мест, которые не изменяются
        result = list()
        max_len = max(list(map(lambda x: len(x.keys()), lfd_sort)))  # минимальная длина словаря feature_dict

        for yeld in range(1, max_len):

            similarity = np.zeros((len(lfd_sort), len(lfd_sort)))

            for i in range(0, len(lfd_sort)):

                one = list(lfd_sort[i].keys())
                if len(one) < max_len:
                    one = one + [i for i in range(0, max_len - len(one))]

                for j in range(0, len(lfd_sort)):
                    two = list(lfd_sort[j].keys())
                    if len(one) < max_len:
                        two = two + [i for i in range(0, max_len - len(two))]

                    one_set = set(one[:yeld])
                    two_set = set(two[:yeld])
                    sim = len(one_set.intersection(two_set)) / yeld
                    similarity[i, j] = sim

            result.append(np.mean(similarity))

        self.fold_sizes = result


from sklearn.utils import resample


class Bootstrap:
    '''Bootstrap cross validator.'''

    def __init__(self, n_bootstraps=5):
        self.nb = n_bootstraps

    def split(self, X, y=None):
        '''"""Generate indices to split data into training and test set.'''
        iX = np.mgrid[0:X.shape[0]]
        for i in range(self.nb):
            train = resample(iX)
            test = [item for item in iX if item not in train]
            yield (train, test)


class FeatureTester:

    def __init__(self, X, y, model, features, n_iter):
        self.X = X
        self.y = y
        self.features = features
        self.model = model
        self.n_iter = n_iter

    def bootstrap_roc_auc(self, bf):

        Boot = Bootstrap(n_bootstraps=self.n_iter)

        X = self.X.filter(bf, axis=1)
        roc_auc = list()

        for train, test in Boot.split(X):
            clfStack = self.model.fit(X.iloc[train, :], self.y[train])
            roc_auc.append(metrics.roc_auc_score(y_score=clfStack.predict(X.iloc[test, :]), y_true=self.y[test]))
        return roc_auc

    def bootstrap_model_test(self, bob_feature=None):

        if bob_feature == None:
            derictional = 'remove'
        else:
            derictional = 'append'

        roc_auc_pull = dict()
        print('directional = ', derictional, self.n_iter, ' iteraion')

        for feature in ['all'] + self.features:
            print(feature, '...', sep='')

            if derictional == 'remove':
                if feature != 'all':
                    bf = self.features.copy()
                    bf.remove(feature)
                else:
                    bf = self.features
            elif derictional == 'append':
                if feature in bob_feature: continue
                if feature != 'all':
                    bf = self.features.copy()
                    bf.append(feature)
                else:
                    bf = bob_feature

            roc_auc_pull[feature] = self.bootstrap_roc_auc(bf=bf)

        return [roc_auc_pull, derictional]

    @staticmethod
    def feature_stats(roc_auc_pull, derictional='remove'):

        interest_feature = dict()

        # print('median(all) - median(-feature)', 'directional =', derictional, end='\n')

        for key in roc_auc_pull.keys():
            pvalue = stats.mannwhitneyu(roc_auc_pull[key], roc_auc_pull['all']).pvalue
            feature_impact = np.median(roc_auc_pull['all']) - np.median(roc_auc_pull[key])

            #            print(key, feature_impact, pvalue, sep='\t\t')

            if derictional == 'remove':
                if pvalue < 0.05 and np.mean(roc_auc_pull[key]) < np.mean(roc_auc_pull['all']):
                    interest_feature[key] = np.abs(feature_impact)
            elif derictional == 'append':
                if pvalue < 0.05 and np.mean(roc_auc_pull[key]) > np.mean(roc_auc_pull['all']):
                    interest_feature[key] = np.abs(feature_impact)

        return interest_feature

    def find_minimal_feature_set(self):
        print(self.feature_stats(*self.bootstrap_model_test()))