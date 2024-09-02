# Normalize,aug,search
# -*- coding: utf-8 -*-
import logging
import numpy as np
import pandas as pd
import random
import multiprocess

from matrixprofile.algorithms import mpdist
from matrixprofile.algorithms.snippets import snippets
from sklearn import preprocessing
import json

# from lib.SANNI.Preprocess.const import *
import time
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score

# from API import Image

# from utils.logs import log_func as lg_f
#
#
# def log_func(*args):
#     lg_f(*args, level='sanni_preprocc')


def check_intersections(arr,
                        percent=0.25):
    i_arr = np.arange(1, len(arr))
    # log_func(arr.shape)
    snippets = []
    snippet_i = 0
    logging.info(f'check_intersections:{len(i_arr)}')
    # return
    while len(i_arr) > 0:
        start_time = time.time()

        snippets.append(snippet_i)
        # print(arr[snippet_i])

        sub_sub_a = arr[snippet_i]

        def check_two_sub(i):
            indx = i_arr[i]
            sub_sub_b = arr[indx]

            # in1d = np.where(np.in1d(sub_sub_a, sub_sub_b))[0]
            count = 0
            lens = len(sub_sub_a)
            for j, data in enumerate(sub_sub_b):
                if data in sub_sub_a:
                    count += 1
                #   index += 1
                # elif index != 0 and count / lens >= percent:
                #   return i
                # elif index != 0:
                #    break
                if count / lens >= percent:
                    #    print(123)
                    return i
            # return indx
            #     if len(in1d) > arr.shape[1] * percent:
            """
                                start = in1d[0]
                                last = start
                                max_ = -np.inf
                                for sub_in1d in in1d[1:]:
                                    if sub_in1d - 1 != last:
                                        len_ = last - start
                                        if len_ > max_:
                                            max_ = len_
                                        start = sub_in1d
                                    last = sub_in1d
                                len_ = sub_in1d - start
                                if len_ > max_:
                                    max_ = len_
                                if max_ >= 1:
                                    return i
                                """

        #         return i

        pool_obj = multiprocess.Pool()
        inds = np.arange(len(i_arr))
        del_arr = np.array(pool_obj.map(check_two_sub, inds))
        del_arr = del_arr[del_arr != np.array(None)]
        del_arr = del_arr.astype(np.int32)
        #   print(del_arr)
        i_arr = np.delete(i_arr, del_arr)
        # print(i[])
        pool_obj.close()
        pool_obj.join()

        #  i_arr = [i for j, i in enumerate(i_arr) if j not in del_arr]
        if len(i_arr) > 0:
            snippet_i = i_arr[0]
    #    print("--- %s seconds ---" % (time.time() - start_time))

    #   print('check_intersections:', len(i_arr))
    return snippets


def augmentation(data: pd.DataFrame, e=0.01):
    """
    Увеличение и балансировка соседей.
    Все соседи сниппетов увеличиваются до количеста соседей у сниппета с максиальным fraction
    :param data: dataframe, в котором хранятся сниппеты и их соседи
    :param e: 0<e<1 процент, на который можно сдвинуть точку
    :return: возвращается datafraaugmentationme той же структуры, но со сбалансированными соседями
    """
    subseq_count = [(i, len(np.array(data.neighbors.iloc[i]))) for i in range(0, len(data.neighbors))]
    max_subseq_count = max([subseq_count[i][1] for i in range(0, len(subseq_count))])

    new_neighbors_all = []
    for cl in range(0, len(data.neighbors)):
        if subseq_count[cl][1] == max_subseq_count:
            new_neighbors_all.append(data.neighbors[cl].copy())
            continue
        neighbors = data.neighbors[cl].copy()
        need_new_neighbors = (max_subseq_count - subseq_count[cl][1])
        need_double_new = need_new_neighbors - subseq_count[cl][1] if need_new_neighbors - subseq_count[cl][
            1] > 0 else 0
        need_new_neighbors -= need_double_new
        for i in range(0, need_new_neighbors):
            new_neighbor = neighbors[i]
            new_neighbor[random.randint(0, len(neighbors[i]) - 1)] *= 1 + random.uniform(-e, e)
            neighbors.append(new_neighbor)
            if need_double_new > 0:
                new_neighbor = neighbors[i]
                new_neighbor[random.randint(0, len(neighbors[i]) - 1)] *= 1 + random.uniform(-e, e)
                neighbors.append(new_neighbor)
                need_double_new -= 1
        new_neighbors_all.append(neighbors)

    data['neighbors'] = new_neighbors_all
    return data


# def create_dataset(size_subsequent: int, dataset, snippet_count=0):
#     """
#     Создает zip архив в директории датасета с размеченными датасетами
#     :param size_subsequent: Размер подпоследовательности
#     :param dataset: Директория датасета
#     :param snippet_count: минимальный fraction
#     :return Возращает колличество сниппетов
#     """
#
#     data_norm, scaler = normalize(dataset)
#     # pickle.dump(scaler, (dataset / FILE_SCALER).open('wb'))
#     # data_norm = data
#     # print(data_norm.shape)
#     # FIXME на время сравнения с орбитс
#     # np.savetxt(dataset / NORM_DATA_FILE_NAME, data_norm)
#     log_func("Начал поиск сниппетов", __name__)
#
#     max_snippet = -1
#     for idx, data in enumerate(data_norm.T):
#         if snippet_count == 0:
#             distant = get_distances(ts=data,
#                                     snippet_size=size_subsequent)
#             count_snippet = get_count(distances=distant,
#                                       snippet_size=size_subsequent,
#                                       len_ts=len(data))
#             snippet_count = count_snippet
#         else:
#             count_snippet = snippet_count
#         if count_snippet > max_snippet:
#             max_snippet = count_snippet
#         log_func(f"Для {idx + 1} признака найденно снипеттов:{count_snippet}")
#         snippet_list = search_snippet(data=data,
#                                       snippet_count=snippet_count,
#                                       size_subsequent=size_subsequent)
#         snippet_list.snippet = snippet_list.snippet.apply(lambda x: json.dumps(x.tolist()))
#         snippet_list.to_csv(dataset / SNIPPET_FILE_NAME.format(idx + 1), compression='gzip')
#
#     result = {
#         "size_subsequent": size_subsequent,
#         "snippet_count": max_snippet,
#         "classifier_model": False,
#         "save": True
#     }
#
#     with open(dataset / CURRENT_PARAMS_FILE_NAME, 'w') as outfile:
#         json.dump(result, outfile)
#     print("Сохранил сниппеты")
#     return max_snippet, scaler


def search_snippet(ts: np.ndarray,
                   num_snippets: int,
                   distances,
                   indices):
    """
    Поиск снипетов
    :param data: Директория временного ряда: str
    :param snippet_count: int
    :param size_subsequent: Размер подпоследовательности - int
    :return: Массив снипеетов - np.ndarray
    """
    # fixme: многомерность
    snippet_size = ts.shape[1]
    time_series_len = ts.shape[0]
    snippets = []
    minis = np.inf
    total_min = None
    for n in range(num_snippets):
        minims = np.inf
        #    print(len(indices))
        index = -1
        for i in range(len(indices)):
            s = np.sum(np.minimum(distances[i, :], minis))
            #  print(s)
            if minims > s:
                minims = s
                index = i
        #   print(minims)
        if index == -1:
            raise ValueError('not find shippet')
        minis = np.minimum(distances[index, :], minis)
        actual_index = indices[index]
        snippet = ts[actual_index]
        snippet_distance = distances[index]
        snippets.append({
            'index': actual_index,
            'snippet': snippet,
            'distance': snippet_distance
        })

        if isinstance(total_min, type(None)):
            total_min = snippet_distance
        else:
            total_min = np.minimum(total_min, snippet_distance)

    for snippet in snippets:
        mask = (snippet['distance'] <= total_min)
        arr = np.arange(len(mask))
        max_index = time_series_len - snippet_size
        snippet['neighbors'] = list(filter(lambda x: x <= max_index, arr[mask]))
        if max_index in snippet['neighbors']:
            last_m_indices = list(range(max_index + 1, time_series_len))
            snippet['neighbors'].extend(last_m_indices)
        snippet['fraction'] = mask.sum() / (len(ts))
        total_min = total_min - mask
        del snippet['distance']

    return snippets


def normalize(sequent: np.ndarray) -> (np.ndarray, preprocessing.MinMaxScaler):
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(sequent)
    return x_scaled, min_max_scaler


def get_distances(ts, indices, window_size=None):
    if window_size is None:
        window_size = int(np.floor(ts.shape[1] / 2))

    def func_calc(j):
        buffer = []
        for i in range(0, len(ts)):
            buffer.append(mpdist(ts=ts[i],
                                 ts_b=ts[j],
                                 w=int(window_size)))
        return buffer

    pool_obj = multiprocess.Pool()
    distances = np.array(pool_obj.map(func_calc, indices))
    pool_obj.close()
    del pool_obj
    return distances


def get_count(distances, indices, max_k=9):
    profilearea = []

    minis = np.inf
    for n in np.arange(max_k):
        minims = np.inf
        for i in np.arange(len(indices)):
            s = np.sum(np.minimum(distances[i, :], minis))

            if minims > s:
                minims = s
                index = i

        minis = np.minimum(distances[index, :], minis)
        profilearea.append(np.sum(minis))
    change = -np.diff(profilearea)
    for i in np.arange(2, len(change)):
        count = (np.trapz(change[:i], dx=1) - np.trapz(change[:i - 1])) / (np.trapz(change[:i], dx=1) + 1)
        if count < 0.3:
            return i - 1
    return len(change)


def smape(a, f):
    return 100 / len(a) * np.sum(np.abs(f - a) / ((np.abs(a) + np.abs(f)) / 2))


def get_score(y_true, y_predict):
    y_true = np.array(y_true)
    y_predict = np.array(y_predict)
    loss = {}
    loss["mse"] = mean_squared_error(y_true, y_predict)
    loss["rmse"] = mean_squared_error(y_true, y_predict, squared=False)
    loss["mae"] = mean_absolute_error(y_true, y_predict)
    loss["mape"] = mean_absolute_percentage_error(y_true + 1, y_predict + 1)
    loss["smape"] = smape(y_true + 1, y_predict + 1)
    loss["r2"] = r2_score(y_true + 1, y_predict + 1)
    return loss


def get_snippets(arr, count_snippet=-1, windows_size=None):
    # nan_index = np.unique(np.where(np.isnan(arr))[0])
    # not_nan_index = np.arange(len(arr))
    # not_nan_index = [i for j, i in enumerate(not_nan_index) if j not in nan_index]

    not_nan_index = get_not_nan_indices(arr)
    arr_not_nan = arr[not_nan_index]
    # print(arr_not_nan.shape)
    # print(arr.shape)
    indices = check_intersections(arr_not_nan)
    snippets = []
    logging.info('Search snippet')
    for i in np.arange(arr.shape[2]):
        distantes = get_distances(arr_not_nan[:, :, i], indices, windows_size)
        if count_snippet == -1:
            count_snippet = get_count(distantes, indices)
        snippet = search_snippet(arr_not_nan[:, :, i],
                                 count_snippet, distantes, indices=indices)
        snippets.append(snippet)
    return snippets


def get_not_nan_indices(arr):
    nan_index = np.unique(np.where(np.isnan(arr))[0])
    not_nan_index = np.arange(len(arr))
    not_nan_index = [i for j, i in enumerate(not_nan_index) if j not in nan_index]
    return not_nan_index
