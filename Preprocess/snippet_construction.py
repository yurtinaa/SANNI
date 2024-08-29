import numpy as np
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool


def calculate_distance(data, snippet_new, mask, i_idx, len_data):
    """
    Вычисляет расстояние между фрагментом данных и фрагментом сниппета.

    Args:
        data (numpy.ndarray): Массив данных.
        snippet_new (numpy.ndarray): Повторенный сниппет.
        mask (numpy.ndarray): Маска для фильтрации пропусков в данных.
        i_idx (int): Индекс для итерации по данным.
        len_data (int): Длина данных.

    Returns:
        tuple: Кортеж, содержащий индекс и расстояние.
    """
    filtered_snippet = snippet_new[i_idx:i_idx + len_data][mask]
    distance = np.linalg.norm(data[mask] - filtered_snippet)
    return i_idx, distance


def find_start_index(data, snippet):
    """
    Находит индексы начала и конца оптимального фрагмента данных для сниппета.

    Args:
        data (numpy.ndarray): Массив данных.
        snippet (numpy.ndarray): Сниппет для сравнения.
    Returns:
        tuple: Кортеж, содержащий индексы начала и конца оптимального фрагмента.
    """
    snippet_new = np.concatenate([snippet, snippet])
    result_indx = -1
    min_ = np.inf
    len_data = len(data)
    mask = ~np.isnan(data[:len_data // 2])

    with ThreadPoolExecutor() as executor:
        futures = []
        for i_idx in range(len_data):
            futures.append(executor.submit(calculate_distance,
                                           data[:len_data // 2],
                                           snippet_new,
                                           mask,
                                           i_idx,
                                           len_data // 2))

        for future in futures:
            i_idx, distance = future.result()
            if min_ > distance:
                min_ = distance
                result_indx = i_idx

    start_index = len(data[result_indx:])
    len_data = len(data) - start_index
    first_index = result_indx
    min_ = np.inf
    result_indx = -1
    mask = ~np.isnan(data[start_index:])

    with ThreadPoolExecutor() as executor:
        futures = []
        for i_idx in range(len(data)):
            futures.append(
                executor.submit(calculate_distance,
                                data[start_index:],
                                snippet_new,
                                mask,
                                i_idx,
                                len_data))

        for future in futures:
            i_idx, distance = future.result()
            if min_ > distance:
                min_ = distance
                result_indx = i_idx
    second_part_len = len(data) - len(snippet[first_index:]) + result_indx
    # print(second_part_len, len(data))
    return first_index, result_indx


def process_chunk(args):
    chunk, arr_class, snippets, data, win_size = args
    constract_snippet = []
    for i in chunk:
        class_number = arr_class[i]
        snippet = snippets[class_number]
        sub_seq = data[i].copy()
        idx, second_idx = find_start_index(sub_seq, snippet)
        second_part_len = len(sub_seq) - len(snippet[idx:]) + second_idx
        dub_snippet = np.concatenate([snippet, snippet])

        plot_snippet = np.concatenate([snippet[idx:], dub_snippet[second_idx:second_part_len]])[:win_size]
        constract_snippet.append((i, plot_snippet))
    return constract_snippet


def parallel_processing(arr_class, snippets, data, win_size, chunk_size, num_processes=8):
    chunks = [list(range(i, min(i + chunk_size, len(data)))) for i in range(0, len(data), chunk_size)]
    result = []
    with Pool() as pool:
        # Вместо pool.imap_unordered используем pool.map
        for chunk_result in pool.map(process_chunk,
                                     zip(chunks, [arr_class] * len(chunks), [snippets] * len(chunks),
                                         [data] * len(chunks), [win_size] * len(chunks))):
            result.extend(chunk_result)

    # Сортировка результатов по индексам
    # result.sort(key=lambda x: x[0])

    # Возврат только второго элемента из каждого кортежа
    return [x[1] for x in result]
