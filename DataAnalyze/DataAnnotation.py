from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from Preprocess import get_snippets


@dataclass
class SnippetAnnotation:
    fragment_size: int
    class_count: int

    def annotate(self, X: np.ndarray) -> Tuple[List[Dict[int, np.ndarray]],
    List[List[Tuple[int, np.ndarray]]]]:
        snippet_list = get_snippets(X,
                                    count_snippet=self.class_count,
                                    windows_size=self.fragment_size)

        all_snippet = []
        for arr_snippet in snippet_list:
            snippets = {}
            for idx, snippet in enumerate(arr_snippet):
                snippets[idx] = np.array(snippet['snippet'])
            all_snippet.append(snippets)

        train_set = []
        for i in np.arange(len(X)):
            buffer = []
            for dim_idx, dim_data in enumerate(snippet_list):
                for index_snippet, snippet in enumerate(dim_data):
                    if i in snippet['neighbors']:
                        buffer.append((index_snippet, X[i, :, dim_idx]))
                        break
            train_set.append(buffer)

        return all_snippet, train_set
