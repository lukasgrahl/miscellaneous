import numpy as np
import pandas as pd
from itertools import chain, combinations


def is_matrix_transitive(M) -> bool:
    """
    This function checks matrix transitivity checking whether zeros "survive" matrix multiplication
    :param M: matrix of interest
    :return: is transitive
    """
    return np.sum(np.sum((((M @ M) > 0) * 1) - M, axis=0), axis=0) == 0


def to_unity(v) -> np.array:
    """
    this function norms a vector to unity as implemented in spacy
    :param v: vector
    :return: normed vector
    """
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    else:
        return v / norm


def vec_similarity(a: np.array, b: np.array) -> np.ndarray:
    """
    The similarity score of two vectors is calculated as the dot product of two vectors normed to unity
    :param a:
    :param b:
    :return:
    """
    return np.dot(to_unity(a), to_unity(b))


def get_transitivity_candidates(all_tokens, similarity_threshold: float = .8) -> np.array:
    """
    This function considers all token pairs and returns an array of pairs with a similarity score larger than
    the similarity threshold
    :param all_tokens: list of spacy tokens
    :param similarity_threshold:
    :return: array of format np.array([["lemma1", "lemma2", bool]])
    """
    lemmas = [i.lemma_ for i in all_tokens]
    unq_str_tokens_idx = [lemmas.index(i) for i in set(lemmas)]
    unq_str_tokens = [[all_tokens[i]] for i in unq_str_tokens_idx]

    combs = [*combinations(unq_str_tokens, 2)]
    combs = [(i[0][0], i[1][0]) for i in combs]

    similarity = [vec_similarity(i[0].vector, i[1].vector) for i in combs]
    df = pd.DataFrame(similarity, index=pd.Index([(i[0].lemma_, i[1].lemma_) for i in combs]), columns=['similarity'])
    similar_pairs = df[df.similarity >= similarity_threshold].reset_index().values[:, :2]

    return similar_pairs


def transitivity_check_first_level(all_pairs: np.array, pairs_tb_checked: list) -> set:
    """
    1st step for checking transitvity
    For a given set of pairs with relation A (pairs_tb_checked), this functions finds all other pairs in (all_pairs)
    that are linked to pairs_tb_checked. Linkage is understood as an intersection between pairs e.g. two pairs contain
    the word "EU" are linked for that reason.
    :param all_pairs: array of all pairs in form of np.array([(word#1, word#2), ])
    :param pairs_tb_checked: list of pairs tb checked [(word#1, word#2)
    :return: set of topics
    """

    set_topics = set()
    # checks for a given pair, whether a str_tb_checked in lst_tb_checked is contained in XOR value of the pair
    for item in all_pairs:
        for str_tb_checked in pairs_tb_checked:
            # XOR should be excluded through drop duplicates, thus OR
            if (str_tb_checked == item[0]) ^ (str_tb_checked == item[1]):
                set_topics.add(tuple(item[:2]))
    return set_topics


def transitivity_check_second_level(set_topics2: set, set_topics1: set) -> list:
    """
    2nd level to transitivity check
    For each word in the past iterations set of pairs (set_topics1), check wether from step 1 new pairs have been added,
    introducing new words with potential linkage
    :param set_topics2:
    :param set_topics1:
    :return:
    """

    # checks for words in pairs, if the word is not contained in any of the previous iteration's pairs
    new_items = [
        i for i in [
            *chain(
                *[[item for item in items] for items in set_topics2]
            )
        ] if i not in [
            *chain(
                *[[item for item in items] for items in set_topics1]
            )
        ]
    ]
    return list(set(new_items))


def get_transitive_cluster(arr_similar_pairs: np.array, max_recursion: int = 100) -> list:
    """
    This functions obtains cluster candidates by checking which similar pairs are connected
    The cluster candidate is obtained once all interconnections are found, it is then checked for transitivity
    :param arr_similar_pairs: np.array of expression pair tuples
    :param max_recursion: maximum number of recursions, to control runtime
    :return: list of clusters and boolean for transitivity
    """
    OUTPUT = []  # output of cluster with boolean for transitivity
    LST_HAS_BEEN_CHECKED = []  # list of all expression which have been checked as part of a cluster candidate

    for idx in range(0, arr_similar_pairs.shape[0]):

        counter = 0  # recursion counter
        # expressions which have not been checked for interconections
        lst_tb_checked_recursion = list(arr_similar_pairs[idx])

        # if both expression are contained in a previous cluster candidate skip
        if (lst_tb_checked_recursion[0] in LST_HAS_BEEN_CHECKED) \
                and (lst_tb_checked_recursion[1] in LST_HAS_BEEN_CHECKED):
            continue

        # cluster candidate consisting of one or more pairs
        set_cluster_candidates = set()
        set_cluster_candidates.add(tuple(lst_tb_checked_recursion))

        while len(lst_tb_checked_recursion) > 0:
            # 1st level: finds all pairs that contain the expression from lst_tb_checked_recursion
            set_topics = transitivity_check_first_level(arr_similar_pairs, lst_tb_checked_recursion)
            # 2nd level: checks whether the new pairs from set_topics, link to other pairs -> recursion
            lst_tb_checked_recursion = transitivity_check_second_level(set_topics, set_cluster_candidates)
            # updating the cluster candidates
            set_cluster_candidates.update(set_topics)

            if counter > max_recursion:
                print(f'maximal recursion depth {max_recursion} has been reached for: {arr_similar_pairs}')
                break
            # tracking recursion depth
            counter += 1

        # each cluster candiates can be represented in a matrix, where the relation between expressions is a boolean
        matrix = pd.DataFrame(
            [
                *chain(
                    *[
                        [
                            (i[0], i[1], True), (i[1], i[0], True)
                        ] for i in set_cluster_candidates
                    ]
                )
            ],
            columns=['level1', 'level2', 'is_relation']
        ).pivot_table(index='level1', columns='level2', values='is_relation').fillna(0)
        np.fill_diagonal(matrix.values, True)  # fill diagonal to be True, true by definition -> square matrix

        # check transitivity
        is_transitive = is_matrix_transitive(matrix)

        # append output
        lst_overall_set = list(set_cluster_candidates)
        lst_overall_set.sort()
        OUTPUT.append((tuple(lst_overall_set), is_transitive))

        # append items to has been_checked
        LST_HAS_BEEN_CHECKED.extend(list({*chain(*[list(item) for item in lst_overall_set])}))

        # return list of unique lemmas for each transitive cluster
        lst_cluster_lemma = [list({*chain(*[list(i) for i in cluster[0]])}) for cluster in OUTPUT if cluster[-1]]

    return lst_cluster_lemma
