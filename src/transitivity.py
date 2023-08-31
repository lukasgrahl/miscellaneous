import numpy as np
import pandas as pd
from itertools import chain, combinations

def get_candidates(all_tokens, similarity_factor: float = .8):

    lemmas = [i.lemma_ for i in all_tokens]
    unq_str_tokens_idx = [lemmas.index(i) for i in set(lemmas)]
    unq_str_tokens = [[all_tokens[i]] for i in unq_str_tokens_idx]

    combs = [*combinations(unq_str_tokens, 2)]
    combs = [(i[0][0], i[1][0]) for i in combs]

    similarity = [vec_similarity(i[0].vector, i[1].vector) for i in combs]
    df = pd.DataFrame(similarity, index=pd.Index([(i[0].lemma_, i[1].lemma_) for i in combs]), columns=['similarity'])
    df['bool_sim'] = df.similarity >= similarity_factor
    df = df[df.bool_sim].reset_index()
    df = df.drop('similarity', axis=1)

    return df.values


def is_matrix_transitive(M):
    return np.sum((((M @ M) > 0) * 1) - M) == 0

def to_unity(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    else:
        return v / norm

def vec_similarity(a, b) -> np.ndarray:
    return np.dot(to_unity(a), to_unity(b))

def get_vector_similarity(a, b) -> np.ndarray:
    """
    returns the similarity of two NEntityClusters
    :param a:
    :param b:
    :return:
    """
    return vec_similarity(a.vector, b.vector)

def _transitvity_check_first_level (all_pairs: np.array, pairs_tb_checked: list) -> set:
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
                set_topics.add(tuple (item [:2]))
    return set_topics

def _transitivity_check_second_level(set_topics2: set, set_topics1: set) -> list:
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
    return list(set (new_items))

def _recursion_transitive_clusters(candidates: np.array, max_recursion: int = 40) -> list:
    """
    This function recursively builds a set of pairs that are linked from a list of candidates.
     For each such set of linked pairs, this function then checks transitivty
    :param candidates: np.array of tuple pairs
    :param max_recursion:
    :return:
    """
    out = [] # output
    lst_has_been_checked = [] # list containing all expression that have been checked

    # print(f'started recursion for {candidates.shape[0]}') # candidates: contains pairs of candidates which stand in relation A
    for idx in range(0, candidates.shape[0]): # progressBar([*range (0, candidates.shape[0])], **kwargs):

        counter = 0 # recursion counter
        #word #1 A word#2 -> 1st_tb_checked for other pairs containig either #1 xor #2

        lst_tb_checked = list(candidates[idx, :2])

        # if canidates have been checked already skip
        # candidates might be related to a previously explored set, thus avoid unecessary runs

        if (lst_tb_checked[0] in lst_has_been_checked) and (lst_tb_checked [1] in lst_has_been_checked):
            continue
        # set of all pairs with relation A, which are linked to either #1 or #2

        OVERALL_SET = set()
        OVERALL_SET.add(tuple (lst_tb_checked))


        while len(lst_tb_checked) > 0:
            # 1st level: finds all pairs that contain (xor) words in lst_tb_checked set_topics = _transitvity_check_first_level(candidates, lst_tb_checked) # 2nd level: checks whether adding new pairs has added new words #N 1st_tb_checked = _transitivity_check_second_level(set_topics, OVERALL_SET) OVERALL SET.update(set_topics)
            # for new words #N, pairs containing #N words are added
            # adding new pairs might add new words # the recursion stops when no new words, requiring adding new pairs occur
            # recursion max to avoid infinite loop, adjust as needed
            if counter > max_recursion:
                # logger.warning(f'maximal recursion depth (max_recursion} has been reached for: {candidates}')
                print(f'maximal recursion depth {max_recursion} has been reached for: {candidates}')
                break

            # tracking recursion depth
            counter += 1

            # each pair yields a set of related pairs, these can be represented in a matrix
            matrix = pd.DataFrame(*chain(*[[(i[0], i[1], True), (i[1], i[0], True)]for i in OVERALL_SET]),
                                  columns = ['Level1', 'level2', 'is_relation'])
            matrix = matrix.pivot_table(index='level1', columns='level2', values='is_relation').fillna(0)
            np.fill_diagonal(matrix.values, True)  # fill diagonal to be True, (true by definition as word#1 == word#1)

            # check tranistivity
            is_transitive = is_matrix_transitive(matrix)
            # append output
            lst_overall_set = list(OVERALL_SET)
            lst_overall_set.sort()
            out.append((tuple(lst_overall_set), is_transitive))

            # append items to has been_checked
            lst_has_been_checked.extend(list({*chain(*[list(item) for item in lst_overall_set])}))

    return out