from settings import DATA_DIR
import eurostat

import pandas as pd
import numpy as np

import yfinance

from datetime import datetime
from statsmodels.tsa.seasonal import seasonal_decompose

import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from webdriver_manager.microsoft import EdgeChromiumDriverManager
from webdriver_manager.chrome import ChromeDriverManager

from tqdm import tqdm
import chromadb
from chromadb.utils import embedding_functions

import pandas as pd
import numpy as np
import string

from itertools import compress

import json
import time


def clean_punctuation(list_of_tokens: list):
    translate_table = dict((ord(char), None) for char in string.punctuation)
    lst_cleaned = [s.translate(translate_table) for s in list_of_tokens]
    return [s for s in lst_cleaned if s != '']
    
def get_pair_lag(spe: dict, item: str, lag: int=1):
    l =[
        [spe[idx][item] for idx in list(spe.keys())][:-lag],
        [spe[idx][item] for idx in list(spe.keys())][lag:],
    ]
    return list(map(list, zip(*l)))


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


def get_chromaDB_collection(name: str):
    CHROMA_DATA_PATH = "chroma_data/"
    EMBED_MODEL = "all-MiniLM-L6-v2"

    client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)

    collection = client.get_or_create_collection(
        name=name,
        embedding_function=embedding_func, metadata={"hnsw:space": "cosine"}
    )
    return client, collection


def vec_similarity(a: np.array, b: np.array) -> np.ndarray:
    """
    The similarity score of two vectors is calculated as the dot product of two vectors normed to unity
    :param a:
    :param b:
    :return:
    """
    return np.dot(to_unity(a), to_unity(b))


def get_selenium_driver(url):
    d = webdriver.Chrome(executable_path=ChromeDriverManager().install())
    # driver = webdriver.Edge(EdgeChromiumDriverManager().install())
    d.get(url)
    time.sleep(3)
    return d


def dump_json(file, file_path):
    with open(file_path, "w") as outfile:
        json_obj = json.dumps(file)
        outfile.write(json_obj)
    pass


def load_json(file_path):
    return json.load(open(file_path))


def get_lazy_load(d, clever_scroll: bool=False):

    if clever_scroll:
        last_height = d.execute_script("return document.body.scrollHeight")
        while True:
            d.execute_script(f"window.scrollTo({str(last_height)}, document.body.scrollHeight);")
            time.sleep(3)
            new_height = d.execute_script("return document.body.scrollHeight")

            if new_height == last_height:
                break
            last_height = new_height
    else:
        for i in range(0, 20):
            time.sleep(2)
            current_height = d.execute_script("return document.body.scrollHeight")
            d.execute_script(f"window.scrollTo({str(current_height)}, {str(current_height+500)});")
    return d



if __name__ == "__main__":

    # ecb data
    codes = [
        'TEINA110',  # gdp deflator
        # 'SDG_08_10', # real gdp per capita
        'prc_hicp_midx',  # consumer price index
        'TIPSAU20',  # quarterly gdp at market prices
    ]

    # load data
    datas = [eurostat.get_data(code=i) for i in codes]
    gdp_dfl, cpi, gdp = [pd.DataFrame(i[1:], columns=i[0]) for i in datas]

    gdp = pd.DataFrame(gdp.iloc[:, 5:].sum(axis=0).rename('gdp')).astype(float).dropna().iloc[:-1]
    gdp.index = [datetime(int(i[:4]), (int(i[6:7])) * 3 - 3 + 1, 1) for i in gdp.index]

    cpi = pd.DataFrame(cpi[(cpi['geo\\TIME_PERIOD'] == 'EA19')].transpose().iloc[5:, 0].rename('cpi')).astype(
        float).dropna()
    cpi = cpi.join(np.log(cpi).diff().cpi.rename('cpi_diff'))
    cpi.index = [datetime(int(i[:4]), int(i[5:8]), 1) for i in cpi.index]

    controls = gdp.join(cpi).dropna()

    sd = seasonal_decompose(controls.cpi, period=4, two_sided=False)
    controls['cpi_cyc'] = sd.resid

    sd = seasonal_decompose(controls.gdp, period=4, two_sided=False)
    controls['gdp_cyc'] = sd.resid

    # eurostox 50
    start, end = datetime(1990, 1, 1), datetime.now()
    estox = yfinance.download(tickers = "^STOXX50E", start=start, end=end)
    estox.index.name = 'date'

    controls = controls.join(estox['Adj Close'].rename('estoxx_50'), how='outer')

    controls.to_csv(os.path.jon(DATA_DIR, 'controls.csv'))