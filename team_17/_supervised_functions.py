import hashlib 
import numpy as np
import pandas as pd
import re

from pathlib import Path

def load_sessions(path: Path = Path('assets/evwatts.public.session.csv')) -> pd.DataFrame:
    '''Loads in the sessions data'''

    # Read in the csv
    df = pd.read_csv(path)

    return df 

def load_evse(path: Path = Path('assets/evwatts.public.evse.csv')) -> pd.DataFrame: 
    '''Loads the evse data'''

    # Read in the csv 
    df = pd.read_csv(path)

    return df 

def test_set_check(identifier: int, test_ratio: float, hash: callable) -> bool:
    '''Checks if instance is splitable or not'''
    
    return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio

def split_train_test_by_id(data: pd.DataFrame, test_ratio: float, id_column: str, hash=hashlib.md5) -> pd.DataFrame:
    '''Splits a dataset into training and test data'''
    
    # Get the id's to split by
    ids = data[id_column]
    
    # Get the test set id's
    in_test_set = ids.apply(lambda x: test_set_check(x, test_ratio, hash))
    
    return data.loc[~in_test_set], data.loc[in_test_set]

def search_pattern(pattern: str, search_term: str) -> str:
    '''Performs a search and returns variable if present'''
    
    try:
        return re.findall(pattern, search_term)[0]
    except IndexError:
        return search_term
    except TypeError:
        return float('NaN')
