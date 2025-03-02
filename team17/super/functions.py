import altair as alt 
import hashlib 
import numpy as np
import os
import pandas as pd
import re
import vega_datasets

from pathlib import Path

class DataCleaningFunctions:
    def __init__(self):
        pass

    def filter_values(self, df: pd.DataFrame) -> pd.DataFrame: 
        '''Implement rules for filtering'''

        # Filter out anomalous data (charging equipment failed)
        df = df[df['flag_id'] == 0]

        # Filter out charge rates that are errorneous
        df = df[(df['energy_kwh'] / df['charge_duration']) < 350]

        return df

    def deal_with_na(self, df: pd.DataFrame) -> pd.DataFrame:
        '''Function to modularize the workings of na's'''
        
        # Remove undesignated rows
        df = df[df.pricing != 'Undesignated']
        df = df[df.metro_area != 'Undesignated']
        
        # Assume charge durations < 0 are false
        df = df[df.charge_duration >= 0]
        
        # Check the end_soc - start_soc
        df = df[(df.end_soc - df.start_soc) > 0]

        
        return df.dropna()

    def drop_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        '''Function to drop columns and explain why'''
        
        # For sure columns
        df = df.drop(columns=['session_id'])      # Not Needed: Index of data. Don't need for predictions
        df = df.drop(columns=['end_datetime'])    # Leakage: knowing start/end datetimes is insight to total time
        df = df.drop(columns=['total_duration'])  # Leakage: Knowing total duration is directly linked to charge 
                                                  #          duration
        df = df.drop(columns=['flag_id'])         # Leakage: Maintenance codes are signs of trouble (not charging)
                                                  #          and are only seen after a prediction is made
        df = df.drop(columns=['region'])          # Not Needed: Per Matt's demo
        
        # Iffy columns
        df = df.drop(columns=['energy_kwh'])      # Leakage: Knowing the energy received could be insight to total 
                                                  #          time. Though we know the start soc which. 
        df = df.drop(columns=['end_soc'])         # Leakage: Might need this. Though I might use it later to filter 
                                                  #          to only end soc > 95% or something
        df = df.drop(columns=['charge_level'])    # Not Needed: idk... it makes it slightly easier and might be leakage
            
        return df

    def clean_date_column(self, df: pd.DataFrame) -> pd.DataFrame:
        '''Function to modify the date'''
        
        # Clean up date
        df.start_datetime = pd.to_datetime(df.start_datetime)
        df['year'] = df.start_datetime.dt.year
        df['month'] = df.start_datetime.dt.month
        df['day'] = df.start_datetime.dt.day
        df['hour'] = df.start_datetime.dt.hour
        df['day_of_year'] = df.start_datetime.dt.day_of_year
        df['quarter'] = df.start_datetime.dt.quarter
        df['weekday'] = df.start_datetime.dt.weekday
        df['week_num'] = df.start_datetime.dt.day_of_year // 7
        
        # Drop the original date
        df = df.drop(columns=['start_datetime'])
        
        return df

    def change_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        '''Corrects data types'''
        
        df.num_ports = df.num_ports.astype(int)  # Can only have integer ports
        
        return df
    
    def clean_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        '''Function to clean dataset'''
        # Filter data
        df = df[(df.end_soc - df.start_soc) > 0]  # Anomalies
        df = df[df.end_soc >= 0.95]      # Anything before this, the operator disconnected before finishing
        df = df[df.charge_duration <= 1]  # People be leaving these on the chargers
        df = df[df['flag_id'] == 0]
        df = df[(df['energy_kwh'] / df['charge_duration']) < 350]
        
        # Deal with NaN's
        df = self.deal_with_na(df)
        
        # Clean the date
        df = self.clean_date_column(df)
        
        # Data types
        df = self.change_data_types(df)

        # Filter 
        df = self.filter_values(df)

        # Drop data leak columns
        df = self.drop_columns(df)
        
        return df

class LoadData:
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

    def load_connector(path: Path = Path('assets/evwatts.public.connector.csv')) -> pd.DataFrame:
        '''Loads the connector data'''

        # Read in the csv 
        df = pd.read_csv(path)

        return df

    def merge_data(
        sessions: pd.DataFrame, 
        evse: pd.DataFrame,
        connector: pd.DataFrame) -> pd.DataFrame:
        '''Merges the data'''

        # Merge the connectors and evse 
        temp_df = pd.merge(evse, connector[['evse_id', 'power_kw']], how='left', on='evse_id')

        return pd.merge(sessions, temp_df, how='left', on='evse_id')

class DataIndex:
    def __init__(self, dataset_name: str, n: int, n_samples: int = 200):
        
        self.dataset_name = dataset_name  # Name of array
        self.n = n                        # Length of data
        self.n_samples = n_samples        # Number of samples per sampling
        self._subsets = {}                # Dictionary to track subsets

        # Add the current subset
        self._subsets['full'] = np.arange(n)
        self._subsets['current'] = np.arange(n)
        
    def new_subset(self, subset_name: str) -> np.array:
        '''Generate a new subset'''

        # Validate subset_name
        if subset_name in self._subsets.keys():
            return self._subsets[subset_name]

        # Get a new subset
        current = self._subsets['current'][:]
        subset = np.random.choice(current, self.n_samples, replace=False)

        # Update current
        self._subsets['current'] = np.array(list(set(current) - set(subset)))
        self._subsets[subset_name] = subset

        return self._subsets[subset_name]

    def subset(self, subset_name: str) -> np.array:
        '''Return the subset of a given name'''

        return self._subsets[subset_name]

def chart_topologogical(source: pd.DataFrame) -> alt.Chart:
    '''Creates a geoplot used in the data cleaning
    
    This function can easily be modified for future use for any geo map. source 
    is a pd.DataFrame that must include a "lat" and "lng" coordinate. The size 
    of the points is based on charge_duration, but this can easily be swapped 
    to any variable.
    '''

    # Get states topo
    states = alt.topo_feature(vega_datasets.data.us_10m.url, feature='states')

    # Create the background (US states)
    background = alt.Chart(states).mark_geoshape(
        fill="lightgray",
        stroke="white"
    ).properties(
        width=750,
        height=500
    ).project("albersUsa")

    # Create the points ()
    points = alt.Chart(source).mark_circle().encode(
        latitude='lat',
        longitude='lng',
        size=alt.Size("charge_duration:Q").legend(None).scale(range=[0, 1000]),
        tooltip=['metro_area', alt.Tooltip('charge_duration', format='.2f')]
    )


    return (background + points).properties(title='Mean Charge Duration By Metro Area')

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

def get_paths(path_to_assets: str = None) -> tuple[Path, Path]:
    '''Returns a tuple of paths'''

    # Automatically set path_to_assets
    if not path_to_assets:
        path_to_assets = 'assets'

    # Join path strings and create Paths
    sessions_path = Path(os.path.join(path_to_assets, 'evwatts.public.session.csv'))
    evse_path = Path(os.path.join(path_to_assets, 'evwatts.public.evse.csv')) 

    return sessions_path, evse_path

