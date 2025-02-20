if __name__=='__main__':
    from team17_mjonesgm_atunison_szymons._supervised_functions import DataCleaningFunctions, LoadData

    # Read in data
    sessions = LoadData.load_sessions()
    evse = LoadData.load_evse()

    # Initialize data cleaning functions 
    dcf = DataCleaningFunctions()

