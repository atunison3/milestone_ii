# In this script we're going to look at the total distribution of charging times (in general), and how they vary based 
# on different power levels, or SOC durations 

import pandas as pd
import EDA_part1

    
def combine_charging_data():
    """
    This function comnines data from the charging stations, charging connectors, and charging sessions
    INPUTS: 
    None
           
    OUTPUTS:
    df_merged: a pandas dataframe object combined from the major datafiles 
    """
    
    #filepaths for the desired data 
    filepath = r'data\\evwatts.public.session.csv'
    station_file = r'data\\evwatts.public.evse.csv'
    connector_file = r'data\\evwatts.public.connector.csv'

    #Read and merge the dataframes of interest 
    df = pd.read_csv(filepath) #Read file
    df["soc_charged"] = df["end_soc"] - df["start_soc"]
    df_redux = EDA_part1.prep_data(input_df=df, condition=2)
    df_stations =  pd.read_csv(station_file)
    df_connectors = pd.read_csv(connector_file)

    #Merge togethercharging data with connector information
    df_merged = pd.merge(df_redux, df_stations, on='evse_id', how='left')
    df_merged = pd.merge(df_merged, df_connectors, on='evse_id', how='left')
    return df_merged


def review_charging(input_df, power_rating, num_ports, specific_soc_duration=False, num_samples = 100000): 
    """
    This function shows the distribution of charging profiles based on the specified power rating and port quantity
    for a given dataframe. 

    INPUTS: 
    input_df: a pandas dataframe containing charging data combined from charging sessions, stations, and the connectors
    power_rating: a string specifying the power rating that user wishes to view a data distribution for 
    num_ports: an integer number specifying how many ports the vehicle is plugged into during charging 
    specific_soc_duration: a boolean entry declarying whether all data is considered or just the specific_soc_duration 
    num_samples: an integer count of the number of samples desired to plot
    
    OUTPUTS: 
    Displays a matplotlib histogram plot showing the PDF & CDF for the data at the given conditions  
    """

    #Apply the specified filter
    plot_df = input_df[(input_df.power_kw==power_rating) & (input_df.num_ports == num_ports)]

    #check if worth plotting
    if len(plot_df) <10: 
        print("Less than 10 observations found for the requested condition. No plotting will be done")

    else: 
        #check num ports to get part of the title string
        if num_ports ==2: 
            port_string = "dual"
        else: 
            port_string = "single"

        #Set title based on plot type
        if specific_soc_duration == False: 
            title_text = f"Total charge time for {power_rating} Power and {port_string} port connection"
        else: 
            min_soc = int(round(input_df.start_soc.mean(),-1))
            max_soc = int(round(input_df.end_soc.mean(),-1)) 
            title_text = f"{min_soc}-{max_soc}% Charge Time for {power_rating} Power and {port_string} port connection"

        #Plot results
        xlabel_text = "Charge Duration (minutes)"
        plot = EDA_part1.hist_plotter(plot_df["charge_duration"]*60, sample_size = num_samples, normalize=True,
                                      xlabel=xlabel_text, title=title_text)


def generate_desired_plots(input_df): 
    """
    This function generates desired plots for to examine the total charging time spent under different conditions
    INPUTS: 
    input_df: a pandas dataframe used for plotting 

    OUTPUTS: 
    numerous matplotlib plots 
    """
    
    #First develop the Plots for General Charge Time vs. Power
    power_list = [">100 kW", ">100 kW", "30 kW - 100 kW", "<30 kW"]
    port_list = [2,1,1,1]

    #Create total charge time plots by power rating and number of chargers 
    for i in range(len(power_list)): 
        review_charging(input_df, power_list[i], port_list[i], specific_soc_duration=False, num_samples = 100000)

    #Prep data for 20-80 charge time: 
    df_20_80 = input_df[(
                         (input_df["start_soc"] < 20.2) & 
                         (input_df["start_soc"] > 19.8) & 
                         (input_df["end_soc"] > 79.8) & 
                         (input_df["end_soc"] < 80.2)
                         )]

    #Now plot for 20-80 Charge Time:
    for i in range(len(power_list)): 
        if i>0: 
            review_charging(input_df, power_list[i], port_list[i], specific_soc_duration=True, num_samples = 100000)


#Execution: 
if __name__ == "__main__": 
    df = combine_charging_data()
    generate_desired_plots(df)

