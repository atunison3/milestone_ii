#Import packages
import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde
import matplotlib.ticker as mtick


def violin_plotter(
    input_df: pd.DataFrame, cols: list[str], sample_size: int = 10000, 
    normalize: bool = True, colors: list[str] = None, 
    alt_names: list[str] = None, title: str = None) -> None:
    """This function creates a violin to describe the features found in a dataframe to support 
    exploratory data analysis.
    
    INPUTS
    input_df (required): the pandas dataframe object that the user wishes to generate boxplots for 
    cols (required): a list of column names that the user seeks to include in the boxplot 
    sample_size (optional): the number of samples to create the plot with
    normalize (optional): optional to normalize the plot so feature(s) are aligned to the same scale
    colors (optional): a list of specific colors the user wishes to provide for the boxplot 
    alt_names (optional): a list of names to provide for the columns to show on the plot 
    title (optional): the title for the boxplot
    
    OUTPUTS
    output_plot: a plt plot rendering of the data provided
    """
    # Create a temporary df
    temp_df = input_df[cols].sample(sample_size) 
    
    if normalize: 
        plot_df = (temp_df - temp_df.min()) / (temp_df.max() - temp_df.min())
    
        out_plot = plt.violinplot(plot_df, showmeans=True, showmedians=True, showextrema=False)
      
        #Get the median to plot with unique style point. 
        #Attribution: https://stackoverflow.com/questions/44253808/change-mean-indicator-in-violin-plot-to-a-circle
        mean_coords = [[plot.vertices[:,0].mean(),plot.vertices[0,1]] for plot in out_plot['cmeans'].get_paths()]
        mean_coords = np.array(mean_coords)
    
        #Create the plot(s)
        plt.scatter(mean_coords[:,0], mean_coords[:,1],s=35, c="k", marker="d")
        plt.title(title, fontsize=16)
        plt.xticks(ticks = range(1,len(cols)+1), labels=alt_names, rotation=35, ha='right')
        plt.xlabel("Feature(s) Studied", color='black')
        plt.ylabel("Min-Max Normalized Distribution", color='black')
        plt.grid(False)
        plt.gca().set_facecolor('white')
        plt.gca().spines['bottom'].set_color('black')
        plt.gca().spines['left'].set_color('black')
        out_plot['cmeans'].set_color('k')
        out_plot['cmedians'].set_color('g')
    
        #Add a legend to differentiate the mean and median
        mean_line = Line2D([0], [0], color='k', lw=2, label="Mean", marker='d', markerfacecolor='k', markersize=8)
        median_line = Line2D([0], [0], color='g', lw=2, label="Median") 
    
        #set the legend space dynamic for normalized or non-normalized scaling
        if normalize: 
            xmax = 1
            ymax = 1 
    
        #finish legend
        plt.legend(handles=[mean_line, median_line], loc='upper right', bbox_to_anchor=(1.3 * xmax, 0.95 * ymax))
        plt.show()
        
    else: 
        #If the plot data is not normalized then each plot should have it's own subplot in the total figure
        plot_df = temp_df
        ncols = 3
        nrows = (len(cols) + ncols - 1) // ncols
        num_plots = len(cols)
        adj_ncols = min(ncols, num_plots) 
        
        fig, axes = plt.subplots(nrows, adj_ncols, figsize=(5 * adj_ncols, 4 * nrows), layout="tight")
        axes = np.array(axes).reshape(-1)  
        
        # Loop through the total plots
        for i in range(len(axes)):
            if i < num_plots:
                # Select the column data for the plot
                column_data = plot_df[cols[i]]
    
                # Create boxplot for the column data
                out_plot = axes[i].violinplot(column_data, showmeans=True, showmedians=True)
    
                # Extract mean coordinates
                mean_coords = [[p.vertices[:,0].mean(), p.vertices[0,1]] for p in out_plot['cmeans'].get_paths()]
                mean_coords = np.array(mean_coords)
    
                # Plot the mean values as purple triangles
                axes[i].scatter(mean_coords[:, 0], mean_coords[:, 1], s=90, c="k", marker="d", zorder=5)
    
                # Set title and labels for the subplot
                axes[i].set_title(alt_names[i], fontsize=12)
                axes[i].set_xticklabels([]) 
                axes[i].grid(False)

                # Customize subplot appearance
                axes[i].set_facecolor("white")
                axes[i].spines["bottom"].set_color("black")
                axes[i].spines["left"].set_color("black")

                out_plot['cmeans'].set_color('k')
                out_plot['cmedians'].set_color('g')
                            
                # Add a legend for the mean and median
                mean_line = Line2D(
                    [0], [0], color='k', 
                    lw=2, label="Mean", marker='d', 
                    markerfacecolor='k', markersize=8)
                median_line = Line2D([0], [0], color='g', lw=2, label="Median") 
                
                #dynamically set the ylabels 
                if i % 3 == 0: 
                    axes[i].set_ylabel("Value", color="black")
                    axes[i].legend(handles=[mean_line, median_line], loc='upper left', fontsize=10)

                
                else: 
                    axes[i].set_ylabel(None)
    
            else:
                fig.delaxes(axes[i])  # Remove unused axes     
    

        
        plt.suptitle(title, fontsize=20)
        plt.tight_layout(h_pad=5, pad=2)
        plt.show()
    return
    

def box_plotter(
    input_df: pd.DataFrame, cols: list[str], 
    sample_size: int = 10000, normalize: bool = True, 
    colors: list[str] = None, alt_names: list[str] = None, 
    title: str = None) -> None:
    """This function creates a boxplot to describe the features found in a dataframe to 
    support exploratory data analysis.
    
    INPUTS
    input_df (required): the pandas dataframe object that the user wishes to generate boxplots for 
    cols (required): a list of column names that the user seeks to include in the boxplot 
    sample_size (optional): the number of samples to create the plot with
    normalize (optional): optional to normalize the plot so feature(s) are aligned to the same scale
    colors (optional): a list of specific colors the user wishes to provide for the boxplot 
    alt_names (optional): a list of names to provide for the columns to show on the plot 
    title (optional): the title for the boxplot
    
    OUTPUTS
    output_plot: a plt plot rendering of the data provided
    """
    temp_df = input_df[cols].sample(sample_size) 
    
    if normalize: 
        #Normalize the data and plot a single figure result
        plot_df = (temp_df - temp_df.min()) / (temp_df.max() - temp_df.min())

        out_plot = plt.boxplot(plot_df,showmeans=True, showfliers=False,  
                               meanprops={"marker": "o",
                                          "markerfacecolor": "none",  
                                          "markeredgecolor": "none",
                                         },
                              medianprops={"color": "green", 
                                           "linewidth": 1,
                                          })
        
        #Get the median to plot with unique style point. 
        #Attribution: https://stackoverflow.com/questions/44253808/change-mean-indicator-in-violin-plot-to-a-circle
        mean_coords = [[plot.get_xdata()[0], plot.get_ydata()[0]] for plot in out_plot['means']]    
        mean_coords = np.array(mean_coords)
    
        #Create the plot(s)
        plt.scatter(mean_coords[:,0], mean_coords[:,1],s=35, c="purple", marker="^")
        plt.title(title, fontsize=16)
        plt.xticks(ticks = range(1,len(cols)+1), labels=alt_names, rotation=35, ha='right')
        plt.xlabel("Feature(s) Studied", color='black')
        plt.ylabel("Min-Max Normalized Distribution", color='black')
        plt.grid(False)
        plt.gca().set_facecolor('white')
        plt.gca().spines['bottom'].set_color('black')
        plt.gca().spines['left'].set_color('black')
    
        #Add a legend to differentiate the mean and median
        mean_line = Line2D([0], [0], color='k', lw=0, label="Mean", marker='^', markerfacecolor='purple', markersize=8)
        median_line = Line2D([0], [0], color='g', lw=2, label="Median") 
    
        #set the legend space dynamic for normalized or non-normalized scaling
        xmax = 1
        ymax = 1 

        #finish legend & plot
        plt.legend(handles=[mean_line, median_line], loc='upper right', bbox_to_anchor=(1.3 * xmax, 0.95 * ymax))
        plt.show()

    else: 
        #If the plot data is not normalized then each plot should have it's own subplot in the total figure
        plot_df = temp_df
        ncols = 3
        nrows = (len(cols) + ncols - 1) // ncols
        num_plots = len(cols)
        adj_ncols = min(ncols, num_plots) 
        
        fig, axes = plt.subplots(nrows, adj_ncols, figsize=(5 * adj_ncols, 4 * nrows), layout="tight")
        axes = np.array(axes).reshape(-1)  
        
        # Loop through the total plots
        for i in range(len(axes)):
            if i < num_plots:
                # Select the column data for the plot
                column_data = plot_df[cols[i]]
    
                # Create boxplot for the column data
                out_plot = axes[i].boxplot(column_data, showmeans=True, showfliers=False,
                                           meanprops={
                                            "marker": "o", 
                                            "markerfacecolor": "none", 
                                            "markeredgecolor": "none"},
                                           medianprops={"color": "green", "linewidth": 2})
    
                # Extract mean coordinates
                mean_coords = [[plot.get_xdata()[0], plot.get_ydata()[0]] for plot in out_plot['means']]
                mean_coords = np.array(mean_coords)
    
                # Plot the mean values as purple triangles
                axes[i].scatter(mean_coords[:, 0], mean_coords[:, 1], s=90, c="purple", marker="^")
    
                # Set title and labels for the subplot
                axes[i].set_title(alt_names[i], fontsize=12)
                axes[i].set_xticklabels([]) 
                axes[i].grid(False)

                # Customize subplot appearance
                axes[i].set_facecolor("white")
                axes[i].spines["bottom"].set_color("black")
                axes[i].spines["left"].set_color("black")
    
                # Add a legend for the mean and median
                mean_line = Line2D(
                    [0], [0], color="k", lw=0, 
                    label="Mean", marker="^", markerfacecolor="purple", markersize=8
                    )
                median_line = Line2D([0], [0], color="g", lw=2, label="Median")

                #dynamically set the ylabels 
                if i % 3 == 0: 
                    axes[i].set_ylabel("Value", color="black")
                    axes[i].legend(handles=[mean_line, median_line], loc="upper left", fontsize=10)

                else: 
                    axes[i].set_ylabel(None)
    
            else:
                fig.delaxes(axes[i])  # Remove unused axes     

        plt.suptitle(title, fontsize=20)
        plt.tight_layout(h_pad=5, pad=2)

        #save plot:
        plt.show()
        return


def prep_data(input_df, condition): 
    """
    This function reads the EVSE charging datafile, and then performance appropriate cleaning operations based on
    the condition specified for EDA.
    
    INPUTS: 
    input_df: a pandas daaframe object of the loaded data
    condition: an integer variable specifying which condition to apply for cleaning. 
               0=Do Nothing, 1=Only remove nulls, 2=Remove nulls and exclude faulty charging events

    OUTPUTS:
    out_df: a pandas dataframe object with the desired cleaning operations applied
    """

    #Conditional Cleaning to view desired features and how they vary 
    if condition ==0: 
        out_df = input_df
        
    if condition == 1: 
        out_df = input_df.dropna()

    elif condition == 2: 
        #Remove impractical values 
        out_df = input_df.dropna()
        out_df = out_df[out_df['soc_charged'] >= 0]

        #Remove charging sessions with insufficient resolution (SOC did not change): 
        out_df[out_df["start_soc"] != out_df["end_soc"]]
        out_df[out_df["flag_id"] != 64]
        out_df[out_df["flag_id"] != 32]

    else: 
        raise ValueError("Input conditions must be 0, 1, or 2")

    return out_df 


def hist_plotter(
    input_ser: pd.Series, sample_size: int, normalize: bool = False, 
    exclude_outliers: bool = True, title: str = None, xlabel: str = None
    ) -> None: 
    """This function creates a histogram for the single series feature provided by the user
    INPUTS: 
    input_ser: a pandas series object with quantitative data
    sample_size: an interger specifying the number of samples to take from the data 
    normalize: a boolean specifying whether the data should be normalized or not
    exclude_outliers: a boolean specifying whether outliers should be included
    title: a string entry specifying the title for the output histogram plot
    xlabel: a string entry specifying the label for the X-axis data
    
    OUTPUTS: 
    None, a plot is generated but the function "returns" nothing
    """

    if sample_size < len(input_ser): 
        # Gets a sample if input_ser is long enough
        input_ser = input_ser.sample(sample_size)
    
    if normalize: 
        density_tag = True
        y_label = "Probability Density" 
        hist_label = "Normalized Histogram"
    else: 
        density_tag = False
        y_label ="Count in Bin"
        hist_label = "Histogram"


    if exclude_outliers:
        #Apply IQR analysis to determine outliers
        Q1 = np.percentile(input_ser, 25)
        Q3 = np.percentile(input_ser, 75) 
        IQR = Q3 - Q1 
        LB = Q1 - 1.5 * IQR 
        UB = Q3 + 1.5 * IQR

        input_ser = input_ser[(input_ser >= LB) & (input_ser <= UB)]

    # Apply Freedman-Diaconis' Rule to find Appropriate bin-qty: 
    h = 2 * IQR * (len(input_ser))**(-1/3)
    num_bins = int(round((input_ser.max() - input_ser.min()) / h, 0))
    
    # Add PDF: 
    kde = gaussian_kde(input_ser)
    x_values = np.linspace(input_ser.min(), input_ser.max(), 1000)

    # Calculate CDF: 
    cdf_values = np.cumsum(kde(x_values)) * (x_values[1] - x_values[0])

    
    # Create the histogram and PDF on a dual-axis plot:
    fig, ax1 = plt.subplots(figsize=(8,6))

    # Plot histogram and pdf on the first axis (ax1)
    ax1.hist(input_ser, bins=num_bins, density=density_tag, color='firebrick', label=hist_label, histtype="bar")
    ax1.plot(x_values, kde(x_values), '--', color='black', lw=2, label="Probability density")

    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(y_label, color='firebrick')
    ax1.tick_params(axis='y', colors='firebrick')
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    
    # Create a second axis for the PDF
    ax2 = ax1.twinx()
    ax2.plot(x_values, cdf_values, '-', color='green', lw=2, label='Cumulative Density')
    ax2.set_ylabel("Cumulative Density Function", color='green')    
    ax2.tick_params(axis='y', colors='green')
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))


    # Add a text for the 50% finish time
    y_max = ax2.get_ylim()[1]  # Top of the Y-axis
    time_50 = np.interp(0.5, cdf_values, x_values)
    ax2.text(x_values.max(), y_max*0.8, f"50% Finish Charging\nin {round(time_50,1)} min.", 
             ha='right', va='top', color='green', fontsize=12)
    
    # Set legend
    handles, labels = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    handles.extend(handles2)
    labels.extend(labels2)    

    # Set title & legend
    plt.title(title,fontweight='bold', fontsize=14)
    plt.legend(handles, labels, loc='center right')
    plt.show()


def create_desired_plots(input_df: pd.DataFrame) -> None: 
    """
    This function creates 4 figures that are useful for exploratory data analysis of the EV public charging dataset
    Fig 1. Shows the normalized distribution of charging metrics, including faulty charging events
    Fig 2. Shows the normalized distribution of charging metrics, excluding faulty charging events
    Fig 3. is the box and whisker plot of the same data as fig 2, made by calling box_plotter()
    Fig 4. is the non-normalized version of fig 2, separating each feature into a subplot

    Note: box_plotter() could be used to get a similar figure to figure 3, if desired
    """
    

    # Specify list of features desired for plotting
    features = ["total_duration", "charge_duration", "energy_kwh", "start_soc", "end_soc", "soc_charged"]
    names = [
        "Total Occupied Time (min)", "Charge Duration (min)", 
        "Energy Charged (kWh)", "Start SOC (%)", "End SOC (%)", "SOC Charged (%)"
        ]
    plot_titles = ["Fig 1. Normalized Charging Metric Distributions: Including Anomalies", 
                   "Fig 2. Normalized Charging Metric Distributions: Excluding Anomalies", 
                   "Fig 3. Normalized Boxplot of Charging Metrics: Excluding Anomalies",
                   "Fig 4. Detailed Charging Metric Distributions: Non-Normalized and Excluding Anomalies",
                  ]

    # Prep the dataframes
    df_w_faults = prep_data(input_df, 1)
    df_no_faults = prep_data(input_df, 2)

    # Plot the figures:
    violin_plotter(df_w_faults, cols=features, normalize=True, colors=None, alt_names=names, title=plot_titles[0])
    violin_plotter(df_no_faults, cols=features, normalize=True, colors=None, alt_names=names, title=plot_titles[1])
    box_plotter(df_no_faults, cols=features, normalize=True, colors=None, alt_names=names, title=plot_titles[2])
    violin_plotter(
        df_no_faults, cols=features, 
        normalize=False, colors=None, 
        alt_names=names, title=plot_titles[3])     
    hist_plotter(df_no_faults["charge_duration"]*60, sample_size = 10000, normalize=True, 
                 xlabel="Charge Duration - minutes", 
                 title="Fig 5. Total charge duration (minutes) Distribution"
                 )


def show_EDA1_results(filepath: str) -> None:
    """This function runs the desired process to create the desired violin and histogram plots"""
    
    # Read the data
    df = pd.read_csv(filepath) #Read file
    df["soc_charged"] = df["end_soc"] - df["start_soc"]  #calculate the SOC charged during event
    
    # Create violin and box plots
    create_desired_plots(df) #create desired plots

#Execution: 
if __name__ == "__main__": 
    show_EDA1_results()

