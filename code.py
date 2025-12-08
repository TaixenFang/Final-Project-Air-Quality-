# Required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import animation

path = r"C:\Users\Taixen\PycharmProjects\PythonProject3\AirQuality_Monitor_sample.csv"  # File path


def load_data(path):
    # Load data from a CSV file into Pandas Dataframe for analysis
    df = pd.read_csv(path)
    # This function converts the date column to datetime format to be able to perform operations related to date
    df['Date'] = pd.to_datetime(df['Date'])

    # Return the dataframe (the date)
    return df


def inspect(df):

    # Prints no data rows, only the heading
    print("Head: ")
    print(df.head(0))

    # Prints the info related to data frame
    print("Info: ")
    df.info()

    # Prints the statistics of the data frame
    print("Basic Stats: ")
    print(df.describe())


def clean_data(df):
    # Defining the columns of pollution data that need cleaning
    numeric_columns = ['PM2.5', 'PM10', 'NO2']
    # For loop checks if column exists in the DataFrame
    for column in numeric_columns:
        if column in df.columns:
            # Fills any missing values with median values
            df[column].fillna(df[column].median(), inplace=True)
    # Removing rows where both the Date and PM2.5 levels are missing
    df.dropna(subset=['PM2.5', 'Date'], inplace=True)
    return df


def compute_summary(df):
    if 'StationID' in df.columns:
        # Group each data by its StationID
        summary = df.groupby('StationID').agg({
            # Find the mean, standard deviation, minimum and maximum for the PM2.5 pollutant
            'PM2.5': ['mean', 'std', 'min', 'max'],
            # FInd the mean for PM10
            'PM10': 'mean',
            # Find the mean for NO2
            'NO2': 'mean'
            # Rounds to two decimal points
        }).round(2)
        # Return a table containing the requested data
        return summary


def find_anomalies(df):
    # Creates a dataframe out of the PM2.5 data that is over 25
    anomalies = df[df['PM2.5'] > 25]
    # Prints the number of anomalies, which is also the number of elements in the new anomalies dataframe
    print(
        f"Found {len(anomalies)} anomaly records (PM2.5>25)")
    # Returns a DataFrame containing anomalies
    return anomalies


def create_plots(df):
    # 1) PM2.5 Levels vs Time linegraph

    # Set graph size
    plt.figure(figsize=(12, 6))
    # Using Seaborn, set DataFrame as data used for this graph, use Date & PM2.5 column values, use 'hue' to identify each linegraph
    sns.lineplot(data=df, x="Date", y="PM2.5",hue="StationID")
    # Title of the graph
    plt.title('PM2.5 Over Time')
    # Save the graph as a png image file
    plt.savefig(r"C:\Users\Taixen\PycharmProjects\PythonProject3\pm25_timeseries.png")
    # Close the file
    plt.close()

    # 2) PM2.5 pollutant histogram

    # Set graph size
    plt.figure(figsize=(10, 6))
    # Using Seaborn, set DataFrame as data used for this graph, use only PM2.5 from the csv file, add a frequency stat for Y-Axis
    sns.histplot(data=df, x="PM2.5", stat='frequency',hue="StationID")
    # Title of the graph
    plt.title("Distribution of PM2.5 values")
    # Save the graph as a png image file
    plt.savefig(r"C:\Users\Taixen\PycharmProjects\PythonProject3\hist_pm25.png")
    # Close the file
    plt.close()

    # 3) PM2.5 boxplot by station

    # Set graph size
    plt.figure(figsize=(10, 6))
    # Using Seaborn, set DataFrame as data used for this graph, use StationID & PM2.5 columns to compare which stations recorded more or less pollutant levels
    sns.boxplot(data=df, x="StationID", y="PM2.5")
    # Title of the graph
    plt.title("PM2.5 Distribution by StationID")
    # Saves the graph as a png image file
    plt.savefig(r"C:\Users\Taixen\PycharmProjects\PythonProject3\box_pm25.png")
    # Close the file
    plt.close()

    # 4) scatter_pm10_no2.png

    # Set graph size
    plt.figure(figsize=(10, 6))
    # Using Seaborn, set DataFrame as data used for this graph, use PM10 & NO2 columns to compare which pollutant is more present
    sns.scatterplot(data=df, x='PM10', y='NO2', hue='StationID')
    # Title of the graph
    plt.title("PM10 vs NO2 by Station")
    # Saves the graph as a png image file
    plt.savefig(r"C:\Users\Taixen\PycharmProjects\PythonProject3\scatter_pm10_no2.png")
    # Close the file
    plt.close()


def make_animation(df):
    # Animation function for the first graph - PM2.5 vs time

    # Group data by date and calculating the daily average PM2.5 concentration
    daily = df.groupby("Date")["PM2.5"].mean().reset_index()
    # Extracting date values for X-Axis
    dates = daily["Date"].values
    # Extracting PM2.5 values for Y-Axis
    pollutant = daily["PM2.5"].values
    # Set figure size
    fig, ax = plt.subplots(figsize=(10, 4))
    # Set X-Axis span from earliest to latest recorded date in the DataFrame
    ax.set_xlim(min(dates), max(dates))
    # Add margin to the Y-Axis (-1 from min and +2 from max) so that the graph doesn't touch the edges of the plot
    ax.set_ylim(min(pollutant) - 1, max(pollutant) + 2)
    # Axis titles
    ax.set_xlabel("Date")
    ax.set_ylabel("PM2.5")
    # Create an empty line object that will be animated
    line, = ax.plot([], [], '-')

    def animate(i): # 'i' is the frame number
    # Set X coordinates from the start to current frame
        x = dates[:i + 1]
    # Set Y coordinates for corresponding PM2.5 values
        y = pollutant[:i + 1]
    # This animates the line graph progressively as the days go by
        line.set_data(x, y)
    # Title updates to show the day count
        ax.set_title(f"Day {i + 1}")
    # Return the line object as a single-element tuple
        return line,

    # Animating the figure described above, calling the function for every frame, set number of frames to number of days (1 frame per day), interval of 600 milliseconds between each frame
    anim = animation.FuncAnimation(
        fig, animate, frames=len(dates), interval=600
    )
    # Display the animation
    plt.show()
    # Saving the animation as a gif
    anim.save(r"C:\Users\Taixen\PycharmProjects\PythonProject3\PM2.5_animation.gif")
    # Closing the animation
    plt.close()


def export_results(df):
    # Calling the find_anomalies(df) function to export the created DataFrame as a csv file
    find_anomalies(df).to_csv(
        r"C:\Users\Taixen\PycharmProjects\PythonProject3\anomalies.csv")
    # Calling the compute_summary(df) function to export the created DataFrame as a csv file
    compute_summary(df).to_csv(
        r"C:\Users\Taixen\PycharmProjects\PythonProject3\summary_by_station.csv")

if __name__ == "__main__":
    df = load_data(path)
    inspect(df)
    df = clean_data(df)
    compute_summary(df)
    find_anomalies(df)
    create_plots(df)
    make_animation(df)
    export_results(df)
    print("All tasks completed — continue refining for submission!")
