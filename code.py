import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import animation

path = r"C:\Users\Taixen\PycharmProjects\PythonProject3\AirQuality_Monitor_sample.csv"  # File path


def load_data(path):
    # TODO: Load CSV with pandas
    # TODO: Convert Date column to datetime
    df = pd.read_csv(path)  # Load data from a CSV file into Pandas Dataframe for analysis.
    df['Date'] = pd.to_datetime(df[
                                    'Date'])  # This function converts the date column to datetime format to be able to perform operations related to date

    return df  # Return the dataframe (the date)


def inspect(df):
    # TODO: Print head, info, basic stats

    print("Head: ")
    print(df.head(0))  # Prints no data rows, only the heading

    print("Info: ")
    df.info()  # Prints the info related to data frame

    print("Basic Stats: ")
    print(df.describe())  # Prints the statistics of the data frame


def clean_data(df):
    # TODO: Handle missing values using median or drop
    numeric_columns = ['PM2.5', 'PM10', 'NO2']
    for column in numeric_columns:
        if column not in df.columns:
            df[column].fillna(df[column].median(), inplace=True)

    df.dropna(subset=['PM2.5', 'Date'], inplace=True)
    return df


def compute_summary(df):
    # TODO: Compute per-station summary (means)
    if 'StationID' in df.columns:
        summary = df.groupby('StationID').agg({  # Group each data by its StationID
            'PM2.5': ['mean', 'std', 'min', 'max'],  # Find the mean, standard deviation, minimum and maximum for PM2.5
            'PM10': 'mean',  # FInd the mean for PM10
            'NO2': 'mean'  # Find the mean for NO2
        }).round(2)  # Rounds to two decimal points
        return summary  # Return a table containing the requested data


def find_anomalies(df):
    # TODO: Flag rows with PM2.5 > 25 μg/m³
    anomalies = df[df['PM2.5'] > 25]  # Makes a dataframe out of the PM2.5 data that is over 25
    print(
        f"Found {len(anomalies)} anomaly records (PM2.5>25)")  # Prints the number of anomalities, which is also the number of elements in the new anomalies dataframe
    return anomalies  # Returns the dataframe


def create_plots(df):
    # TODO: Create and save:

    # 1) pm25_timeseries.png
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x="Date", y="PM2.5",
                 hue="StationID")  # Line graph, plot the PM2.5 vs the Date from the data file
    plt.title('PM2.5 Over Time')  # Title of the graph
    plt.savefig(r"C:\Users\Taixen\PycharmProjects\PythonProject3\pm25_timeseries.png")  # Save the graph as a png file on the computer
    plt.close()  # Close the file

    # 2) hist_pm25.png
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x="PM2.5", stat='frequency',
                 hue="StationID")  # Plot a histograph, frequency vs PM2.5 from the data file
    plt.title("Distribution of PM2.5 values")  # Title of the graph
    plt.savefig(r"C:\Users\Taixen\PycharmProjects\PythonProject3\hist_pm25.png")  # Save the graph as a png file on the computer
    plt.close()  # Close the file

    # 3) box_pm25.png
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="StationID", y="PM2.5")  # Plot a box plot PM2.5 vs Station ID from the datafile
    plt.title("PM2.5 Distribution by StationID")  # Title of the graph
    plt.savefig(r"C:\Users\Taixen\PycharmProjects\PythonProject3\box_pm25.png")  # Saves the graph as a png file on the computer
    plt.close()  # Close the file

    # 4) scatter_pm10_no2.png
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='PM10', y='NO2', hue='StationID')  # Plots a NO2 vs PM10 scatterplot
    plt.title("PM10 vs NO2 by Station")  # Title of the graph
    plt.savefig(r"C:\Users\Taixen\PycharmProjects\PythonProject3\scatter_pm10_no2.png")  # Saves the graph as a png file on the computer
    plt.close()  # Close the file


def make_animation(df):
    # TODO: Implement animation showing PM2.5 progression over time
    daily = df.groupby("Date")["PM2.5"].mean().reset_index()
    dates = daily["Date"].values
    scores = daily["PM2.5"].values

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_xlim(min(dates), max(dates))
    ax.set_ylim(min(scores) - 1, max(scores) + 2)
    ax.set_xlabel("Date")
    ax.set_ylabel("PM2.5")
    line, = ax.plot([], [], '-')

    print(line)

    def animate(i):
        x = dates[:i + 1]
        y = scores[:i + 1]
        line.set_data(x, y)
        ax.set_title(f"Day {i + 1}")
        return line,

    anim = animation.FuncAnimation(
        fig, animate, frames=len(dates), interval=600
    )
    plt.show()
    anim.save(r"C:\Users\Taixen\PycharmProjects\PythonProject3\PM2.5_animation.gif")
    plt.close()


def export_results(df):
    # TODO: Save anomalies.csv
    # TODO: Save summary_by_station.csv
    find_anomalies(df).to_csv(
        r"C:\Users\Taixen\PycharmProjects\PythonProject3\anomalies.csv")  # Saves the find_anomalies(df) function as a csv file on the computer...
    compute_summary(df).to_csv(
        r"C:\Users\Taixen\PycharmProjects\PythonProject3\summary_by_station.csv")  # Saves the compute_summary(df) function as a csv file on the computer...


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
