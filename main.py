import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

def load_csv_data(PATH:str = "POS2024.csv") -> pd.DataFrame:
    first = dt.datetime.now()
    df = pd.read_csv(PATH, encoding="SJIS")

    print("csv loaded in: ", dt.datetime.now()-first)
    return df

def load_pkl_data(PATH:str = "POS2024.pkl") -> pd.DataFrame:
    first = dt.datetime.now()
    df = pd.read_pickle(PATH)

    print("PKL loaded in: ", dt.datetime.now()-first, "\n")

    return df

def average_beef_price_by_month(df:pd.DataFrame):
    # Filter dataset to include only rows where cat.Aeng is "Beef"
    beef_data = df[df["cat.Aeng"] == "Beef"]

    # Add a new column for per-unit price (total sales in yen / number of sales)
    beef_data["per_unit_price"] = beef_data["total.YENsales"] / beef_data["num.sales"]

    # Convert year and month to a datetime format for easier plotting
    beef_data["date"] = pd.to_datetime(beef_data[["year", "month"]].assign(day=1))

    # Group by month and calculate the average per-unit price
    monthly_avg_price = beef_data.groupby(beef_data["date"].dt.to_period("M"))["per_unit_price"].mean()

    # Plot the data
    plt.figure(figsize=(10, 6))
    monthly_avg_price.plot(kind="bar")
    plt.title("Average Per-Unit Price of Beef by Month")
    plt.xlabel("Month")
    plt.ylabel("Per-Unit Price (Yen)")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    
    print("\033[93mStarting data processing...\033[0m")

    df = load_pkl_data()

    average_beef_price_by_month(df)

    # print(df.columns)
    # print(df["cat.Aeng"])

    # i = 0
    # for row in df["cat.Aeng"]:
    #     if row == "Beef":
    #         i += 1
    # print(i)
    
    # print(df["cat.Aeng"].value_counts())
            
    


    print("\033[93mEnding process\033[0m")


