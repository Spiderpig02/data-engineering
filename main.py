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
    beef_data = df[df["cat.Aeng"] == "Beef"].copy()

    # Add a new column for per-unit price (total sales in yen / number of sales)
    beef_data["per_unit_price"] = beef_data["total.YENsales"] / beef_data["num.sales"]

    # Convert year and month to a datetime format for easier plotting
    beef_data["date"] = pd.to_datetime(beef_data[["year", "month"]].assign(day=1))

    # Group by month to calculate the average per-unit price and total units sold
    monthly_data = beef_data.groupby(beef_data["date"].dt.to_period("M")).agg({
        "per_unit_price": "mean",
        "num.sales": "sum"
    }).rename(columns={"num.sales": "total_units_sold"})

    # Plot the data
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot the per-unit price as bars
    monthly_data["per_unit_price"].plot(kind="bar", color="skyblue", ax=ax1, position=1, width=0.4)
    ax1.set_ylabel("Per-Unit Price (YEN)", color="skyblue")
    ax1.set_xlabel("Month")
    ax1.set_title("Average Per-Unit Price and Total Units Sold of Beef by Month")

    # Create a second y-axis for the total units sold
    ax2 = ax1.twinx()
    monthly_data["total_units_sold"].plot(kind="bar", color="orange", ax=ax2, alpha=0.6)
    ax2.set_ylabel("Total Units Sold", color="orange")

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


