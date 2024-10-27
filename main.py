import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression

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

def price_by_month_and_amount(df:pd.DataFrame, type:str = "Beef"):
    data = df[df["cat.Aeng"] == type].copy()
    data["per_unit_price"] = data["total.YENsales"] / data["num.sales"]
    data["date"] = pd.to_datetime(data[["year", "month"]].assign(day=1))

    monthly_data = data.groupby(data["date"].dt.to_period("M")).agg({
        "per_unit_price": "mean",
        "num.sales": "sum"
    }).rename(columns={"num.sales": "total_units_sold"})

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Per-Unit Price
    monthly_data["per_unit_price"].plot(kind="bar", color="skyblue", ax=ax1, position=1, width=0.4)
    ax1.set_ylabel("Per-Unit Price (YEN)", color="skyblue")
    ax1.set_xlabel("Month")
    ax1.set_title(f"Average Per-Unit Price and Total Units Sold of {type} by Month")

    # Total Units Sold
    ax2 = ax1.twinx()
    monthly_data["total_units_sold"].plot(kind="bar", color="orange", ax=ax2, alpha=0.6)
    ax2.set_ylabel("Total Units Sold", color="orange")

    plt.grid(True)
    plt.show()

def r_correlation(df:pd.DataFrame, type:str = "Beef"):
    # Filter dataset to include only rows where cat.Aeng is "Beef"
    data = df[df["cat.Aeng"] == type].copy()

    # Calculate the per-unit price
    data["per_unit_price"] = data["total.YENsales"] / data["num.sales"]

    # Drop rows with NaN values in "per_unit_price" or "num.sales"
    data.dropna(subset=["per_unit_price", "num.sales"], inplace=True)

    # Check if data exists after dropping NaN values
    if not data.empty:
        # Calculate correlation
        correlation, _ = pearsonr(data["per_unit_price"], data["num.sales"])
        print(f"Pearson correlation coefficient: {correlation:.2f}")

        # Perform linear regression
        X = data["per_unit_price"].values.reshape(-1, 1)
        y = data["num.sales"].values
        reg = LinearRegression().fit(X, y)
        y_pred = reg.predict(X)

        # Plot the scatter plot with regression line
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x="per_unit_price", y="num.sales", data=data, alpha=0.6)
        plt.plot(data["per_unit_price"], y_pred, color="blue", linewidth=2, label="Regression Line")
        plt.title(f"Correlation between Per-Unit Price and Total Units Sold for {type} (r = {correlation:.2f})")
        plt.xlabel("Per-Unit Price (YEN)")
        plt.ylabel("Total Units Sold")
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        print("No data available after removing NaN values.")


if __name__ == "__main__":
    
    print("\033[93mStarting data processing...\033[0m", flush=True)

    df = load_pkl_data()

    price_by_month_and_amount(df, "Pork")
    r_correlation(df, "Pork")

    # print(df.columns)
    # print(df["cat.Aeng"])

    # i = 0
    # for row in df["cat.Aeng"]:
    #     if row == "Beef":
    #         i += 1
    # print(i)
    
    # print(df["cat.Aeng"].value_counts())
            
    


    print("\033[93mEnding process\033[0m", flush=True)


