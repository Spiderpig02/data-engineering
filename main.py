import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
import inquirer

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

def r_correlation(df:pd.DataFrame, type:str = "Beef", year:int = None, month:int = None):
    data = df[df["cat.Aeng"] == type].copy()
    
    if year is not None and month is not None:
        data = data[(data["year"] < year) | ((data["year"] == year) & (data["month"] <= month))]

    data["per_unit_price"] = data["total.YENsales"] / data["num.sales"]

    data.dropna(subset=["per_unit_price", "num.sales"], inplace=True)


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

def predict_future_price(df:pd.DataFrame, type:str = "Beef", months:int = 6):
    data = df[df["cat.Aeng"] == type].copy()
    data["per_unit_price"] = data["total.YENsales"] / data["num.sales"]
    data["date"] = pd.to_datetime(data[["year", "month"]].assign(day=1))
    data = data.sort_values("date")

    # Drop rows with NaN values in 'per_unit_price'
    data = data.dropna(subset=["per_unit_price"])

    # Prepare the data for linear regression
    data["month_num"] = np.arange(len(data))
    X = data["month_num"].values.reshape(-1, 1)
    y = data["per_unit_price"].values

    # Train the linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Predict future prices
    future_months = np.arange(len(data), len(data) + months).reshape(-1, 1)
    future_prices = model.predict(future_months)

    # Plot the historical and predicted prices
    plt.figure(figsize=(10, 6))
    plt.plot(data["date"], data["per_unit_price"], label="Historical Prices")
    future_dates = pd.date_range(start=data["date"].iloc[-1], periods=months + 1, freq='ME')[1:]
    plt.plot(future_dates, future_prices, label="Predicted Prices", linestyle='--')
    plt.title(f'Price Prediction for {type} for the Next {months} Months')
    plt.xlabel('Date')
    plt.ylabel('Per-Unit Price (YEN)')
    plt.legend()
    plt.grid(True)
    plt.show()



if __name__ == "__main__":
    
    print("\033[93mStarting data processing...\033[0m", flush=True)

    df = load_pkl_data()
    key_pair = df["cat.Aeng"].unique()

    questions_topic = [
    inquirer.List(
        "Topic",
        message="Choose a Topic to analyze:",
        choices=[*key_pair],
    ),
]

    answer_topic = inquirer.prompt(questions_topic)["Topic"]

    # Prompt user to choose functions to call
    questions_functions = [
        inquirer.Checkbox(
            "Functions",
            message="Select functions to call:",
            choices=[
                ("Price by Month and Amount", "price_by_month_and_amount"),
                ("Correlation Analysis", "r_correlation"),
                ("Predict Future Price", "predict_future_price"),
            ],
        ),
    ]

    answer_functions = inquirer.prompt(questions_functions)["Functions"]
    print(f"\033[93mAnalyzing data for {answer_topic}...\033[0m", flush=True)

    try: 
        # Call the selected functions
        if "price_by_month_and_amount" in answer_functions:
            price_by_month_and_amount(df, answer_topic)

        if "r_correlation" in answer_functions:
            r_correlation(df, answer_topic)

        if "predict_future_price" in answer_functions:
                questions_months = [
                    inquirer.Text(
                        "Months",
                        message="Enter the number of months to predict",
                        validate=lambda _, x: x.isdigit() and int(x) > 0,
                    ),
                ]
                answer_months = int(inquirer.prompt(questions_months)["Months"])
                predict_future_price(df, answer_topic, answer_months)
    except Exception as e:
        print("\033[91mThe chosen data does not support generating the selected content.\nPlease choose another topic.\033[0m", flush=True)

    
    # print(df["cat.Aeng"].value_counts())
            
    
    print("\033[93mEnding process\033[0m", flush=True)


