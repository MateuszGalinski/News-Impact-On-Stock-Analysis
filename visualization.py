import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def visualize_stock_data(csv_path):
    """
    Visualizes stock OHLCV data from a CSV file with format:
    <TICKER>,<PER>,<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>,<OPENINT>
    """

    # Load data
    df = pd.read_csv(csv_path)

    # Create datetime column
    df["DATETIME"] = pd.to_datetime(
        df["<DATE>"].astype(str) + df["<TIME>"].astype(str).str.zfill(6),
        format="%Y%m%d%H%M%S"
    )

    print("DATE")
    print(min(df["DATETIME"]))

    df = df.sort_values("DATETIME")
    df.reset_index(drop=True, inplace=True)

    # Create figure
    fig, (ax_price, ax_vol) = plt.subplots(2, 1, sharex=True, figsize=(12, 8))

    # Plot candlesticks
    width = 0.6

    for i in range(len(df)):
        open_price = df.loc[i, "<OPEN>"]
        close_price = df.loc[i, "<CLOSE>"]
        high_price = df.loc[i, "<HIGH>"]
        low_price = df.loc[i, "<LOW>"]

        color = "green" if close_price >= open_price else "red"

        # Wick
        ax_price.plot([i, i], [low_price, high_price])

        # Body
        rect = Rectangle(
            (i - width / 2, min(open_price, close_price)),
            width,
            abs(close_price - open_price)
        )
        ax_price.add_patch(rect)

    ax_price.set_ylabel("Price")
    ax_price.set_title(f"{df.loc[0, '<TICKER>']} Candlestick Chart")

    # Plot volume
    ax_vol.bar(range(len(df)), df["<VOL>"])
    ax_vol.set_ylabel("Volume")

    # Show fewer x-axis labels
    step = max(len(df) // 10, 1)   # show about 10 labels max
    xticks = range(0, len(df), step)

    ax_vol.set_xticks(xticks)
    ax_vol.set_xticklabels(
        df["DATETIME"].iloc[::step].dt.strftime("%H:%M"),
        rotation=45
    )

    plt.tight_layout()
    plt.show()

def main():
    # visualize_stock_data('data\\5 min\\pl\\wse stocks\\cdr.txt')
    # visualize_stock_data('data_stooq\\daily\\pl\\wse stocks\\cdr.txt')
    # visualize_stock_data('data_stooq\\daily\\us\\nasdaq stocks\\1\\aapl.us.txt')
    visualize_stock_data('data_stooq\\daily\\us\\nasdaq stocks\\2\\nvda.us.txt')
    # visualize_stock_data('data\\5 min\\pl\\wse stocks\\cdr.txt')

if __name__ == "__main__":
    main()