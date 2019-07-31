import argparse
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd

TITLE_FORMAT = "Train & Validation {} vs Epoch"

def draw_plot(path, **kwargs):
    draw_loss = kwargs.get("draw_loss", True)
    draw_psnr = kwargs.get("draw_psnr", True)

    try:
        df = pd.read_csv(path)
    except FileNotFoundError as e:
        print(e)
        exit(1)

    epochs = list(range(1, len(df.index) + 1))
    
    if draw_loss:
        plot_data(df["train_loss"], df["val_loss"], epochs, ylabel="Loss")

    if draw_psnr:
        plot_data(df["train_psnr"], df["val_psnr"], epochs, ylabel="PSNR")

def plot_data(train_data, val_data, epochs, ylabel):
    plt.plot(epochs, train_data, "o-")
    plt.plot(epochs, val_data, "o-")
    plt.xticks(epochs)
    plt.legend(["Train", "Validation"], loc='upper left')
    plt.title(TITLE_FORMAT.format(ylabel))
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Plotting Train & Validation Data', add_help=False)
    parser.add_argument("-h", "--help", help="Help message", action="store_true")
    parser.add_argument("--path", help="Read CSV file in path", default="", type=str)
    parser.add_argument("--loss", help="Plot Loss vs Epoch", default=False, action="store_true")
    parser.add_argument("--psnr", help="Plot PSNR vs Epoch", default=False, action="store_true")
    args = parser.parse_args()
    if args.help:
        parser.print_help()
    else:
        draw_plot(args.path, draw_loss=args.loss, draw_psnr=args.psnr)


if __name__ == "__main__":
    main()