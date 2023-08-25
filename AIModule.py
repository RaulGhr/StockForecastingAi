import pandas as pd
import datetime
from neuralforecast import NeuralForecast
from neuralforecast.models import LSTM
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", None)
intData = pd.read_csv("dateInterogate.csv")
intData["ds"] = pd.to_datetime(intData["ds"], format="%Y-%m-%dT%H:%M:%S.%f+0000")
print(intData.loc[0, "ds"])
dt = datetime.datetime(2023, 2, 1)
firstTrainingSet = intData[intData["ds"] < dt]
# print(firstTrainingSet)
horizon = 30


def createModel():
    model = [
        LSTM(
            h=horizon,
            max_steps=100,  # Number of steps to train
            scaler_type="standard",  # Type of scaler to normalize data
            encoder_hidden_size=64,  # Defines the size of the hidden state of the LSTM
            decoder_hidden_size=64,
            hist_exog_list=[
                "FedInterest",
                "High",
                "Low",
                "Open",
                "SMA_1M",
                "SMA_1W",
                "SMA_3M",
                "SMA_V_1M",
                "SMA_V_1W",
                "SMA_V_3M",
                "Volume",
                "Week Number",
                "Week Number Average",
            ],
        )
    ]
    nf = NeuralForecast(models=model, freq="B")
    nf.fit(df=firstTrainingSet)
    nf.save(path="./testModel", model_index=None, overwrite=True, save_dataset=True)


def trainModel():
    nf2 = NeuralForecast.load(path="./testModel")

    nf2.fit(df=firstTrainingSet)
    nf2.save(path="./testModel", model_index=None, overwrite=True, save_dataset=True)


def predict():
    nf = NeuralForecast.load(path="./testModel")
    start_of_feb = datetime.datetime(2023, 2, 1, 0, 0, 0)
    firstCVSet = intData[intData["ds"].dt.month == start_of_feb.month]
    firstCVSet = firstCVSet[firstCVSet["ds"].dt.year == start_of_feb.year]
    print(firstCVSet)
    Y_hat_df = nf.predict()
    print(Y_hat_df)

    # fig, ax = plt.subplots(1, 1, figsize=(20, 7))
    # plot_df = pd.concat([firstCVSet, Y_hat_df]).set_index('ds')  # Concatenate the train and forecast dataframes
    # plot_df[['y', 'LSTM']].plot(ax=ax, linewidth=2)
    #
    # ax.set_title('Results', fontsize=22)
    # ax.set_ylabel('Monthly Passengers', fontsize=20)
    # ax.set_xlabel('Timestamp [t]', fontsize=20)
    # ax.legend(prop={'size': 15})
    # ax.grid()
    # Creează un plot
    # Afișează datele
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(firstCVSet["ds"], firstCVSet["y"], label="firstCVSet")
    ax.plot(Y_hat_df["ds"], Y_hat_df["LSTM"], label="Y_hat_df (LSTM)")

    ax.set_title("Comparison of firstCVSet and Y_hat_df (LSTM)", fontsize=16)
    ax.set_xlabel("Date", fontsize=14)
    ax.set_ylabel("Values", fontsize=14)

    ax.legend()
    ax.grid()
    plt.show()


if __name__ == "__main__":
    # createModel()
    # for i in range(20):
    #     print(i)
    trainModel()
    predict()
    exit()
