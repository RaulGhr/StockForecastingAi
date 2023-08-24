import numpy as np

import yfinance as yf
import talib
import datetime
import pandas as pd
import polars as pl
from full_fred.fred import Fred
import os, time
from influxdb_client import InfluxDBClient
from influxdb_client.client.write_api import SYNCHRONOUS
from google.cloud import secretmanager

client = secretmanager.SecretManagerServiceClient()
secret_name = "projects/proiect-vara/secrets/influx_key/versions/latest"
response = client.access_secret_version(request={"name": secret_name})

influxDB_token = response.payload.data.decode("UTF-8")
org = "PriceForecasting"
host = "https://us-central1-1.gcp.cloud2.influxdata.com"
database = "TrainingData"
client = InfluxDBClient(url=host, token=influxDB_token, org=org)


def colectData():
    print("!!INCEPE COLECTAREA DATELOR!!")
    # PRET STOCK
    crsr = yf.Ticker("CRSR")

    hist = crsr.history(period="max")

    hist = hist.drop('Dividends', axis=1).drop('Stock Splits', axis=1)

    hist = hist.reset_index()

    hist = pl.DataFrame(hist)
    hist = hist.filter((hist['Open'] > 8) & (hist['Open'] < 100))
    hist = hist.filter((hist['High'] > 8) & (hist['High'] < 100))
    hist = hist.filter((hist['Low'] > 8) & (hist['Low'] < 100))
    hist = hist.filter((hist['Close'] > 8) & (hist['Close'] < 100))
    hist = hist.with_columns(pl.col('Date').cast(pl.Date))

    # INDICI SAPTAMANA
    indici_sapt = hist['Date'].to_numpy()
    data_ini = indici_sapt[0] - 2
    indici_sapt = (indici_sapt - data_ini).astype('timedelta64[D]').astype(np.int64) // 7
    indici_sapt = pl.DataFrame({'Week Number': indici_sapt})
    hist = hist.hstack(indici_sapt)

    # MEDIE PE SAPTAMANA
    nr_sapt = hist['Week Number'].max() + 1
    suma = [0] * nr_sapt
    frecv = [0] * nr_sapt
    week_nr_avr = []

    for row in hist.iter_rows(named=True):
        indice = row['Week Number']
        frecv[indice] += 1
        suma[indice] += row['Close']

    for i in range(nr_sapt):
        medie = suma[i] / frecv[i]
        for i in range(frecv[i]):
            week_nr_avr.append(medie)

    week_nr_avr = pl.DataFrame({'Week Number Average': week_nr_avr})
    hist = hist.hstack(week_nr_avr)

    # INDICI
    close_prices = hist['Close']
    volume = hist['Volume']

    SMA_1W = talib.SMA(close_prices.to_numpy(), timeperiod=5)
    SMA_1W = pl.DataFrame({'SMA_1W': SMA_1W})
    hist = hist.hstack(SMA_1W)

    SMA_1M = talib.SMA(close_prices.to_numpy(), timeperiod=20)
    SMA_1M = pl.DataFrame({'SMA_1M': SMA_1M})
    hist = hist.hstack(SMA_1M)

    SMA_3M = talib.SMA(close_prices.to_numpy(), timeperiod=60)
    SMA_3M = pl.DataFrame({'SMA_3M': SMA_3M})
    hist = hist.hstack(SMA_3M)

    SMA_V_1W = talib.SMA(volume.cast(pl.Float64).to_numpy(), timeperiod=5)
    SMA_V_1W = pl.DataFrame({'SMA_V_1W': SMA_V_1W})
    hist = hist.hstack(SMA_V_1W)

    SMA_V_1M = talib.SMA(volume.cast(pl.Float64).to_numpy(), timeperiod=20)
    SMA_V_1M = pl.DataFrame({'SMA_V_1M': SMA_V_1M})
    hist = hist.hstack(SMA_V_1M)

    SMA_V_3M = talib.SMA(volume.cast(pl.Float64).to_numpy(), timeperiod=60)
    SMA_V_3M = pl.DataFrame({'SMA_V_3M': SMA_V_3M})
    hist = hist.hstack(SMA_V_3M)

    # FED RESERVE DATE
    fred = Fred('apiKey.txt')
    dff = fred.get_series_df('DFF')
    dff = dff.drop('realtime_start', axis=1).drop('realtime_end', axis=1)
    dff = pl.DataFrame(dff)
    dff = dff.with_columns(pl.col('date').str.strptime(pl.Date, format='%Y-%m-%d'))
    dff = dff.with_columns(pl.col('value').cast(pl.Float64))
    dff = dff.rename({"date": "Date", "value": "FedInterest"})
    hist = hist.join(dff, on='Date', how='left')

    # GOLIRE CAMPURI GOALE
    hist = hist.fill_nan(None).drop_nulls()
    print("!!DATE COLECTATE!!")

    hist.write_csv('datePrelucrate.csv')
    hist = hist.with_columns(pl.col('Date').cast(pl.Datetime))
    return hist


def saveData(polarTS):
    print("!!INCEPE SALVAREA DATELOR!!")
    write_api = client.write_api(SYNCHRONOUS)
    date = polarTS.to_pandas()

    date = date.set_index("Date")
    print(date)
    write_api.write(database, org, record=date, data_frame_measurement_name="stock")


def queryData():
    query_api = client.query_api()

    query = 'from(bucket: "TrainingData")\
              |> range(start: 0)\
              |> filter(fn: (r) => r["_measurement"] == "stock")\
              |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")'
    result = query_api.query_data_frame(org=org, query=query)

    result = pl.DataFrame(result)

    result = result.drop(['result', 'table', '_start', '_stop', '_measurement'])

    result = result.rename({"_time": "Date"})

    print(result)
    result.write_csv('dateInterogate.csv')

def deleteAllData():
    delete_api = client.delete_api()
    start = datetime.datetime.min
    stop = datetime.datetime.now()
    delete_api.delete(start, stop, f'_measurement="stock"', bucket=database, org=org)

if __name__ == "__main__":
    queryData()
    # date = colectData()
    # saveData(date)
