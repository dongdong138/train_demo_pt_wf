import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from samformer import SAMFormer
from crawl import run
import datetime 

def read_dataset(seq_len, pred_len, time_increment=1):
    file_name = ".data/weather.csv"
    df_raw = pd.read_csv(file_name, index_col=0)
    train_df = df_raw
    
    scaler = StandardScaler()
    scaler.fit(train_df.values)
    train_df = scaler.transform(train_df.values)

    x_train, y_train = train_data(train_df, seq_len, pred_len, time_increment)

    flatten = lambda y: y.reshape((y.shape[0], y.shape[1] * y.shape[2]))
    y_train = flatten(y_train)

    return (x_train, y_train)

def crawl_and_forcast(seq_len, pred_len, time_increment=1, crawl=True):
    if crawl:
        tod = datetime.datetime.now()
        a = tod - datetime.timedelta(days = 35)
        run(str(a)[0:10], str(tod)[0:10])
    file_name = "data.csv"
    df = pd.read_csv(file_name, index_col=0)
    test_df = df.dropna()
    test_df = test_df[-1440:]
    
    x_test = np.array([test_df.T])

    (x_train, y_train) = read_dataset(seq_len, pred_len, time_increment)
    model = SAMFormer(device='cuda', num_epochs=100, batch_size=128, base_optimizer=torch.optim.Adam, 
                      learning_rate=1e-3, weight_decay=1e-5, rho=0.7, use_revin=True)
    model.fit(x_train, y_train, trainable=False)

    model.load_model(f'samformer_model_{seq_len}_{pred_len}.pth')

    y_pred_test = model.predict(x_test[:seq_len])

    y_pred_test = y_pred_test.reshape(1, 14, pred_len)

    return np.squeeze(y_pred_test.T)

def train_data(data, seq_len, pred_len, time_increment=1):
    n_samples = data.shape[0] - (seq_len - 1) - pred_len
    range_ = np.arange(0, n_samples, time_increment)
    x, y = list(), list()
    for i in range_:
        x.append(data[i:(i + seq_len)].T)
        y.append(data[(i + seq_len):(i + seq_len + pred_len)].T)
    
    return np.array(x), np.array(y)

def save_result(data, path):
    df = pd.read_csv("./data/weather.csv", index_col=0)
    column_types = {
    'temp'            : 'float64',
    'wx_phrase'       : 'int64',
    'dewPt'           : 'float64',
    'heat_index'      : 'float64',
    'rh'              : 'float64',
    'pressure'        : 'float64',
    'vis'             : 'float64',
    'wc'              : 'float64',
    'wdir_cardinal'   : 'int64',
    'wspd'            : 'float64',
    'uv_desc'         : 'int64',
    'feels_like'      : 'float64',
    'uv_index'        : 'float64',
    'clds'            : 'int64',
    }
    data = pd.DataFrame(data, columns=df.columns)
    data = data.astype(column_types)
    current_time = datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    time_index = [current_time + datetime.timedelta(minutes=30 * i) for i in range(len(data))]
    data['DateTime'] = time_index
    cols = ['DateTime'] + [col for col in data if col != 'DateTime']
    data = data[cols]
    data = data.round(2)
    data.to_csv(path, index=False)

def train_model(seq_len, pred_len, time_increment=1):
    (x_train, y_train) = read_dataset(seq_len, pred_len, time_increment)
    model = SAMFormer(device='cuda', num_epochs=100, batch_size=128, base_optimizer=torch.optim.Adam, 
                      learning_rate=1e-3, weight_decay=1e-5, rho=0.7, use_revin=True)
    model.fit(x_train, y_train, trainable=True)
    model.save_model(f'./models/samformer_model_{seq_len}_{pred_len}.pth')

if __name__ == '__main__':
    seq_len=1440 
    pred_len=96
    tmp = crawl_and_forcast(seq_len, pred_len, crawl=False)
    save_result(tmp, f"./results/result_{pred_len}.csv")
    