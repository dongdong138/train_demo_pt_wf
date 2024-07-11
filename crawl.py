import requests
import openpyxl
import datetime
from bs4 import BeautifulSoup
import pandas as pd

def get_headers():
    headers = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
    "Accept-Encoding": "gzip, deflate, br, zstd",
    "Accept-Language": "en-US,en;q=0.9,vi;q=0.8",
    "Cache-Control": "max-age=0",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
}
    return headers


class NetWorkError(Exception):
    pass


def build_request(url, headers=None):
    if headers is None:
        headers = get_headers()
    for i in range(3):
        try:
            scraperapi_url = f"http://api.scraperapi.com?api_key=2515295f4eb0c0fc5cb6fa62ee9e81f0&url={url}"
            response = requests.get(scraperapi_url, headers=headers, stream=True)
            # print(response)
            return response
        except:
            continue
    raise NetWorkError

def write_to_excel(lines,filename,write_only=True):
    excel=openpyxl.Workbook(write_only=write_only)
    sheet=excel.create_sheet()
    for line in lines:
        sheet.append(line)
    excel.save(filename)

def get_next_date(current_date='2017-01-01'):
    current_date=datetime.datetime.strptime(current_date,'%Y-%m-%d')
    oneday = datetime.timedelta(days=1)
    next_date = current_date+oneday
    return str(next_date).split(' ')[0]

def create_url(latitude, longitude, start_date, end_date):
    url = 'https://api.weather.com/v1/geocode/{}/{}/observations/historical.json?apiKey=e1f10a1e78da46f5b10a1e78da96f525&startDate={}&endDate={}&units=e'.format(
        latitude, longitude, start_date, end_date )
    return url

def get_json(url):
    req = build_request(url)
    # print(req)
    html = BeautifulSoup(req.text, 'lxml')
    res_json = eval(html.find("p").get_text().replace('null', '"null"'))
    return res_json

def crawl(latitude, longitude,start_date, end_date):
    res_obs = []
    current_date = start_date
    while current_date != end_date: 
        date = current_date.replace('-','')
        url = create_url(latitude, longitude, date, date)
        res_json = get_json(url)
        obs = res_json['observations']
        # print(obs)
        for ob in obs:
            valid_time_gmt = ob['valid_time_gmt']
            _,s=divmod(valid_time_gmt - 1577811600,86400)
            h,s=divmod(s,3600)
            m,s=divmod(s, 60)
            a = current_date + "  " + str(h) +":"+ str(m).zfill(2) +":"+ str(s).zfill(2)
            ob['DateTime'] = a
            res_obs.append(ob)
        print(current_date,'OK')
        current_date=get_next_date(current_date)
    return res_obs

def label_encoders(data, path):
    from sklearn.preprocessing import LabelEncoder

    # Create a copy of the data to preserve the original
    data_encoded = data.copy()

    # Initialize label encoders
    label_encoders = {}

    # Columns to be label encoded
    categorical_columns = ['wx_phrase', 'wdir_cardinal', 'uv_desc', 'clds']

    # Apply label encoding to each categorical column
    for column in categorical_columns:
        le = LabelEncoder()
        data_encoded[column] = le.fit_transform(data_encoded[column])
        label_encoders[column] = le

    # Create a dictionary to store the mapping of each categorical column
    mapping_dict = {}

    # Extract the mapping for each column
    for column, le in label_encoders.items():
        mapping_dict[column] = dict(zip(le.classes_, le.transform(le.classes_)))


    # Save the mapping dictionary to a text file
    mapping_file_path = path + '.txt'

    with open(mapping_file_path, 'w') as file:
        for column, mapping in mapping_dict.items():
            file.write(f"{column}:\n")
            for key, value in mapping.items():
                file.write(f"  {key}: {value}\n")
            file.write("\n")
    
    return data_encoded

def run(start='2024-06-16', end='2024-06-19', path='data'):
    obs = crawl(10.82, 106.64, start, end)
    df = pd.DataFrame(obs)
    
    columns_order = [
        'DateTime', 'temp', 'wx_phrase', 'dewPt', 
        'heat_index', 'rh', 'pressure', 'vis', 'wc', 
        'wdir_cardinal', 'wspd', 'uv_desc', 
        'feels_like', 'uv_index', 'clds'
    ]
    # Reorder the DataFrame columns
    df = df[columns_order]
    df = label_encoders(df, path= path)
    # Save the DataFrame to a CSV file
    df.to_csv(path+'.csv', index=False)
    return df

