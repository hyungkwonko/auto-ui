from bs4 import BeautifulSoup
import requests
import pandas as pd
import time
from tqdm import tqdm

name = 'links2'

abs = []

links = ['https://doi.org/10.1145/3325480.3325499']
# links = pd.read_csv('links.csv')[name].tolist()

print(links)

for link in tqdm(links):
    time.sleep(0.3)

    print(link)

    html = requests.get(link)
    bs_html = BeautifulSoup(html.content, "html.parser")

    print(bs_html)

    print(bs_html.find_all('div', {'class':'abstractSection'}))

    print(link)
    exit()

    for row in bs_html.find_all('div', {'class':'abstractSection'}):
        try:
            abs.append(row.text)
        except:
            abs.append('None')
    
    print(abs)
    
out = pd.DataFrame()
out['links'] = links
out['abs'] = abs
out.to_csv(f'{name}.csv', index=False)