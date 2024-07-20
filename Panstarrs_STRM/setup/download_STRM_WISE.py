import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# URL of the directory containing .csv.gz files
base_url = "https://archive.stsci.edu/hlsp/wise-ps1-strm"

# Create a directory to store downloaded files
output_dir = "/export/home/rabbit2/temp/"
os.makedirs(output_dir, exist_ok=True)

# Function to download a file from a URL
def download_file(url, output_path):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(output_path, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)
    print(f"Downloaded: {output_path}")

# Fetch the HTML content of the page
response = requests.get(base_url)
response.raise_for_status()
soup = BeautifulSoup(response.text, 'html.parser')

# Find all links ending with .csv.gz
links = soup.find_all('a', href=True)
csv_gz_links = [urljoin(base_url, link['href']) for link in links if link['href'].endswith('.csv.gz')]

# Download each .csv.gz file
for link in csv_gz_links:
    file_name = os.path.basename(link)
    output_path = os.path.join(output_dir, file_name)
    if not os.path.exists(output_path):
        if '-m' not in output_path:
            download_file(link, output_path)

print("All files have been downloaded.")

time.sleep(600)
os.popen(f'mv *.csv.gz /lustre/aoc/sciops/ddong/Catalogs/STRM_WISE/data/')