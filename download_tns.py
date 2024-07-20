from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import pandas as pd
import os
import time
import shutil

# Set up the Chrome WebDriver
driver_path = '/Users/ddong/Downloads/chromedriver-mac-arm64/chromedriver'  # Update with your actual path to chromedriver

# Initialize the Chrome options
chrome_options = webdriver.ChromeOptions()

# Set up the Chrome service
chrome_service = ChromeService(executable_path=driver_path)

# Initialize the WebDriver
driver = webdriver.Chrome(service=chrome_service, options=chrome_options)

# Loop through each page and download the CSV files
total_rows = 14498
rows_per_page = 50
num_pages = total_rows // rows_per_page + 1

download_directory = '/Users/ddong/Downloads'
target_directory = 'tns_downloads'
os.makedirs(target_directory, exist_ok=True)

# Define the number of retries and delay between retries
max_retries = 3
retry_delay = 5  # seconds

def download_csv(url, page):
    retries = 0
    while retries < max_retries:
        try:
            driver.get(url)

            # Wait for the page to load and the CSV link to be clickable
            wait = WebDriverWait(driver, 10)
            csv_link = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "a.search-csv-lnk[data-format='csv']")))

            # Click the CSV link to download the file
            csv_link.click()

            # Give the browser time to download the file
            time.sleep(10)

            # Move the downloaded file to a specific location
            downloaded_file = os.path.join(download_directory, 'tns_search.csv')
            target_file = os.path.join(target_directory, f'tns_data_page_{page}.csv')
            shutil.move(downloaded_file, target_file)

            # Break the loop if successful
            break
        except TimeoutException as e:
            print(f"TimeoutException occurred on page {page}: {e}. Retrying in {retry_delay} seconds...")
            retries += 1
            time.sleep(retry_delay)
        except FileNotFoundError as e:
            print(f"File not found error on page {page}: {e}. Retrying in {retry_delay} seconds...")
            retries += 1
            time.sleep(retry_delay)

    if retries == max_retries:
        print(f"Failed to download the CSV after multiple attempts on page {page}.")
    else:
        print(f"CSV downloaded successfully for page {page}.")

for page in range(201, num_pages):
    # Construct the URL for the current page
    #url = f"https://www.wis-tns.org/search?&page={page}&discovered_period_value=&discovered_period_units=days&unclassified_at=0&classified_sne=1&include_frb=0&name=&name_like=0&isTNS_AT=yes&public=all&ra=&decl=&radius=&coords_unit=arcsec&reporting_groupid%5B%5D=null&groupid%5B%5D=null&classifier_groupid%5B%5D=null&objtype%5B%5D=null&at_type%5B%5D=null&date_start%5Bdate%5D=&date_end%5Bdate%5D=&discovery_mag_min=&discovery_mag_max=&internal_name=&discoverer=&classifier=&spectra_count=&redshift_min=&redshift_max=&hostname=&ext_catid=&ra_range_min=&ra_range_max=&decl_range_min=&decl_range_max=&discovery_instrument%5B%5D=null&classification_instrument%5B%5D=null&associated_groups%5B%5D=null&official_discovery=0&official_classification=0&at_rep_remarks=&class_rep_remarks=&frb_repeat=all&frb_repeater_of_objid=&frb_measured_redshift=0&frb_dm_range_min=&frb_dm_range_max=&frb_rm_range_min=&frb_rm_range_max=&frb_snr_range_min=&frb_snr_range_max=&frb_flux_range_min=&frb_flux_range_max=&num_page=100&display%5Bredshift%5D=1&display%5Bhostname%5D=1&display%5Bhost_redshift%5D=1&display%5Bsource_group_name%5D=1&display%5Bclassifying_source_group_name%5D=0&display%5Bdiscovering_instrument_name%5D=1&display%5Bclassifing_instrument_name%5D=0&display%5Bprograms_name%5D=0&display%5Binternal_name%5D=0&display%5BisTNS_AT%5D=1&display%5Bpublic%5D=0&display%5Bend_pop_period%5D=0&display%5Bspectra_count%5D=1&display%5Bdiscoverymag%5D=1&display%5Bdiscmagfilter%5D=1&display%5Bdiscoverydate%5D=1&display%5Bdiscoverer%5D=0&display%5Bremarks%5D=0&display%5Bsources%5D=0&display%5Bbibcode%5D=0&display%5Bext_catalogs%5D=0"
    #url = f"https://www.wis-tns.org/search?&page={page}&discovered_period_value=8&discovered_period_units=years&unclassified_at=0&classified_sne=1&include_frb=0&name=&name_like=0&isTNS_AT=all&public=all&ra=&decl=&radius=&coords_unit=arcsec&reporting_groupid%5B%5D=null&groupid%5B%5D=null&classifier_groupid%5B%5D=null&objtype%5B%5D=null&at_type%5B%5D=null&date_start%5Bdate%5D=&date_end%5Bdate%5D=&discovery_mag_min=&discovery_mag_max=&internal_name=&discoverer=&classifier=&spectra_count=&redshift_min=&redshift_max=&hostname=&ext_catid=&ra_range_min=&ra_range_max=&decl_range_min=&decl_range_max=&discovery_instrument%5B%5D=null&classification_instrument%5B%5D=null&associated_groups%5B%5D=null&official_discovery=0&official_classification=0&at_rep_remarks=&class_rep_remarks=&frb_repeat=all&frb_repeater_of_objid=&frb_measured_redshift=0&frb_dm_range_min=&frb_dm_range_max=&frb_rm_range_min=&frb_rm_range_max=&frb_snr_range_min=&frb_snr_range_max=&frb_flux_range_min=&frb_flux_range_max=&num_page=50&display%5Bredshift%5D=1&display%5Bhostname%5D=1&display%5Bhost_redshift%5D=1&display%5Bsource_group_name%5D=1&display%5Bclassifying_source_group_name%5D=1&display%5Bdiscovering_instrument_name%5D=0&display%5Bclassifing_instrument_name%5D=0&display%5Bprograms_name%5D=0&display%5Binternal_name%5D=1&display%5BisTNS_AT%5D=0&display%5Bpublic%5D=1&display%5Bend_pop_period%5D=0&display%5Bspectra_count%5D=1&display%5Bdiscoverymag%5D=1&display%5Bdiscmagfilter%5D=1&display%5Bdiscoverydate%5D=1&display%5Bdiscoverer%5D=1&display%5Bremarks%5D=0&display%5Bsources%5D=0&display%5Bbibcode%5D=0&display%5Bext_catalogs%5D=0"
    start_date='2009-08-12'
    end_date='2019-09-18'
    offset = 200
    url=f"https://www.wis-tns.org/search?&page={page-offset}&discovered_period_value=8&discovered_period_units=years&unclassified_at=0&classified_sne=1&include_frb=0&name=&name_like=0&isTNS_AT=all&public=all&ra=&decl=&radius=&coords_unit=arcsec&reporting_groupid%5B%5D=null&groupid%5B%5D=null&classifier_groupid%5B%5D=null&objtype%5B%5D=null&at_type%5B%5D=null&date_start%5Bdate%5D={start_date}&date_end%5Bdate%5D={end_date}&discovery_mag_min=&discovery_mag_max=&internal_name=&discoverer=&classifier=&spectra_count=&redshift_min=&redshift_max=&hostname=&ext_catid=&ra_range_min=&ra_range_max=&decl_range_min=&decl_range_max=&discovery_instrument%5B%5D=null&classification_instrument%5B%5D=null&associated_groups%5B%5D=null&official_discovery=0&official_classification=0&at_rep_remarks=&class_rep_remarks=&frb_repeat=all&frb_repeater_of_objid=&frb_measured_redshift=0&frb_dm_range_min=&frb_dm_range_max=&frb_rm_range_min=&frb_rm_range_max=&frb_snr_range_min=&frb_snr_range_max=&frb_flux_range_min=&frb_flux_range_max=&num_page=50&display%5Bredshift%5D=1&display%5Bhostname%5D=1&display%5Bhost_redshift%5D=1&display%5Bsource_group_name%5D=1&display%5Bclassifying_source_group_name%5D=1&display%5Bdiscovering_instrument_name%5D=0&display%5Bclassifing_instrument_name%5D=0&display%5Bprograms_name%5D=0&display%5Binternal_name%5D=1&display%5BisTNS_AT%5D=0&display%5Bpublic%5D=1&display%5Bend_pop_period%5D=0&display%5Bspectra_count%5D=1&display%5Bdiscoverymag%5D=1&display%5Bdiscmagfilter%5D=1&display%5Bdiscoverydate%5D=1&display%5Bdiscoverer%5D=1&display%5Bremarks%5D=0&display%5Bsources%5D=0&display%5Bbibcode%5D=0&display%5Bext_catalogs%5D=0"
    print(f'Processing {url}\n')
    download_csv(url, page)

# Close the WebDriver
driver.quit()

# Combine all downloaded CSV files into one
all_files = [os.path.join(target_directory, f) for f in os.listdir(target_directory) if f.endswith('.csv')]
combined_csv = pd.concat([pd.read_csv(f) for f in all_files])
combined_csv.to_csv('tns_data_combined.csv', index=False)

print("All CSV files have been combined into 'tns_data_combined.csv'.")
