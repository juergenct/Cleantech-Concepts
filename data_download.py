import os
import zipfile
import requests as req
import multiprocessing as mp

links_brf_sum = ['https://s3.amazonaws.com/data.patentsview.org/brief-summary-text/g_brf_sum_text_2023.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/brief-summary-text/g_brf_sum_text_2022.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/brief-summary-text/g_brf_sum_text_2021.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/brief-summary-text/g_brf_sum_text_2020.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/brief-summary-text/g_brf_sum_text_2019.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/brief-summary-text/g_brf_sum_text_2018.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/brief-summary-text/g_brf_sum_text_2017.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/brief-summary-text/g_brf_sum_text_2016.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/brief-summary-text/g_brf_sum_text_2015.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/brief-summary-text/g_brf_sum_text_2014.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/brief-summary-text/g_brf_sum_text_2013.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/brief-summary-text/g_brf_sum_text_2012.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/brief-summary-text/g_brf_sum_text_2011.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/brief-summary-text/g_brf_sum_text_2010.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/brief-summary-text/g_brf_sum_text_2009.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/brief-summary-text/g_brf_sum_text_2008.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/brief-summary-text/g_brf_sum_text_2007.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/brief-summary-text/g_brf_sum_text_2006.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/brief-summary-text/g_brf_sum_text_2005.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/brief-summary-text/g_brf_sum_text_2004.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/brief-summary-text/g_brf_sum_text_2003.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/brief-summary-text/g_brf_sum_text_2002.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/brief-summary-text/g_brf_sum_text_2001.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/brief-summary-text/g_brf_sum_text_2000.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/brief-summary-text/g_brf_sum_text_1999.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/brief-summary-text/g_brf_sum_text_1998.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/brief-summary-text/g_brf_sum_text_1997.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/brief-summary-text/g_brf_sum_text_1996.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/brief-summary-text/g_brf_sum_text_1995.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/brief-summary-text/g_brf_sum_text_1994.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/brief-summary-text/g_brf_sum_text_1993.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/brief-summary-text/g_brf_sum_text_1992.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/brief-summary-text/g_brf_sum_text_1991.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/brief-summary-text/g_brf_sum_text_1990.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/brief-summary-text/g_brf_sum_text_1989.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/brief-summary-text/g_brf_sum_text_1988.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/brief-summary-text/g_brf_sum_text_1987.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/brief-summary-text/g_brf_sum_text_1986.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/brief-summary-text/g_brf_sum_text_1985.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/brief-summary-text/g_brf_sum_text_1984.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/brief-summary-text/g_brf_sum_text_1983.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/brief-summary-text/g_brf_sum_text_1982.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/brief-summary-text/g_brf_sum_text_1981.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/brief-summary-text/g_brf_sum_text_1980.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/brief-summary-text/g_brf_sum_text_1979.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/brief-summary-text/g_brf_sum_text_1978.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/brief-summary-text/g_brf_sum_text_1977.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/brief-summary-text/g_brf_sum_text_1976.tsv.zip']
links_description = ['https://s3.amazonaws.com/data.patentsview.org/detail-description-text/g_detail_desc_text_2023.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/detail-description-text/g_detail_desc_text_2022.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/detail-description-text/g_detail_desc_text_2021.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/detail-description-text/g_detail_desc_text_2020.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/detail-description-text/g_detail_desc_text_2019.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/detail-description-text/g_detail_desc_text_2018.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/detail-description-text/g_detail_desc_text_2017.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/detail-description-text/g_detail_desc_text_2016.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/detail-description-text/g_detail_desc_text_2015.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/detail-description-text/g_detail_desc_text_2014.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/detail-description-text/g_detail_desc_text_2013.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/detail-description-text/g_detail_desc_text_2012.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/detail-description-text/g_detail_desc_text_2011.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/detail-description-text/g_detail_desc_text_2010.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/detail-description-text/g_detail_desc_text_2009.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/detail-description-text/g_detail_desc_text_2008.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/detail-description-text/g_detail_desc_text_2007.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/detail-description-text/g_detail_desc_text_2006.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/detail-description-text/g_detail_desc_text_2005.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/detail-description-text/g_detail_desc_text_2004.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/detail-description-text/g_detail_desc_text_2003.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/detail-description-text/g_detail_desc_text_2002.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/detail-description-text/g_detail_desc_text_2001.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/detail-description-text/g_detail_desc_text_2000.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/detail-description-text/g_detail_desc_text_1999.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/detail-description-text/g_detail_desc_text_1998.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/detail-description-text/g_detail_desc_text_1997.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/detail-description-text/g_detail_desc_text_1996.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/detail-description-text/g_detail_desc_text_1995.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/detail-description-text/g_detail_desc_text_1994.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/detail-description-text/g_detail_desc_text_1993.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/detail-description-text/g_detail_desc_text_1992.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/detail-description-text/g_detail_desc_text_1991.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/detail-description-text/g_detail_desc_text_1990.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/detail-description-text/g_detail_desc_text_1989.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/detail-description-text/g_detail_desc_text_1988.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/detail-description-text/g_detail_desc_text_1987.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/detail-description-text/g_detail_desc_text_1986.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/detail-description-text/g_detail_desc_text_1985.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/detail-description-text/g_detail_desc_text_1984.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/detail-description-text/g_detail_desc_text_1983.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/detail-description-text/g_detail_desc_text_1982.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/detail-description-text/g_detail_desc_text_1981.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/detail-description-text/g_detail_desc_text_1980.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/detail-description-text/g_detail_desc_text_1979.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/detail-description-text/g_detail_desc_text_1978.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/detail-description-text/g_detail_desc_text_1977.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/detail-description-text/g_detail_desc_text_1976.tsv.zip']
links_claims = ['https://s3.amazonaws.com/data.patentsview.org/claims/g_claims_2023.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/claims/g_claims_2022.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/claims/g_claims_2021.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/claims/g_claims_2020.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/claims/g_claims_2019.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/claims/g_claims_2018.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/claims/g_claims_2017.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/claims/g_claims_2016.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/claims/g_claims_2015.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/claims/g_claims_2014.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/claims/g_claims_2013.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/claims/g_claims_2012.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/claims/g_claims_2011.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/claims/g_claims_2010.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/claims/g_claims_2009.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/claims/g_claims_2008.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/claims/g_claims_2007.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/claims/g_claims_2006.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/claims/g_claims_2005.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/claims/g_claims_2004.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/claims/g_claims_2003.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/claims/g_claims_2002.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/claims/g_claims_2001.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/claims/g_claims_2000.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/claims/g_claims_1999.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/claims/g_claims_1998.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/claims/g_claims_1997.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/claims/g_claims_1996.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/claims/g_claims_1995.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/claims/g_claims_1994.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/claims/g_claims_1993.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/claims/g_claims_1992.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/claims/g_claims_1991.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/claims/g_claims_1990.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/claims/g_claims_1989.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/claims/g_claims_1988.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/claims/g_claims_1987.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/claims/g_claims_1986.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/claims/g_claims_1985.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/claims/g_claims_1984.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/claims/g_claims_1983.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/claims/g_claims_1982.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/claims/g_claims_1981.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/claims/g_claims_1980.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/claims/g_claims_1979.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/claims/g_claims_1978.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/claims/g_claims_1977.tsv.zip',
'https://s3.amazonaws.com/data.patentsview.org/claims/g_claims_1976.tsv.zip']

def download_brf_sum(link):
    file_name = link.split('/')[-1]
    brf_sum_dir = '/mnt/hdd01/patentsview/Brief Summary'
    print('Downloading ' + file_name)
    r = req.get(link, stream=True)
    with open(brf_sum_dir + '/' + file_name, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
    print(file_name + ' downloaded')

# def download_desc(link):
#     file_name = link.split('/')[-1]
#     desc_dir = '/mnt/hdd01/patentsview/Description'
#     print('Downloading ' + file_name)
#     r = req.get(link, stream=True)
#     with open(desc_dir + '/' + file_name, 'wb') as f:
#         for chunk in r.iter_content(chunk_size=1024):
#             if chunk:
#                 f.write(chunk)
#     print(file_name + ' downloaded')

# def download_claim(link):
#     file_name = link.split('/')[-1]
#     claim_dir = '/mnt/hdd01/patentsview/Claims'
#     print('Downloading ' + file_name)
#     r = req.get(link, stream=True)
#     with open(claim_dir + '/' + file_name, 'wb') as f:
#         for chunk in r.iter_content(chunk_size=1024):
#             if chunk:
#                 f.write(chunk)
#     print(file_name + ' downloaded')

def unzip_file_brf_sum(file_path):
    print('Unzipping ' + file_path)
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(brf_sum_dir)
    print(file_path + ' unzipped')
    # Throw away the zip file
    os.remove(file_path)
    print(file_path + ' removed')

def unzip_file_desc(file_path):
    print('Unzipping ' + file_path)
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(desc_dir)
    print(file_path + ' unzipped')
    # Throw away the zip file
    os.remove(file_path)
    print(file_path + ' removed')

def unzip_file_claim(file_path):
    print('Unzipping ' + file_path)
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(claim_dir)
    print(file_path + ' unzipped')
    # Throw away the zip file
    os.remove(file_path)
    print(file_path + ' removed')

if __name__ == '__main__':
    brf_sum_dir = '/mnt/hdd01/patentsview/Brief Summary'
    desc_dir = '/mnt/hdd01/patentsview/Description'
    claim_dir = '/mnt/hdd01/patentsview/Claims'
    
    # Ensure that the directories exist
    if not os.path.exists(claim_dir):
        os.makedirs(claim_dir)
    
    if not os.path.exists(desc_dir):
        os.makedirs(desc_dir)
    
    if not os.path.exists(brf_sum_dir):
        os.makedirs(brf_sum_dir)

    print("Starting download...")

    # pool_brf_sum = mp.Pool(mp.cpu_count()-2)
    # pool_brf_sum.map(download_brf_sum, links_brf_sum)
    # pool_brf_sum.close()

    # pool_desc = mp.Pool(mp.cpu_count()-2)
    # pool_desc.map(download_desc, links_description)
    # pool_desc.close()

    # pool_claim = mp.Pool(mp.cpu_count()-2)
    # pool_claim.map(download_claim, links_claims)
    # pool_claim.close()

    print("Files downloaded successfully.")

    # print("Starting unzipping...")

    # # Get all files in the directories ending with ".zip"
    # zip_files_brf_sum = [os.path.join(brf_sum_dir, file) for file in os.listdir(brf_sum_dir) if file.endswith(".zip")]
    # zip_files_desc = [os.path.join(desc_dir, file) for file in os.listdir(desc_dir) if file.endswith(".zip")]
    # zip_files_claim = [os.path.join(claim_dir, file) for file in os.listdir(claim_dir) if file.endswith(".zip")]

    pool_brf_sum_zip = mp.Pool(mp.cpu_count()-2)
    pool_brf_sum_zip.map(unzip_file_brf_sum, zip_files_brf_sum)
    pool_brf_sum_zip.close()

    pool_desc_zip = mp.Pool(mp.cpu_count()-2)
    pool_desc_zip.map(unzip_file_desc, zip_files_desc)
    pool_desc_zip.close()

    pool_claim_zip = mp.Pool(mp.cpu_count()-2)
    pool_claim_zip.map(unzip_file_claim, zip_files_claim)
    pool_claim_zip.close()

    # print("Files unzipped successfully.")