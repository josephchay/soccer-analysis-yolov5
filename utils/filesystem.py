import os
import zipfile


def extract_zip(zip_path, extract_path='.', del_zip=True):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
        print("Extracted dataset video zip file")

    if del_zip:
        os.remove(zip_path)
        print("Deleted dataset video zip file")
