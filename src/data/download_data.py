import kagglehub
import shutil
import os
import tarfile
from tqdm import tqdm
import sys

def download_and_extract():
    try:
        tmp_path = kagglehub.dataset_download("atulanandjha/lfwpeople")

        target_raw = "data/raw"
        os.makedirs(target_raw, exist_ok=True)
        
        for item in os.listdir(tmp_path):
            s = os.path.join(tmp_path, item)
            d = os.path.join(target_raw, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, dirs_exist_ok=True)
            else:
                shutil.copy2(s, d)
        

        name = "lfw-funneled"
        tgz_file = os.path.join(target_raw, f"{name}.tgz")
        extract_to = os.path.join(target_raw, name) 

        if not os.path.exists(tgz_file):
            raise FileNotFoundError(f"File not found!")

        with tarfile.open(tgz_file, "r:gz") as tar:
            members = tar.getmembers()
            progress_bar = tqdm(iterable=members, total=len(members), unit="file", desc="Extracting")

            for member in progress_bar:
                tar.extract(member, path=target_raw)
                progress_bar.set_postfix(file=member.name[-20:])
        
        print("\nComplete!")

    except Exception as e:
        sys.exit(1) # Thoát chương trình với mã lỗi

if __name__ == "__main__":
    download_and_extract()
