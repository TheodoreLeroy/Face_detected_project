import tarfile

name = "lfw-funneled"
def extract_tgz(file_path, dest_path):
    with tarfile.open(file_path, "r:gz") as tar:
        tar.extractall(path=dest_path)
        print(f"Extract complete")

extract_tgz("data/raw/lfw-funneled.tgz", "data/raw/extracted_folder")