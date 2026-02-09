import os

def explo(path):
    total_size = 0
    total_files = 0
    total_sea_files = 0
    total_non_sea_files = 0

    for dir in os.listdir(path):
        for path, dirs, files in os.walk(path +"/" + dir):
            total_files += len(files)
            if dir == "Mer":
                total_sea_files += len(files)
            else:
                total_non_sea_files += len(files)
        total_size += os.path.getsize(path +"/")

    print(f"Total number of files : {total_files}")
    print(f"Total size : {total_size}")
    print(f"Total number of sea files : {total_sea_files}")
    print(f"Total number of non sea files : {total_non_sea_files}")

if __name__ == "__main__":
    explo("data/raw/Init")