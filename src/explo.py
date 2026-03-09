import os

def explo(path):
    total_size = 0
    total_files = 0
    total_sea_files = 0
    total_non_sea_files = 0

    for folder in os.listdir(path):
        for _, _, files in os.walk(path +"/" + folder):
            total_files += len(files)
            if folder == "Mer":
                total_sea_files += len(files)
            else:
                total_non_sea_files += len(files)

            for file in files:
                total_size += os.path.getsize(os.path.join(path, folder, file)) 

    print(f"Total number of files : {total_files}")
    print(f"Total size : {total_size} octets")
    print(f"Total number of sea files : {total_sea_files}")
    print(f"Total number of non sea files : {total_non_sea_files}")

if __name__ == "__main__":
    explo("data/raw/Init")