import os


def find_tif(search_path):
    result=[]

    # Recursive wal through the directory
    for root, dir, files, in os.walk(search_path):
        for file in files:
            if file.lower().endswith(".tif"):
                result.append(os.path.join(root,file))
    return result

search_path = "../Turing Dataset/Ortomosaicos/"
tif_files = find_tif(search_path)

for f in tif_files:
    print(f)
    
print(f"\n Total {len(tif_files)} found")

print(tif_files)