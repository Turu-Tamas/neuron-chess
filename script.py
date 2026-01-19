import h5py

file_path = "data/lc0-hidden/lichess_elite_2025-11.h5"

with h5py.File(file_path, "r") as f:
    print(f"--- File structure: {file_path} ---")
    for key in f.keys():
        dataset = f[key]
        print(f"Key name:  {key}")
        print(f"  Type:    {type(dataset)}")
        if isinstance(dataset, h5py.Dataset):
            print(f"  Dimensions: {dataset.shape}")
            print(f"  Datatype: {dataset.dtype}")
        print("-" * 30)
