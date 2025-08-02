# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy",
# ]
# ///
import numpy as np

# Load the npy file
file_path = 'src/webapp/2022probs.npy'
data = np.load(file_path, allow_pickle=True)

# Display basic information about the data
print(f"Data type: {type(data)}")
print(f"Data dtype: {data.dtype}")
print(f"Data shape: {data.shape}")
print(f"Data size: {data.size}")

# Show more details based on data structure
if isinstance(data, np.ndarray):
    if data.ndim == 1:
        print(f"\nFirst 10 elements: {data[:10]}")
        print(f"Last 10 elements: {data[-10:]}")
    elif data.ndim == 2:
        print(f"\nFirst 5 rows:\n{data[:5]}")
        print(f"\nLast 5 rows:\n{data[-5:]}")
    else:
        print(f"\nData has {data.ndim} dimensions")
        print(f"Sample slice: {data.flat[:20]}")
else:
    # If it's a pickled object
    print(f"\nData is a {type(data)} object")
    if hasattr(data, '__len__'):
        print(f"Length: {len(data)}")
    if hasattr(data, 'keys') and callable(data.keys):
        print(f"Keys: {list(data.keys())[:10]}")  # Show first 10 keys if dict-like