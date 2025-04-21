import numpy as np

# Load the .npz file
data = np.load('/nobackup/rossby26/users/sm_fuxwa/AI/SRGAN_OUT/EPOCH150/preds.npz')

# Print the names of the arrays stored in the file
print("Arrays in the .npz file:", data.files)

# Display each array's data
for name in data.files:
    print(f"\nArray name: {name}")
    print(data[name])

