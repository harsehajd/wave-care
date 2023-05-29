import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load your EEG data
# Assuming you have EEG data stored in a numpy array 'eeg_data' with shape (num_samples, num_channels, num_timesteps)

# Load depressive disorder labels
# Assuming you have labels stored in a numpy array 'labels' with shape (num_samples,)

# Reshape the EEG data to have a single channel
eeg_data = eeg_data.reshape(eeg_data.shape[0], eeg_data.shape[1], eeg_data.shape[2], 1)

# Normalize the EEG data using StandardScaler
scaler = StandardScaler()
eeg_data = scaler.fit_transform(eeg_data.reshape(-1, eeg_data.shape[-1])).reshape(eeg_data.shape)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(eeg_data, labels, test_size=0.2, random_state=42)

# Continue with the rest of the code for model training and evaluation
