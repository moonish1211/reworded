import pandas as pd
import numpy as np
import scipy.signal as signal
import ast
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier


def features(data):
    avg_amp = np.mean(data)  # average amplitude
    ptp = np.ptp(data)  # peak to peak difference
    upper = np.max(data)  # upper limit
    lower = np.min(data)  # lower limit
    std = np.std(data)  # standard deviation
    return [avg_amp, ptp, upper, lower, std]


def correlation(channel_4, channel_5):
    return np.corrcoef(channel_4, channel_5)[0, 1]


def extract_features(row):
    channel_4_features = features(row['channel_4'])
    channel_5_features = features(row['channel_5'])
    corr = correlation(row['channel_4'], row['channel_5'])
    return channel_4_features + channel_5_features + [corr]


def bandpass(data, lowerfreq=0.1, upperfreq=30, sampling_rate=250):
    lower = lowerfreq / (sampling_rate / 2)
    upper = upperfreq / (sampling_rate / 2)
    x, y = signal.butter(4, [lower, upper], btype='band')
    return signal.filtfilt(x, y, data)


# csv consolidation
data1 = pd.read_csv("data\experiment_data_03-18_15-27_Michael(Good).csv")
data2 = pd.read_csv("data\experiment_data_03-18_15-46_Michael_Trial_2(Good).csv")
data3 = pd.read_csv("data\experiment_data_03-18_15-59_Michael_Trial_3(Good).csv")

data = pd.concat([file for file in [data1, data2, data3]], ignore_index=True)

# data filtering
data = data[['word', 'label', 'user_input', 'channel_4', 'channel_5']]  # channel 4 and 5 correspond to T5 and P3 in N400

# convert channel lists from str to lists
data['channel_4'] = data['channel_4'].apply(lambda x: ast.literal_eval(x))
data['channel_5'] = data['channel_5'].apply(lambda x: ast.literal_eval(x))

# apply filter
data['channel_4'] = data['channel_4'].apply(lambda x: bandpass(np.array(x)))
data['channel_5'] = data['channel_5'].apply(lambda x: bandpass(np.array(x)))

# create prediction_type column
data['prediction_type'] = np.where(
    (data['label'] == 1) & (data['user_input'] == 1), 'TP',  # True Positive
    np.where(
        (data['label'] == 0) & (data['user_input'] == 0), 'TN',  # True Negative
        np.where(
            (data['label'] == 0) & (data['user_input'] == 1), 'FP',  # False Positive
            'FN'  # False Negative
        )
    )
)

# create user correctness column
data['user_target'] = (data['label'] == data['user_input']).astype(int)

X = data.apply(lambda row: extract_features(row), axis=1).tolist()

y_prediction_type = data['prediction_type'].values
y_user_target = data['user_target'].values

X = np.array(X)

# train and test sets for prediction_type
X_train, X_test, y_train, y_test = train_test_split(X, y_prediction_type, test_size=0.2, random_state=42)

# train and test sets for user_target
X_train_user, X_test_user, y_train_user, y_test_user = train_test_split(X, y_user_target, test_size=0.2, random_state=42)

# scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled_user = scaler.fit_transform(X_train_user)
X_test_scaled_user = scaler.transform(X_test_user)

# svm for prediction_type and user_target
svm_model = SVC(kernel='linear', C=1, gamma='scale')
svm_model.fit(X_train_scaled, y_train)
y_pred_svm = svm_model.predict(X_test_scaled)

svm_user_model = SVC(kernel='linear', C=1, gamma='scale')
svm_user_model.fit(X_train_scaled_user, y_train_user)
y_pred_user_svm = svm_user_model.predict(X_test_scaled_user)

# gbm for prediction_type and user_target
gbm_model = GradientBoostingClassifier(n_estimators=200, random_state=42)
gbm_model.fit(X_train_scaled, y_train)
y_pred_gbm = gbm_model.predict(X_test_scaled)

gbm_user_model = GradientBoostingClassifier(n_estimators=200, random_state=42)
gbm_user_model.fit(X_train_scaled_user, y_train_user)
y_pred_user_gbm = gbm_user_model.predict(X_test_scaled_user)

# nn (mlp) for prediction_type and user_target
mlp_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
mlp_model.fit(X_train_scaled, y_train)
y_pred_mlp = mlp_model.predict(X_test_scaled)

mlp_user_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
mlp_user_model.fit(X_train_scaled_user, y_train_user)
y_pred_user_mlp = mlp_user_model.predict(X_test_scaled_user)

print("Prediction Type (SVM):", accuracy_score(y_test, y_pred_svm))
print("Prediction Type (GBM):", accuracy_score(y_test, y_pred_gbm))
print("Prediction Type (MLP):", accuracy_score(y_test, y_pred_mlp))

print("User Identification (SVM):", accuracy_score(y_test_user, y_pred_user_svm))
print("User Identification (GBM):", accuracy_score(y_test_user, y_pred_user_gbm))
print("User Identification (MLP):", accuracy_score(y_test_user, y_pred_user_mlp))
