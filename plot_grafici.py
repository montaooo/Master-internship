import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tesseract import temporal, spatial
from strategies.continual_learning import clean_dsets
'''
aut_precision = [0.9380555555555554, 0.9585277777777779, 0.9513611111111109]
aut_tpr = [0.6924166666666666, 0.886111111111111, 0.9416666666666668]
aut_f1 = [0.7533333333333332, 0.8943055555555554, 0.9311666666666666]
plot_data = {
    "no_retrain": [
        {"Date": "2016-10-01", "Precision": 1.000, "F1": 0.998, "TNR": 1.000, "TPR": 0.997},
        {"Date": "2016-11-01", "Precision": 1.000, "F1": 0.736, "TNR": float('nan'), "TPR": 0.586},
        {"Date": "2016-12-01", "Precision": 1.000, "F1": 0.896, "TNR": float('nan'), "TPR": 0.818},
        {"Date": "2017-01-01", "Precision": 1.000, "F1": 0.373, "TNR": float('nan'), "TPR": 0.229},
        {"Date": "2017-02-01", "Precision": 1.000, "F1": 0.980, "TNR": float('nan'), "TPR": 0.963},
        {"Date": "2017-03-01", "Precision": 1.000, "F1": 0.725, "TNR": float('nan'), "TPR": 0.569},
        {"Date": "2017-04-01", "Precision": 0.701, "F1": 0.662, "TNR": 0.426, "TPR": 0.629},
        {"Date": "2017-05-01", "Precision": 0.281, "F1": 0.395, "TNR": 0.317, "TPR": 0.667},
        {"Date": "2017-06-01", "Precision": 1.000, "F1": 0.711, "TNR": float('nan'), "TPR": 0.554},
        {"Date": "2017-07-01", "Precision": 0.906, "F1": 0.650, "TNR": 0.575, "TPR": 0.507},
        {"Date": "2017-08-01", "Precision": 1.000, "F1": 0.914, "TNR": float('nan'), "TPR": 0.843},
        {"Date": "2017-09-01", "Precision": 0.999, "F1": 0.885, "TNR": 0.998, "TPR": 0.793},
        {"Date": "2017-10-01", "Precision": 1.000, "F1": 0.996, "TNR": float('nan'), "TPR": 0.992},
        {"Date": "2017-11-01", "Precision": 1.000, "F1": 0.851, "TNR": float('nan'), "TPR": 0.742},
        {"Date": "2017-12-01", "Precision": 1.000, "F1": 0.994, "TNR": float('nan'), "TPR": 0.989},
        {"Date": "2018-01-01", "Precision": 1.000, "F1": 0.972, "TNR": float('nan'), "TPR": 0.946},
        {"Date": "2018-02-01", "Precision": 1.000, "F1": 0.178, "TNR": float('nan'), "TPR": 0.102},
        {"Date": "2018-03-01", "Precision": 1.000, "F1": 0.647, "TNR": float('nan'), "TPR": 0.542},
        {"Date": "2018-04-01", "Precision": 0.996, "F1": 0.992, "TNR": 0.671, "TPR": 0.988}
    ],
    "cl": [
        {"Date": "2016-10-01", "Precision": 1.000, "F1": 0.998, "TNR": 1.000, "TPR": 0.997},
        {"Date": "2016-11-01", "Precision": 1.000, "F1": 0.768, "TNR": float('nan'), "TPR": 0.629},
        {"Date": "2016-12-01", "Precision": 1.000, "F1": 0.999, "TNR": float('nan'), "TPR": 0.998},
        {"Date": "2017-01-01", "Precision": 1.000, "F1": 0.700, "TNR": float('nan'), "TPR": 0.540},
        {"Date": "2017-02-01", "Precision": 1.000, "F1": 0.974, "TNR": float('nan'), "TPR": 0.952},
        {"Date": "2017-03-01", "Precision": 1.000, "F1": 0.985, "TNR": float('nan'), "TPR": 0.970},
        {"Date": "2017-04-01", "Precision": 0.719, "F1": 0.835, "TNR": 0.167, "TPR": 0.998},
        {"Date": "2017-05-01", "Precision": 0.557, "F1": 0.704, "TNR": 0.691, "TPR": 0.961},
        {"Date": "2017-06-01", "Precision": 1.000, "F1": 0.998, "TNR": float('nan'), "TPR": 0.997},
        {"Date": "2017-07-01", "Precision": 0.980, "F1": 0.909, "TNR": 0.866, "TPR": 0.848},
        {"Date": "2017-08-01", "Precision": 1.000, "F1": 0.978, "TNR": float('nan'), "TPR": 0.958},
        {"Date": "2017-09-01", "Precision": 0.998, "F1": 0.998, "TNR": 0.981, "TPR": 0.997},
        {"Date": "2017-10-01", "Precision": 1.000, "F1": 0.973, "TNR": float('nan'), "TPR": 0.949},
        {"Date": "2017-11-01", "Precision": 1.000, "F1": 0.999, "TNR": float('nan'), "TPR": 0.998},
        {"Date": "2017-12-01", "Precision": 1.000, "F1": 0.997, "TNR": float('nan'), "TPR": 0.995},
        {"Date": "2018-01-01", "Precision": 1.000, "F1": 0.999, "TNR": float('nan'), "TPR": 0.999},
        {"Date": "2018-02-01", "Precision": 1.000, "F1": 0.285, "TNR": float('nan'), "TPR": 0.166},
        {"Date": "2018-03-01", "Precision": 1.000, "F1": 0.998, "TNR": float('nan'), "TPR": 0.997},
        {"Date": "2018-04-01", "Precision": 0.999, "F1": 0.999, "TNR": 0.993, "TPR": 0.999}
    ],
    "cl_mu": [
        {"Date": "2016-10-01", "Precision": 1.000, "F1": 0.997, "TNR": 1.000, "TPR": 0.994},
        {"Date": "2016-11-01", "Precision": 1.000, "F1": 0.763, "TNR": float('nan'), "TPR": 0.622},
        {"Date": "2016-12-01", "Precision": 1.000, "F1": 0.999, "TNR": float('nan'), "TPR": 0.998},
        {"Date": "2017-01-01", "Precision": 1.000, "F1": 0.706, "TNR": float('nan'), "TPR": 0.546},
        {"Date": "2017-02-01", "Precision": 1.000, "F1": 0.992, "TNR": float('nan'), "TPR": 0.985},
        {"Date": "2017-03-01", "Precision": 1.000, "F1": 0.984, "TNR": float('nan'), "TPR": 0.969},
        {"Date": "2017-04-01", "Precision": 0.799, "F1": 0.882, "TNR": 0.459, "TPR": 0.986},
        {"Date": "2017-05-01", "Precision": 0.336, "F1": 0.495, "TNR": 0.252, "TPR": 0.940},
        {"Date": "2017-06-01", "Precision": 1.000, "F1": 0.998, "TNR": float('nan'), "TPR": 0.998},
        {"Date": "2017-07-01", "Precision": 0.993, "F1": 0.992, "TNR": 0.951, "TPR": 0.992},
        {"Date": "2017-08-01", "Precision": 1.000, "F1": 0.990, "TNR": float('nan'), "TPR": 0.980},
        {"Date": "2017-09-01", "Precision": 0.998, "F1": 0.975, "TNR": 0.977, "TPR": 0.956},
        {"Date": "2017-10-01", "Precision": 1.000, "F1": 0.998, "TNR": float('nan'), "TPR": 0.998},
        {"Date": "2017-11-01", "Precision": 1.000, "F1": 0.999, "TNR": float('nan'), "TPR": 0.999},
        {"Date": "2017-12-01", "Precision": 1.000, "F1": 0.998, "TNR": float('nan'), "TPR": 0.998},
        {"Date": "2018-01-01", "Precision": 1.000, "F1": 0.999, "TNR": float('nan'), "TPR": 0.998},
        {"Date": "2018-02-01", "Precision": 1.000, "F1": 0.995, "TNR": float('nan'), "TPR": 0.991},
        {"Date": "2018-03-01", "Precision": 1.000, "F1": 0.999, "TNR": float('nan'), "TPR": 0.998},
        {"Date": "2018-04-01", "Precision": 0.997, "F1": 0.997, "TNR": 0.817, "TPR": 0.998}
    ]
}

all_dates = [item['Date'] for item in plot_data["no_retrain"]]

metrics = ["Precision", "F1", "TPR", "TNR"]
pendleblue="#1264B7"
pendlegreen="#47a91d"
pendlered="#901212"

labels_info = [
    ('no_retrain', pendlered),
    ('cl', pendleblue),
    ('cl_mu', pendlegreen),
]

for m in metrics:
    fig, ax1 = plt.subplots(figsize=(20, 10))

    for label, color in labels_info:
        df = pd.DataFrame(plot_data[label])
        df_plot = df.dropna(subset=[m])
        dates_dt = pd.to_datetime(df_plot['Date'])
        ax1.plot(dates_dt, df_plot[m], marker='o', color=color, label=label)

    all_dates = pd.to_datetime(all_dates)
    ax1.set_xticks(all_dates)
    ax1.set_xticklabels([d.strftime('%m-%Y') for d in all_dates], rotation=45)
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(fontsize=14)
    plt.xlabel('Data', fontsize=16)
    plt.ylabel(m, fontsize=16)
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


# Istogramma dei valori AUT
no_retrain_vals = [aut_precision[0], aut_tpr[0], aut_f1[0]]
cl_vals = [aut_precision[1], aut_tpr[1], aut_f1[1]]
cl_mu_vals = [aut_precision[2], aut_tpr[2], aut_f1[2]]
metrics_label = ["Precision", "TPR", "F1"]
labels = ['no_retrain', 'cl', 'cl_mu']
colors = [pendlered, pendleblue, pendlegreen]

x = np.arange(3)
width = 0.25

fig, ax = plt.subplots(figsize=(15,10))

graph1 = ax.bar(x - width, no_retrain_vals, width, label='no_retrain', color=pendlered, edgecolor='black', alpha=0.8)
graph2 = ax.bar(x, cl_vals, width, label='cl', color=pendleblue, edgecolor='black', alpha=0.8)
graph3 = ax.bar(x + width, cl_mu_vals, width, label='cl_mu', color=pendlegreen, edgecolor='black', alpha=0.8)

ax.set_ylabel("AUT", fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels(metrics_label, fontsize=16)
ax.set_ylim(0, 1.1)
def add_value_labels(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11)

add_value_labels(graph1)
add_value_labels(graph2)
add_value_labels(graph3)
ax.legend(fontsize=14)
ax.set_xlim(-0.8, 2.8)
plt.tight_layout()
plt.show()
'''

with open("dsets.pickle", "rb") as f:
    dset = pickle.load(f)

t = pd.to_datetime(dset['Date'])
y = dset['Label'].values
X = dset.drop(columns=['Date', 'Label']).values
fixed_start_date = pd.to_datetime("2016-09-01")
train_size = 2

splits = temporal.time_aware_train_test_split(X, y, t, train_size=train_size, test_size=1, granularity="month", start_date=fixed_start_date)

X_train, X_tests, y_train, y_tests, t_train, t_tests = splits
X_tests, y_tests, t_tests = clean_dsets(X_tests, y_tests, t_tests)
X_train, y_train, t_train = spatial.downsample_set(X_train, y_train, t_train.values, min_pos_rate=1/2)

print(len(np.where(y_train == 0)[0]), len(np.where(y_train == 1)[0]))

for i, (X_test, y_test, t_test) in enumerate(zip(X_tests, y_tests, t_tests), 1):

    K = y_test.shape[0] // 2
    y_test = y_test[:K]
    print(len(np.where(y_test == 0)[0]), len(np.where(y_test == 1)[0]))