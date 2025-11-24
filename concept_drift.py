import pandas as pd
from sklearn.metrics import precision_score, f1_score, recall_score, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from imblearn.over_sampling import SMOTE

from tesseract import spatial, temporal, evaluation, metrics

def standard_rf(past_dset:pd.DataFrame, future_dset:pd.DataFrame, test_size):
    '''
    Random Forest standard. È presente una soglia che delimita dati passati (con il quale si effettua il training) e dati futuri.
    Per i dati futuri viene fatto un downsampling 20:1 di negativi/positivi per renderli più realistici (altrimenti la stragrande maggioranza sarebbero dati positivi).
    '''

    print("PAST_VALUES...")
    results = {"Accuracy": [], "Precision": [], "Recall": [], "F1": []}

    y = past_dset['Label'].values
    X = past_dset.drop(columns=['Date', 'Label']).values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

    model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)

    print(f"Accuracy Score: {accuracy}")
    print(f"Precision Score: {precision}")
    print(f"Recall Score: {recall}")
    print(f"F1 Score: {f1}")

    results["Accuracy"].append(accuracy)
    results["Precision"].append(precision)
    results["Recall"].append(recall)
    results["F1"].append(f1)

    print("FUTURE VALUES...")
    t = pd.to_datetime(future_dset['Date'])
    y = future_dset['Label'].values
    X = future_dset.drop(columns=['Date', 'Label']).values
    X, y, t = spatial.downsample_set(X, y, t.values, min_pos_rate=1/21)

    pred = model.predict(X)

    accuracy = accuracy_score(y, pred)
    precision = precision_score(y, pred)
    recall = recall_score(y, pred)
    f1 = f1_score(y, pred)

    print(f"Accuracy Score: {accuracy}")
    print(f"Precision Score: {precision}")
    print(f"Recall Score: {recall}")
    print(f"F1 Score: {f1}")

    results["Accuracy"].append(accuracy)
    results["Precision"].append(precision)
    results["Recall"].append(recall)
    results["F1"].append(f1)

    return results

def rf_downsampling(past_dset: pd.DataFrame, future_dset:pd.DataFrame, test_size):
    '''
    La differenza qua sta nel downsampling dei dati di training per renderli in linea con quelli di testing.
    '''
    print("PAST_VALUES...")
    results = {"Accuracy": [], "Precision": [], "Recall": [], "F1": []}

    t = pd.to_datetime(past_dset['Date'])
    y = past_dset['Label'].values
    X = past_dset.drop(columns=['Date', 'Label']).values

    print(f"Rateo negativi/positivi: {np.count_nonzero(y == 0)} / {np.count_nonzero(y == 1)}")
    print("DOWNSAMPLING...")
    X, y, t = spatial.downsample_set(X, y, t.values, min_pos_rate=1/21)
    print(f"Rateo negativi/positivi: {np.count_nonzero(y == 0)} / {np.count_nonzero(y == 1)}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

    model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)

    print(f"Accuracy Score: {accuracy}")
    print(f"Precision Score: {precision}")
    print(f"Recall Score: {recall}")
    print(f"F1 Score: {f1}")

    results["Accuracy"].append(accuracy)
    results["Precision"].append(precision)
    results["Recall"].append(recall)
    results["F1"].append(f1)

    print("FUTURE VALUES...")
    t = pd.to_datetime(future_dset['Date'])
    y = future_dset['Label'].values
    X = future_dset.drop(columns=['Date', 'Label']).values

    X, y, t = spatial.downsample_set(X, y, t.values, min_pos_rate=1/21)
    
    pred = model.predict(X)

    accuracy = accuracy_score(y, pred)
    precision = precision_score(y, pred)
    recall = recall_score(y, pred)
    f1 = f1_score(y, pred)

    print(f"Accuracy Score: {accuracy}")
    print(f"Precision Score: {precision}")
    print(f"Recall Score: {recall}")
    print(f"F1 Score: {f1}")

    results["Accuracy"].append(accuracy)
    results["Precision"].append(precision)
    results["Recall"].append(recall)
    results["F1"].append(f1)

    return results

def rf_oversampling(past_dset: pd.DataFrame, future_dset: pd.DataFrame, test_size):
    '''
    Qua i dati di training vengono oversamplati per trainare il modello su tutti i dati disponibili.
    '''
    print("PAST_VALUES...")
    results = {"Accuracy": [], "Precision": [], "Recall": [], "F1": []}

    t = pd.to_datetime(past_dset['Date'])
    y = past_dset['Label'].values
    X = past_dset.drop(columns=['Date', 'Label']).values
    
    print(f"Rateo negativi/positivi: {np.count_nonzero(y == 0)} / {np.count_nonzero(y == 1)}")
    print("OVERSAMPLING...")
    oversample = SMOTE(k_neighbors=3, random_state=42)
    X, y = oversample.fit_resample(X, y)
    print(f"Rateo negativi/positivi: {np.count_nonzero(y == 0)} / {np.count_nonzero(y == 1)}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

    model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)

    print(f"Accuracy Score: {accuracy}")
    print(f"Precision Score: {precision}")
    print(f"Recall Score: {recall}")
    print(f"F1 Score: {f1}")

    results["Accuracy"].append(accuracy)
    results["Precision"].append(precision)
    results["Recall"].append(recall)
    results["F1"].append(f1)

    print("FUTURE VALUES...")
    t = pd.to_datetime(future_dset['Date'])
    y = future_dset['Label'].values
    X = future_dset.drop(columns=['Date', 'Label']).values

    X, y, t = spatial.downsample_set(X, y, t.values, min_pos_rate=1/21)

    pred = model.predict(X)

    accuracy = accuracy_score(y, pred)
    precision = precision_score(y, pred)
    recall = recall_score(y, pred)
    f1 = f1_score(y, pred)

    print(f"Accuracy Score: {accuracy}")
    print(f"Precision Score: {precision}")
    print(f"Recall Score: {recall}")
    print(f"F1 Score: {f1}")

    results["Accuracy"].append(accuracy)
    results["Precision"].append(precision)
    results["Recall"].append(recall)
    results["F1"].append(f1)

    return results

def rf_periodic_drift(dset: pd.DataFrame, test_size):
    '''
    Come nei precedenti, il training viene effettuato fino ad una soglia, la differenza sta nel testing:
    Il modello viene testato su singoli mesi o coppie, per visualizzare un degradamento nel tempo.
    '''
    results = {"Precision": [], "Recall": [], "F1": [], "TNR": [], "TPR": [], "Date": []}

    t = pd.to_datetime(dset['Date'])
    y = dset['Label'].values
    X = dset.drop(columns=['Date', 'Label']).values

    # X, y, t = spatial.downsample_set(X, y, t.values, min_pos_rate=1/21)

    start_date_fixed = pd.to_datetime("2016-09-08")
    # train_size=8 arriva ad Aprile compreso, partendo da Settembre 2016
    splits = temporal.time_aware_train_test_split(X, y, t, train_size=8, test_size=1, granularity="month", start_date=start_date_fixed)
    X_train, X_tests, y_train, y_tests, t_train, t_tests = splits
    
    # Rimozione mesi in cui non ci sono dati
    X_tests_puliti = []
    y_tests_puliti = []
    t_tests_puliti = []
    for i, x_test in enumerate(X_tests):
        if x_test.shape[0] > 0:
            X_tests_puliti.append(x_test)
            y_tests_puliti.append(y_tests[i])
            t_tests_puliti.append(t_tests[i])

    # Per avere dataset con più dati e il maggior numero di casi di test, divido in Maggio, Giugno/Luglio, Agosto/Settembre, Aprile/Maggio 2018
    # Per farlo, i test saranno: 8 Maggio - 7 Giugno, 8 Giugno - 7 Agosto, 8 Agosto - 7 Settembre, 8 Aprile - 7 Maggio 2018
    # Unisco i mesi Giugno e Luglio perché avrei pochi dati normali altrimenti.

    X_optimized_tests = []
    y_optimized_tests = []
    t_optimized_tests = []
    
    i = 0
    while i < len(X_tests_puliti):
        if i == 1:
            X_optimized_tests.append(np.concatenate([X_tests_puliti[i], X_tests_puliti[i+1]]))
            y_optimized_tests.append(np.concatenate([y_tests_puliti[i], y_tests_puliti[i+1]]))
            t_optimized_tests.append(pd.concat([t_tests_puliti[i], t_tests_puliti[i+1]]))
            i += 2
        else:
            X_optimized_tests.append(X_tests_puliti[i])
            y_optimized_tests.append(y_tests_puliti[i])
            t_optimized_tests.append(t_tests_puliti[i])
            i += 1
    
    # Ottengo le date dei test splittati
    results["Date"].append("4 - 2017")
    for t in t_optimized_tests:
        date_end = t.iloc[0]
        date_start = t.iloc[-1]
        if date_start.month < date_end.month:
            results["Date"].append(f"{date_start.month}/{date_end.month} - {date_end.year}")
        elif date_start.month > date_end.month:
            results["Date"].append(f"{date_end.month}/{date_start.month} - {date_end.year}")
        else:
            results["Date"].append(f"{date_end.month} - {date_end.year}")
    print(results['Date'])
    
    # Downsampling dati di training
    X_train, y_train, t_train = spatial.downsample_set(X_train, y_train, t_train.values, min_pos_rate=1/2)
    
    for i, (x_i, y_i, t_i) in enumerate(zip(X_optimized_tests, y_optimized_tests, t_optimized_tests)):
        x_i, y_i, t_i = spatial.downsample_set(x_i, y_i, t_i.values, min_pos_rate=1/21)
        X_optimized_tests[i] = x_i
        y_optimized_tests[i] = y_i
        t_optimized_tests[i] = t_i

    splits = (X_train, X_optimized_tests, y_train, y_optimized_tests, t_train, t_optimized_tests)

    clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    preds = evaluation.fit_predict_update(clf, *splits)
    metrics.print_metrics(preds)

    results["Precision"] = preds['precision']
    results["Recall"] = preds['recall']
    results["F1"] = preds['f1']
    
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=test_size, random_state=42, stratify=y_train)

    model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)

    print(f"Precision Score: {precision}")
    print(f"Recall Score: {recall}")
    print(f"F1 Score: {f1}")

    true_negatives, false_positives, false_negatives, true_positives = preds["tn"], preds["fp"], preds["fn"], preds["tp"]
    tnr = []
    tpr = []
    for tn, fp, fn, tp in zip(true_negatives, false_positives, false_negatives, true_positives):
        if tn != "nan":
            tnr.append(float(tn / (tn + fp)))
        if tp != "nan":
            tpr.append(float(tp / (tp + fn)))
    
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel().tolist()
    tnr.insert(0, float(tn / (tn + fp)))
    tpr.insert(0, float(tp / (tp + fn)))

    results["TNR"] = tnr
    results["TPR"] = tpr
    results["Precision"].insert(0, precision)
    results["Recall"].insert(0, recall)
    results["F1"].insert(0, f1)
            
    return results

def rf_monthly(dset: pd.DataFrame, test_size):
    '''
    Il training viene sempre effettuato fino ad una soglia (con sia dati normali che malevoli). Il testing viene effettuato su ogni mese contenente i dati malevoli e utilizzeremo la seguente metrica:
    - True Positive Rate: TPR = tp/(tp+fn)
    '''

    t = pd.to_datetime(dset['Date'])
    y = dset['Label'].values
    X = dset.drop(columns=['Date', 'Label']).values

    start_date_fixed = pd.to_datetime("2016-09-1")
    # Train_size = 8 significa training fino ad Aprile
    splits = temporal.time_aware_train_test_split(X, y, t, train_size=8, test_size=1, granularity="month", start_date=start_date_fixed)

    X_train, X_tests, y_train, y_tests, t_train, t_tests = splits
    
    # Rimozione mesi in cui non ci sono dati
    X_tests_puliti = []
    y_tests_puliti = []
    t_tests_puliti = []
    for i, x_test in enumerate(X_tests):
        if x_test.shape[0] > 0:
            X_tests_puliti.append(x_test)
            y_tests_puliti.append(y_tests[i])
            t_tests_puliti.append(t_tests[i])
    
    # Downsample dei dati di training
    X_train, y_train, t_train = spatial.downsample_set(X_train, y_train, t_train.values, min_pos_rate=1/21)
    
    splits = (X_train, X_tests_puliti, y_train, y_tests_puliti, t_train, t_tests_puliti)
    clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    preds = evaluation.fit_predict_update(clf, *splits)
    metrics.print_metrics(preds)
    
    # Ottengo i valori di True Negative, True Positive, False Negative e False Positive
    tn_, fp_, fn_, tp_ = preds["tn"], preds["fp"], preds["fn"], preds["tp"]
    tpr = []
    for tn, fp, fn, tp in zip(tn_, fp_, fn_, tp_):
        if tp != "nan":
            tpr.append(float(tp / (tp + fn)))
    
    # Controllo le performance anche sul dataset di training
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=test_size, random_state=42, stratify=y_train)
    model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)
    train_pred = model.predict(X_test)
    recall = recall_score(y_test, train_pred)
    tpr.insert(0, recall)

    # Ottengo le date dei vari test per inserirle nell'asse x dei grafici
    test_dates = []
    test_dates.append("5 - 2017")
    for t in t_tests_puliti:
        date = pd.Timestamp(t.iloc[0])
        test_dates.append(f"{date.month} - {date.year}")
    
    
    return {'TPR': tpr, 'Date': test_dates}

