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

# ------------------------- OLD ------------------------
# def rf_cl(dset: pd.DataFrame, test_size, botnet: str):
    results = {"Precision": [], "F1": [], "TNR": [], "TPR": [], "Date": []}
    t = pd.to_datetime(dset['Date'])
    y = dset['Label'].values
    X = dset.drop(columns=['Date', 'Label']).values
    X_columns = dset.drop(columns=['Date', 'Label']).columns
    fixed_start_date = pd.to_datetime("2016-09-08")
    train_size = 8

    splits = temporal.time_aware_train_test_split(X, y, t, train_size=train_size, test_size=1, granularity="month", start_date=fixed_start_date)
    
    X_train, y_train, t_train, X_optimized_tests, y_optimized_tests, t_optimized_tests = splits_handle(splits, botnet)
    calculate_dates(fixed_start_date, train_size, results, t_optimized_tests)

    # Downsample dati di training (per performance) e testing (troppi malware)
    if botnet == "all":
        X_train, y_train, t_train = spatial.downsample_set(X_train, y_train, t_train.values, min_pos_rate=1/2)

        for i, (x_i, y_i, t_i) in enumerate(zip(X_optimized_tests, y_optimized_tests, t_optimized_tests)):
            x_i, y_i, t_i = spatial.downsample_set(x_i, y_i, t_i.values, min_pos_rate=1/21)
            X_optimized_tests[i] = x_i
            y_optimized_tests[i] = y_i
            t_optimized_tests[i] = t_i
    elif botnet == "single":
        X_train, y_train, t_train = spatial.downsample_set(X_train, y_train, t_train.values, min_pos_rate=1/21)
    else:
        raise ValueError(f"botnet name '{botnet}' invalid")

    clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)

    print("Old data training...")
    X_train_past, X_test_past, y_train_past, y_test_past = train_test_split(X_train, y_train, test_size=test_size, random_state=42, stratify=y_train)

    clf.fit(X_train_past, y_train_past)
    pred = clf.predict(X_test_past)
    calculate_metrics(y_test_past, pred, results, botnet)

    # K = y_train.shape[0] // 2
    X_train_sliding, y_train_sliding = best_K_data(clf, X_train, y_train, botnet)

    # -------------------- CONTINUAL LEARNING --------------------    
    starttime = time.time()
    f = open("performances/tmp.txt", "a")
    print(f"Start time: {datetime.datetime.now().time()}", file=f)

    for i, (x_test, y_test) in enumerate(zip(X_optimized_tests, y_optimized_tests), 1):
        print(f"Cycle {i}")
        
        clf.fit(X_train_sliding, y_train_sliding)
        pred = clf.predict(x_test)
        # print(check_importances(clf, X_columns))
        calculate_metrics(y_test, pred, results, botnet)
        
        # K = y_test.shape[0] // 2
        X_retraining, y_retraining = best_K_data(clf, x_test, y_test, botnet)
        X_train_sliding = np.vstack((X_train_sliding, X_retraining))
        y_train_sliding = np.concatenate((y_train_sliding, y_retraining))

    endtime = time.time() - starttime
    print(f"Time taken: {endtime}", file=f)

    print_metrics(results, f)
    f.close()
    return results

# def ensemble_cl(dset: pd.DataFrame, test_size, botnet: str):
    results = {"Precision": [], "F1": [], "TNR": [], "TPR": [], "Date": []}
    t = pd.to_datetime(dset['Date'])
    y = dset['Label'].values
    X = dset.drop(columns=['Date', 'Label']).values
    X_columns = dset.drop(columns=['Date', 'Label']).columns
    fixed_start_date = pd.to_datetime("2016-09-08")
    train_size = 8
    ensemble_models = []
    weights = []

    splits = temporal.time_aware_train_test_split(X, y, t, train_size=train_size, test_size=1, granularity="month", start_date=fixed_start_date)
    
    X_train, y_train, t_train, X_optimized_tests, y_optimized_tests, t_optimized_tests = splits_handle(splits, botnet)
    calculate_dates(fixed_start_date, train_size, results, t_optimized_tests)

    # Downsample dati di training (per performance)
    if botnet == "all":
        X_train, y_train, t_train = spatial.downsample_set(X_train, y_train, t_train.values, min_pos_rate=1/2)
    elif botnet == "single":
        X_train, y_train, t_train = spatial.downsample_set(X_train, y_train, t_train.values, min_pos_rate=1/21)
    else:
        raise ValueError(f"botnet name '{botnet}' invalid")

    clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)

    print("Old data training...")
    X_train_past, X_test_past, y_train_past, y_test_past = train_test_split(X_train, y_train, test_size=test_size, random_state=42, stratify=y_train)

    clf.fit(X_train_past, y_train_past)
    ensemble_models.append(clf)
    weights.append(1)
    pred = clf.predict(X_test_past)
    calculate_metrics(y_test_past, pred, results, botnet)

    # all_probs = clf.predict_proba(X_test_past)
    # max_probs = np.max(all_probs, axis=1)
    # K = 2000
    # indexes = np.argsort(max_probs)[:K]

    # buffer_X = X_test_past[indexes]
    # buffer_y = y_test_past[indexes]


    if botnet == "single":
        mask_neg = (y_train == 0)
        # K = np.sum(y_train == 0) // 2
        X_neg, y_neg = best_K_data(clf, X_train[mask_neg], y_train[mask_neg], botnet)
    
    # -------------------- CONTINUAL LEARNING --------------------

    starttime = time.time()
    f = open("performances/tmp.txt", "a")
    print(f"Start time: {datetime.datetime.now().time()}", file=f)

    
    for i, (x_test, y_test) in enumerate(zip(X_optimized_tests, y_optimized_tests), 1):
        unlearning = {"Precision": [], "F1": [], "TNR": [], "TPR": [], "Date": []}
        print(f"Cycle {i}")
        pred, all_probs, avg_probs = ensemble_predict_weighted(ensemble_models, x_test, weights)
        # print(check_importances(clf, X_columns))
        calculate_metrics(y_test, pred, results, botnet)

        # Preparazione dati (nuovi + buffer) per nuovo modello dell'ensemble
        max_probs = np.max(avg_probs, axis=1)
        K = y_test.shape[0] // 2
        indexes = np.argsort(max_probs)[:K]

        X_new_ensemble = x_test[indexes]
        y_new_ensemble = y_test[indexes]
        
        # idx_neg = np.where(y_test == 0)[0]
        # idx_pos = np.where(y_test == 1)[0]
        # sel_neg = idx_neg[np.argsort(max_probs[idx_neg])[:1000]]
        # sel_pos = idx_pos[np.argsort(max_probs[idx_pos])[:1000]]
        # idx_buffer = np.concatenate([sel_neg, sel_pos])
        # new_X_buffer = x_test[idx_buffer]
        # new_y_buffer = y_test[idx_buffer]
        
        
        # X_train_mix = np.vstack((X_new_ensemble, buffer_X))
        # y_train_mix = np.concatenate((y_new_ensemble, buffer_y))
        # buffer_X, buffer_y = update_buffer(buffer_X, buffer_y, new_X_buffer, new_y_buffer, size=2000)

        weights = calculate_weights(y_test, all_probs, botnet)

        # -------------------- MACHINE UNLEARNING --------------------
        if botnet == "all":
            if i != 1:
                total_tpr = results['TPR'][-1]
                total_tnr = results['TNR'][-1]
                total_f1 = results['F1'][-1]
                for j in range(len(ensemble_models)):
                    models_left = ensemble_models[:j] + ensemble_models[j+1:]
                    weights_lef = weights[:j] + weights[j+1:]
                    pred, all_probs, avg_probs = ensemble_predict_weighted(models_left, x_test, weights_lef)
                    calculate_metrics(y_test, pred, unlearning, botnet)
                
                unlearning['F1'] = [el + (0.03 * (len(unlearning['F1']) - unlearning['F1'].index(el))) for el in unlearning['F1']]
                unlearning['TNR'] = [el + (0.03 * (len(unlearning['TNR']) - unlearning['TNR'].index(el))) for el in unlearning['TNR']]
                unlearning['TPR'] = [el + (0.03 * (len(unlearning['TPR']) - unlearning['TPR'].index(el))) for el in unlearning['TPR']]

                max_index = unlearning['F1'].index(max(unlearning['F1']))
                
                if unlearning['F1'][max_index] >= total_f1 and unlearning['TNR'][max_index] >= total_tnr and unlearning['TPR'][max_index] >= total_tpr:
                    del ensemble_models[max_index]
                    del weights[max_index]
                    print(f"Rimozione modello {max_index}")
            
            ensemble_models.append(RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42).fit(X_new_ensemble, y_new_ensemble))

        elif botnet == "single":
            if i != 1:
                total_tpr = results['TPR'][-1]
                for j in range(len(ensemble_models)):
                    models_left = ensemble_models[:j] + ensemble_models[j+1:]
                    weights_lef = weights[:j] + weights[j+1:]
                    pred, all_probs, avg_probs = ensemble_predict_weighted(models_left, x_test, weights_lef)
                    calculate_metrics(y_test, pred, unlearning, botnet)
                max_index = unlearning['TPR'].index(max(unlearning['TPR']))

                if unlearning['TPR'][max_index] >= total_tpr - 0.05:
                    del ensemble_models[max_index]
                    del weights[max_index]
            
            X_sliding = np.vstack((X_neg, x_test[indexes]))
            y_sliding = np.concatenate((y_neg, y_test[indexes]))
            ensemble_models.append(RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42).fit(X_sliding, y_sliding))

    endtime = time.time() - starttime
    print(f"Time taken: {endtime}", file=f)
    print(len(ensemble_models))
    print_metrics(results, f)
    f.close()
    return results