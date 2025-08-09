# FOR 2D ONLY!!!
# My imports
from models import Model
from process_data import ProcessData
from utils import stats_to_json, binary_transform

# Normal imports
from numpy import array
from sklearn.preprocessing import MinMaxScaler 
from time import sleep
import pandas as pd


def main(LOAD_MODEL, MODEL_PATH, TO_EVEN_DATA, TEST_RATIO, SELECTED_MODEL):
    
    MODELS = ["RandomForestClassifier","LinearSVC", "KNeighborsClassifier","MultinomialNB", "LogisticRegression", "SVC", "Convolutional Neural Network", "Multi-layer Perceptron"]
    
    # For quick tests
    check_predicton_values = False
    if TEST_RATIO <= 0.0005: check_predicton_values = True
    
    if LOAD_MODEL:
        # FIND WHICH MODEL IS BEING LOADED
        for i in range(len(MODELS)):
            if MODELS[i] in MODEL_PATH:
                SELECTED_MODEL = i
                break


    # DATA PROCESSING
    process_data = ProcessData(target_column="phishing", model_index=SELECTED_MODEL)

    # READ FROM CSV
    df = process_data.read_csv("Dataset//dataset.csv")

    # IF .CSV EVEN EXISTS
    if df is not None:

        # Drops duplicates and null values (če je kje v rowu prazn stolpec)
        df = df.drop_duplicates()
        df = df.dropna()

        # Normalizira vrednosti v pd (boljše za ene modele - al v času treniranja/napovedovanja ali pa v accuracy) - med 0 in 1
        if SELECTED_MODEL in [1, 2, 5, 6]:
            scaler = MinMaxScaler()
            df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

        # EVEN THE DATA
        if TO_EVEN_DATA: df = process_data.even_data(df)

        # GET features and labels
        x, y = process_data.features_and_labels(df) 

        # Training and testing
        X_train, X_test, y_train, y_test = process_data.train_test_data(x, y, SELECTED_MODEL, test_size=TEST_RATIO)

        # FOR SEPERATING MULTIPLE RUNS FOR BETTER CLARITY
        print("_________________________________________________________")
        

        # Neural Network likes numpy arrays
        if SELECTED_MODEL == 6:
            X_train, y_train = X_train.to_numpy(), y_train.to_numpy()
            X_test, y_test = X_test.to_numpy(), y_test.to_numpy()

        # MODEL
        model = Model(SELECTED_MODEL, LOAD_MODEL=LOAD_MODEL)

        if LOAD_MODEL: model.load_saved_model(MODEL_PATH)
        else: 
            # TRAINING MODEL
            model.train(X_train, y_train)



        print(f"DATA: {"EVEN" if TO_EVEN_DATA else "NOT EVEN"}!")

        # MAKING PREDICTION
        prediction = model.predict_with_time(X_test)


        # IF TEST DAT % IS LOW IT CAN SHOW REAL AND PREDICTION VALUES 
        if check_predicton_values:
            print("Real:  ", end=" ")
            for i in y_test: print(i, end=" ")
            print()
            print("Pred: ",prediction)


        # Metrics - {...}
        metrics = model.evaluate_all_metrics(y_test, prediction, TO_EVEN_DATA)

    
        # REALLLLLL TESTTTTTTTT
        # REALLLLLL TESTTTTTTTT
        # REALLLLLL TESTTTTTTTT

        # 10 EXAMPLES FOR REAL TEST: 0 - legit website, 1 - phishing website
        test_x = process_data.test

        # To np.array SO NEURAL NETWORK CAN WORK WITH IT
        if SELECTED_MODEL == 6: test_x = array(test_x)

        # PREDICT
        result = model.predict(test_x)

        # FOR OTHER MODELS
        if SELECTED_MODEL != 6: 
            real_values_test = process_data.val
            prediction_test = str(result).replace(" ", "").replace("[", "").replace("]", "").replace(".", "")
            print("PRED:", prediction_test)
            prediction_test = [int(out) for out in prediction_test]

            print(f"Mini test: {str(process_data.val).replace(",", "")}, Prediction: {result}")


        # 0 OR 1 FOR NEURAL NETWORK
        elif SELECTED_MODEL == 6:
            prediction_test = binary_transform(result)
            real_values_test = process_data.val

            print(f"Mini test: {str(real_values_test).replace(",", "")}, Prediction: {str(prediction_test).replace(",", "")}")
            
            

        # ADDING PREDICTION AND REAL VALUES OF REAL TEST
        metrics["real_values_test"] = real_values_test
        metrics["prediction_test"] = prediction_test



        # LAST THING SAVE FILES AND JSON!!
        model.save(metrics, x.columns)

        # STATS TO JSON
        stats_to_json(metrics, model.MODELS[model.index], TEST_RATIO)


        # FOR SEPERATING MULTIPLE RUNS FOR BETTER CLARITY
        print("_________________________________________________________")
        print()






if __name__ == "__main__":

    # LOAD MODEL
    LOAD_MODEL = False
    MODEL_PATH  = "saved_models\\2024-09-19 14-58-41 Convolutional Neural Network\\Convolutional Neural Network.keras"

    # Even the data so len(category1) == len(category2)
    TO_EVEN_DATA = True

    # The size of test dataset in % - 0.23
    TEST_RATIO = 0.125

    # FOR SELECTED MODEL
    # RandomForestClassifier: 0, LinearSVC: 1, KNeighborsClassifier: 2, 
    # MultinomialNB: 3, LogistiCRegression: 4, SVC: 5, 
    # Neural Network: 6, Multi-layer Perceptron : 7
    SELECTED_MODEL = 0

    
    # FOR ONE MODEL
    main(LOAD_MODEL, MODEL_PATH, TO_EVEN_DATA, TEST_RATIO, SELECTED_MODEL)

