import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# MODELS
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.svm import SVC
from keras import layers, models, Sequential

# ACCUARCY
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# NORMAL IMPORTS
from datetime import datetime
import time
import pickle
import re
import os
import numpy as np
from warnings import filterwarnings
filterwarnings("ignore", category=ConvergenceWarning)
filterwarnings("ignore", category=UserWarning)

# FOR utils_nn
import sys
sys.path.append("//")
from utils import binary_transform


class Model:
    def __init__(self, index, LOAD_MODEL = False):
        # Avaible models
        self.MODELS = ["RandomForestClassifier","LinearSVC", "KNeighborsClassifier","MultinomialNB", "LogisticRegression", "SVC",
                        "Convolutional Neural Network", "Multi-layer Perceptron"]

        # So we know which model is selected
        self.index = index

        # Model variable
        self.model = None

        # For timing
        self.start_time_train = 0
        self.end_time_train = 0
        self.start_time_prediction = 0
        self.end_time_prediction = 0

        print()
        print(f"Selected Model is {self.MODELS[self.index]} ({self.index})!")

        # Folder to save model and INFO
        date = datetime.today()
        self.DIR = f'saved_models//{date.strftime("%Y-%m-%d %H-%M-%S")} {self.MODELS[self.index]}'
        self.model_path = f"{date.strftime("%Y-%m-%d %H-%M-%S")} {self.MODELS[self.index]}"
        os.mkdir(self.DIR)

        # FOR NEURAL NETWORK
        self.vs = 0.25
        self.epochs = 10
        self.batch = 128

        # For LOAD MODEL - training time
        self.LOAD_MODEL = LOAD_MODEL


    def train(self, x, y):
        self.start_time_train = time.time()
        
        # RandomForestClassifier
        if self.index == 0:
            self.model = RandomForestClassifier(n_estimators=200, max_features=15, random_state=2, n_jobs= -1)
            
        # LinearSVC
        elif self.index == 1:
            self.model = LinearSVC(max_iter= 1800)

        # KNeighborsClassifier
        elif self.index == 2:    
            self.model = KNeighborsClassifier(n_neighbors=3, weights= "uniform", n_jobs= -1)

        # MultinomialNB  - without negative values
        elif self.index == 3:
            self.model = MultinomialNB()

        # LogistiCRegression
        elif self.index == 4:
            self.model = LogisticRegression(penalty="l2", n_jobs= -1, max_iter = 150, solver="newton-cholesky")

        # SVC
        elif self.index == 5:
            self.model = SVC(random_state=42)

        # Convolutional Neural Network
        elif self.index == 6:
            self.model = Sequential([
                    #    X_train.shape[1]   
                    layers.Input((111, 1)),
                    layers.Conv1D(filters = 16,kernel_size = 3,activation = 'relu',padding = 'same'),
                    layers.Dropout(0.2),
                    layers.BatchNormalization(),
                    layers.MaxPooling1D(pool_size = 2,padding = 'same'),
                    layers.Conv1D(filters = 32,kernel_size = 3,activation = 'relu',padding = 'same'),
                    layers.Dropout(0.2),
                    layers.BatchNormalization(),
                    layers.MaxPooling1D(pool_size = 2,padding = 'same'),
                    layers.Conv1D(filters = 64,kernel_size = 3,activation = 'relu',padding = 'same'),
                    layers.Dropout(0.2),
                    layers.BatchNormalization(),
                    layers.MaxPooling1D(pool_size = 2,padding = 'same'),
                    layers.Conv1D(filters = 128,kernel_size = 3,activation = 'relu',padding = 'same'),
                    layers.Dropout(0.2),
                    layers.BatchNormalization(),
                    layers.MaxPooling1D(pool_size = 2,padding = 'same'),
                    layers.Conv1D(filters = 256,kernel_size = 3,activation = 'relu',padding = 'same'),
                    layers.Dropout(0.2),
                    layers.BatchNormalization(),
                    layers.LSTM(128,return_sequences=True),
                    layers.Dropout(0.3),
                    layers.Flatten(),
                    layers.Dense(128,activation = 'relu'),
                    layers.Dropout(0.3),
                    layers.Dense(128,activation = 'relu'),
                    layers.Dropout(0.3),
                    layers.Dense(1,activation = 'sigmoid'),
            ])  
        
            #Compile it
            self.model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

        # MLP Classifier
        elif self.index == 7:
            self.model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=150, random_state=2)


        # Train the model
        if self.index != 6:
            self.model.fit(x, y)

        elif self.index == 6:
            self.model.fit(x, y, validation_split=self.vs, batch_size=self.batch, epochs= self.epochs)

        self.end_time_train = time.time()

    # With timing
    def predict_with_time(self, x):
        
        self.start_time_prediction = time.time()

        result = self.model.predict(x)

        self.end_time_prediction = time.time()

        return result

    def predict(self, x): return self.model.predict(x)


    def evaluate_all_metrics(self, y, prediction, TO_EVEN_DATA):
        print()
        
        # FOR NN 
        if self.index == 6:
            arr = binary_transform(prediction)
            prediction = np.array(arr)

        # All the metrics for accuracy
        asc = accuracy_score(y, prediction)
        print(f"Accuracy Score: {asc}")

        rs = recall_score(y, prediction)
        print(f"Recall Score: {rs}")

        ps = precision_score(y, prediction)
        print(f"Precision Score: {ps}")

        f1 = f1_score(y, prediction)
        print(f"F1 Score: {f1}")

        # Time
        train_time = round(self.end_time_train - self.start_time_train, 8)
        predict_time = round(self.end_time_prediction - self.start_time_prediction, 8)

                
        # For training time
        if self.LOAD_MODEL: train_time = self.get_train_time_from_info()

        print("Time needed to train:", train_time, "seconds   //doesn't include time of preparing dataset")
        print("Time needed to predict:", predict_time, "seconds")
        print()

        return {"index": self.index, "asc": asc, "rs": rs, "ps": ps, "f1": f1, "train_time": train_time, "predict_time": predict_time, 
                "dir": self.model_path, "validation_split": self.vs, "epochs": self.epochs, "batch_size":self.batch, 
                "even_data": TO_EVEN_DATA}


    def save(self, metrics, col_names):

        # MODEL FILE
        if self.index != 6:
            filename = os.path.join(self.DIR, f'{self.MODELS[self.index]}.pickle')
            pickle.dump(self.model, open(filename, 'wb'))

        elif self.index == 6:
            filename = os.path.join(self.DIR, f'{self.MODELS[self.index]}.keras')
            self.model.save(filename)


        # IMPORTANT FEATURES
        data = self.important_features(col_names)


        # INFO
        text = f"""\n\n
                Accuracy Score: {metrics["asc"]}\n
                Recall Score: {metrics["rs"]}\n
                Precision Score: {metrics["ps"]}\n
                F1 Score: {metrics["f1"]}\n
                Time needed to train: {metrics["train_time"]}\n
                Time needed to predict: {metrics["predict_time"]}\n
                Data even: {metrics["even_data"]}\n

                {data}
                """
        text_path = os.path.join(self.DIR, "INFO.txt")
        with open(text_path, "w") as file:
            file.write(text)


        print()
        print(f"Saved to {self.DIR}")


    def load_saved_model(self, PATH):
        self.PATH = PATH
        print("Model Is Loaded")
        # Load the model
        if self.index != 6:
            with open(PATH, 'rb') as file:
                self.model = pickle.load(file)

        elif self.index == 6:
            self.model = models.load_model(PATH)
        


    # IMPORTANT FEATURES
    def important_features(self, col_name):

        data = ""

        #RFC
        if self.index == 0:
            features = zip(col_name, self.model.feature_importances_)

            features = sorted(features, key = lambda x: x[1], reverse= True)

            for i, j in features:
                data += f"\n\t\t{i}  =  {round(j*100, 4)}"

        #LSVC
        elif self.index == 1:
            features = zip(col_name, self.model.coef_[0])

            features = sorted(features, key = lambda x: x[1], reverse= True)

            for i, j in features:
                data += f"\n\t\t{i}  =  {round(j, 4)}"


        return data
    

    # If loading an existing model
    def get_train_time_from_info(self):

        # Remove the last part
        DIR = os.path.dirname(self.PATH)

        # Get INFO data
        with open(os.path.join(DIR, "INFO.txt"), 'r') as f:
            content = f.read()


        # Regular expression to find the time needed to train
        match = re.search(r"Time needed to train: (\d+\.\d+)", content)

        # If there is training time
        if match: return float(match.group(1))
        
        return 0