from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler   # For MultinomialNB
from pandas import read_csv, concat


class ProcessData():
    def __init__(self, target_column, model_index) -> None:

        # Targeted column
        self.TARGET_COL = target_column

        # 2 rows for a real test
        self.test = []
        
        self.index = model_index

    
    # To read the database AKA csv
    def read_csv(self, path):
        try:
            return read_csv(path)
        
        except Exception as e:
            print(e)
            return None
        

    # Even the data so for all categories there the same number of entries 
    def even_data(self, df):

        # Phishing websites
        df_1 = df[df[self.TARGET_COL] == 1]

        # Real websites
        df_0 = df[df[self.TARGET_COL] == 0]   

        # # FOR REAL TEST
        # df_0, df_1 = self.get_real_test_data(df_0, df_1)

        if len(df_1) == len(df_0): return df

        # Even it
        if len(df_1) > len(df_0): df_1 = df_1[:len(df_0)]
        elif len(df_0) > len(df_1): df_0 = df_0[:len(df_1)]

        # For each targeted value the same number of rows
        return concat([df_1, df_0]) 


    # Da v features (X) in v target variable (y)
    def features_and_labels(self, df):

        # Phishing websites
        df_1 = df[df[self.TARGET_COL] == 1]

        # Real websites
        df_0 = df[df[self.TARGET_COL] == 0]   

        # FOR REAL TEST
        df_0, df_1 = self.get_real_test_data(df_0, df_1)

        # PUT DATAFRAME BACKTOGETHER
        df = concat([df_1, df_0]) 

        X = df.drop(self.TARGET_COL, axis = 1)
        y = df[self.TARGET_COL]
            
        return X, y
    


    # For training and testing
    def train_test_data(self, x, y, selected_model, test_size = 0.25):
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size)


        # Ni negativnih vrednosti
        if selected_model == 3:
            scaler = MinMaxScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)


        return  X_train, X_test, y_train, y_test
    


    def get_real_test_data(self, df_0, df_1):

        # Last 5 rows of df
        c0 = len(df_0)
        last_rows0 = df_0.iloc[c0-5:]

        c1 = len(df_1)
        last_rows1 = df_1.iloc[c1-5:]


        # Targeted values
        self.val = []
        for i in range(5): self.val.append(int(last_rows0.iloc[i].get(self.TARGET_COL)))
        for j in range(5): self.val.append(int(last_rows1.iloc[j].get(self.TARGET_COL)))


        # Save to global variable test (X) - features
        if self.index != 6:
            for i in range(5): self.test.append(last_rows0.iloc[i].drop(self.TARGET_COL))
            for j in range(5): self.test.append(last_rows1.iloc[j].drop(self.TARGET_COL))

        elif self.index == 6:
            for i in range(5): self.test.append(last_rows0.iloc[i].drop(self.TARGET_COL).to_numpy())
            for j in range(5): self.test.append(last_rows1.iloc[j].drop(self.TARGET_COL).to_numpy())
           

        # Remove the last 5 rows
        df_0 = df_0.iloc[:c0-5]
        df_1 = df_1.iloc[:c1-5]

        return df_0, df_1