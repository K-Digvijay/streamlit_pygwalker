import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score

df = pd.read_csv(r'D:\Walker_streamlit\diamond_data.csv')
class Process:
    def __init__(self, df):
        self.df = df
    @staticmethod
    def Clean(df):
        # df = df.drop_duplicates()
        df = df.dropna()
        df = pd.get_dummies(df,drop_first=True)
        return df
    @staticmethod
    def Train(df):
        X = df.drop(columns=['price'],axis=1)
        y = df['price']
        return X, y
    
    @staticmethod
    def Model(X, y):
        model = LinearRegression()
        model.fit(X, y)
        return model
    
    @staticmethod
    def Predict(model, X):
        return model.predict(X)
    
    @staticmethod
    def Evaluate(model, X, y):
        return model.score(X, y)
    
    @staticmethod
    def Scores(model, X, y):
        # y_pred already predicted in Predict() method
        y_pred = model.predict(X)
        rmse = np.sqrt(np.mean((y - y_pred) ** 2))
        r2 = r2_score(y, y_pred)
        return rmse, r2
    
def BoxPlot(model, X, y):
    sns.boxplot(X)
    plt.xticks(rotation=90)
    plt.show()
    
# def Plotting(model, X, y):
#     plt.plot(X, y, color='blue', label='Actual Price')
#     plt.plot(X, model.predict(X), color='red', label='Predicted Price')
#     plt.xlabel('Title')
#     plt.ylabel('Price')
#     plt.legend()
#     plt.show()

def main():
    df = pd.read_csv(r'D:\Walker_streamlit\diamond_data.csv')
    df = Process.Clean(df)
    X, y = Process.Train(df)
    model = Process.Model(X, y)
    y_pred = Process.Predict(model, X)
    rmse, r2 = Process.Scores(model, X,y)
    BoxPlot(model, X, y)
    print(rmse,r2)

if __name__ == '__main__':
    main()












