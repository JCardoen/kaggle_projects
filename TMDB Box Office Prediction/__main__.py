import pandas as pd
import matplotlib.pyplot as plot
import numpy as np
from scipy import stats
from sklearn import linear_model
from sklearn.metrics import r2_score

if __name__ == '__main__':
    df = pd.read_csv('train.csv', delimiter=',', header=0)
    # print(df.head())
    # print(df.shape)
    # print(df.describe())
    # Check linearity
    plot.scatter(df['budget'], df['revenue'])
    plot.show()
    # Correlation check
    '''
    corr = []

    for column in df:
        if column != "id":
            if df[column].dtype == np.float64 or df[column].dtype == np.int64:
                print("Currently checking column: %s" % column)
                print("------------------------------------------------------")
                pearsonCorr = df[column].corr(df['revenue'], method='pearson')
                kendallCorr = df[column].corr(df['revenue'], method='kendall')
                spearmanCorr = df[column].corr(df['revenue'], method='spearman')
                print("Pearson corr of %s with %f" % (column, pearsonCorr))
                print("Kendall corr of %s with %f" % (column, kendallCorr))
                print("Spearman corr of %s with %f" % (column, spearmanCorr))

                pearsonStatsCorr = stats.pearsonr(df[column], df['revenue'])
                spearmanStatscorr = stats.spearmanr(df[column], df['revenue'])
                kendallStatsCor = stats.kendalltau(df[column], df['revenue'])
                print("Scipy STATS Pearson Correlation of: %s and revenue = %s" % (column, pearsonStatsCorr))
                print("Scipy STATS kendallTau Correlation of: %s and revenue = %s" % (column, kendallStatsCor))
                print("Scipy STATS Spearman Correlation of: %s and revenue = %s" % (column, spearmanStatscorr))
                print("------------------------------------------------------")
    '''
    msk = np.random.rand(len(df)) < 0.8
    train = df[msk]
    test = df[~msk]

    # Create test and train dataset
    train_x = np.asanyarray(train[['budget']])
    train_y = np.asanyarray(train[['revenue']])

    # Create model
    model = linear_model.LinearRegression()
    model.fit(train_x, train_y)

    # The coefficients, theta 0 and theta 1
    print('Coefficients: ', model.coef_)  # theta 0
    print('Intercept: ', model.intercept_)  # theta 1: slope or gradient

    # Model evaluation
    test_x = np.asanyarray(test[['budget']])
    test_y = np.asanyarray(test[['revenue']])
    test_y_ = model.predict(test_x)

    # Let's make a prediction
    # If the engine-size would be 5, what would be our emision?
    prediction = model.predict(np.array([[8000000]]))
    print("Prediction where budget would be 8m is %.2f" % prediction)

    print(
        "MAE: %.2f" % np.mean(
            np.absolute(test_y_ - test_y)))
    print(
        "MSE : %.2f" % np.mean(
            (test_y_ - test_y) ** 2))
    print(
        "R2 : %.2f" % r2_score(
            test_y_, test_y))

    df = pd.read_csv('test.csv', delimiter=',', header=0)

    appendingDf = pd.DataFrame()

    revs = []
    for i, row in df.iterrows():
        budget = row['budget']
        prediction = model.predict(np.array([[budget]]))
        revs.append(prediction[0][0])

    df['revenue'] = revs
    df.to_csv('solution.csv', columns=['id', 'revenue'], index=0)






