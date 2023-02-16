import yfinance as yf

#Importing and getting to know GOOGL and having information

googl = yf.Ticker("GOOGL")
googl = googl.history(period="max")
#print(googl)

#print(googl.index) #he index() method returns the position at the first occurrence of the specified value

#print(googl.describe())

#print(googl.info)

googl["Tomorrow"] = googl["Close"].shift(-1) #Adds column that shows tomorrow's price
#print(googl)

googl["Target"] = (googl["Tomorrow"] > googl["Close"]).astype(int) #what we're trying to predict, gives 1 if price goes up
#print(googl)

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=300, min_samples_split=200, random_state=1)

train = googl.iloc[:100]
test = googl.iloc[-100:]

predictors = ["Close", "Volume", "Open", "High", "Low"]
model.fit(train[predictors], train["Target"])

from sklearn.metrics import precision_score
import pandas as pd

preds = model.predict(test[predictors])
preds = pd.Series(preds, index=test.index)
#print(precision_score(test["Target"], preds))

combined = pd.concat([test["Target"], preds], axis=1)
#print(combined.plot())

def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict(test[predictors])
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined

def backtest(data, model, predictors, start=1500, step=250):
    all_predictions = []

    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)
        return pd.concat(all_predictions)

predictions = backtest(googl, model, predictors)
#print(predictions["Predictions"].value_counts())

#print(precision_score(predictions["Target"], predictions["Predictions"]))

#print(predictions["Target"].value_counts() / predictions.shape[0])

horizons = [2, 5, 60, 250, 750]
new_predictors = []

for horizon in horizons:
    rolling_averages = googl.rolling(horizon).mean()

    ratio_column = f"Close_ratio_{horizon}"
    googl[ratio_column] = googl["Close"] / rolling_averages["Close"]

    trend_column = f"Trend_{horizon}"
    googl[trend_column] = googl.shift(1).rolling(horizon).sum()["Target"]

    new_predictors += [ratio_column, trend_column]

#print(googl)

googl = googl.dropna() #getting rif of NaNs
#print(googl)

model =RandomForestClassifier(n_estimators=400, min_samples_split=100, random_state=1)

def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict_proba(test[predictors])[:,1]
    preds[preds >= .6] = 1
    preds[preds < .6] = 0
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined

predictions = backtest(googl, model, new_predictors)
#print(predictions["Predictions"].value_counts())

print(precision_score(predictions["Target"], predictions["Predictions"]))

