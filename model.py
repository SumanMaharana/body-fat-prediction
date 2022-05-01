import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle
data = pd.read_csv("bodyfat.csv")
X = data[["Abdomen","Chest","Hip","Weight","Thigh"]]
y = data["BodyFat"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
reg = LinearRegression()
reg.fit(X_train,y_train)
y_pred = reg.predict(X_test)
score = reg.score(X_test,y_test)
rmse = mean_squared_error(y_test,y_pred,squared=False)
pickle.dump(reg,open('reg.pkl','wb'))