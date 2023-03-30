# %%
import pandas as  pd
teams = pd.read_csv (r'teams.csv')
teams

# %%
teams=teams[["team","country","year","athletes","age","prev_medals","medals"]]
teams

# %%
teams.corr()["medals"]

# %%
import seaborn as sns


# %%
sns.lmplot(x="athletes",y="medals",data=teams,fit_reg=True,ci=None)

# %%
sns.lmplot(x="age",y="medals",data=teams,fit_reg=True,ci=None)

# %%
teams.plot.hist(y="medals")

# %%
teams[teams.isnull().any(axis=1)]

# %%
teams=teams.dropna()
teams

# %%
train = teams[teams["year"]<2012].copy()
test = teams[teams["year"]>=2012].copy()


# %%
print(train.shape)
print(test.shape)

# %%
from sklearn.linear_model import LinearRegression

reg = LinearRegression()

# %%
predictors=["athletes","prev_medals"]
target="medals"

# %%
reg.fit(train[predictors].values,train[target].values)

# %%
predictions=reg.predict(test[predictors].values)

# %%
predictions

# %%
test["predictions"]=predictions
test

# %%
test.loc[test["predictions"] < 0, "predictions"]=0
test["predictions"]=test["predictions"].round()
test

# %%
from statistics import mean
from sklearn.metrics import mean_absolute_error
error = mean_absolute_error(test["medals"],test["predictions"])
error

# %%
teams.describe()["medals"]
# error should be less than standard deviation

# %%
test[test["team"]=="USA"]

# %%
test[test["team"]=="IND"]

# %%
errors=(test["medals"]-test["predictions"]).abs()
errors

# %%
error_by_team=errors.groupby(test["team"]).mean()
error_by_team

# %%
medals_by_team=test["medals"].groupby(test["team"]).mean()
error_ratio=error_by_team/medals_by_team
error_ratio

result=reg.predict([[53,10]])
print(result[0].round())

from joblib import dump,load

dump(reg,"model.joblib")

