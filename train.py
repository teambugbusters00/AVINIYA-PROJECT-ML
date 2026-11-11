"""Train a reproducible tabular baseline from data.csv; target column defaults to target."""
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
p=argparse.ArgumentParser();p.add_argument("--data",default="data.csv");p.add_argument("--target",default="target");p.add_argument("--output",default="model.joblib");a=p.parse_args()
df=pd.read_csv(a.data)
if a.target not in df:raise SystemExit(f"Target column {a.target!r} not found.")
X=df.drop(columns=[a.target]);y=df[a.target];num=X.select_dtypes(include="number").columns;cat=X.columns.difference(num)
pre=ColumnTransformer([("num",Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler())]),num),("cat",Pipeline([("impute",SimpleImputer(strategy="most_frequent")),("encode",OneHotEncoder(handle_unknown="ignore"))]),cat)])
model=Pipeline([("preprocess",pre),("classifier",RandomForestClassifier(n_estimators=300,random_state=42,class_weight="balanced"))])
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y);model.fit(Xtr,ytr);print(classification_report(yte,model.predict(Xte)));joblib.dump(model,a.output);print("Saved",a.output)