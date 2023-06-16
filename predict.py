# you can download the csv file from https://drive.google.com/file/d/1q5jpI5M1EA9x3YPrLupmiu3gffkmGlHj/view
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import joblib
import re#The functions in this module allow you to determine whether a given text fits a given regular expression, known as a regular expression.
data=pd.read_csv("News.csv")
data=data.drop(["date","index"],axis=1)
#data.subject=data.subject.apply(lambda x:x.strip())
groupSubjects=data.groupby("subject")["subject"].agg("count").sort_values(ascending=True)
poorely_described_subjects=groupSubjects[groupSubjects<3]
data.subject=data.subject.apply(lambda x:"Other" if x in poorely_described_subjects else x)
data=data[["title","text","subject","class"]]
data=data.dropna()

#to identify whether image is fake or not its title and subjject are useless just focus on the text

data=data.drop(["title","subject"],axis=1)
def wordopt(data):
    data=data.lower()
    data=re.sub('\[.*?\]',"",data)#replace all words within bracket
    data=re.sub("\\W"," ",data)#replace \W
    data=re.sub("https?://\S+|www\.\S+"," ",data)#rreplace occurrences of URLs with whitespace in the `data` variable.
    data=re.sub("<.*?>+","",data)#replace  all the HTML tags within the `data` string will be replaced with an empty string, effectively removing them.
    data=re.sub("\n","",data)
    data=re.sub("\W*\d\w*","",data)#replace 
    return data
data["text"]=data.text.apply(wordopt)
data["text"]=data.text.apply(lambda x:x.strip())
x=data.text
y=data["class"]
#print(x.head())
#print(y.head())


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)



#vecterixe data to be trained
vecterizer=TfidfVectorizer()
x_train_vecterize=vecterizer.fit_transform(x_train)
#vecterixe data to be tested
x_test_vecterize=vecterizer.transform(x_test)

'''
model_params = {

    {
     "MultinomialNB":{ 
         'model':MultinomialNB(force_alpha=True),
         'params':{
            'alpha':[1.0,0.75,0.50,0.25,0.00],
         }
    },
    'random_forest': {
        'model': RandomForestClassifier(),
        'params' : {
            'n_estimators': [1,5,10]
        }
    },
    'logistic_regression' : {
        'model': LogisticRegression(solver='liblinear',multi_class='auto'),
        'params': {
            'C': [1,5,10]
        }
    }
}

scores = []

for model_name, mp in model_params.items(): 
    clf =  GridSearchCV(mp['model'], mp['params'], cv=5, return_train_score=False)
    clf.fit(x_train,y_train)
    scores.append({
        'model': model_name,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })
    
df = pd.DataFrame(scores,columns=['model','best_score','best_params'])

print(df) from here you can see that logistic regresssion is better
'''

log=LogisticRegression()
model=log.fit(x_train_vecterize,y_train)
accuracy=model.score(x_test_vecterize,y_test)
predict=model.predict(x_test_vecterize)
print(classification_report(y_test,predict))

#lets make a simple interface
x=input("input a text that you want to check if it is spam: ")
x=wordopt(x)
x_count=vecterizer.transform([x])
prediction=model.predict(x_count)
prediction_probablity=model.predict_proba(x_count)
if prediction[0]==1:
  print("please becarefull!!!, this news is Fake.")
else:
  print("you are free, this news is not Fake .")
print("pediction probablity of being not  Fake: ",str(round(prediction_probablity[0][0],3)*100)+"%")
print("pediction probablity of being Fake: ",str(round(prediction_probablity[0][1],3)*100)+"%")

print("this model is ",str(round(accuracy,3)*100)+"%","accurate")

joblib.dump(model,"fake_new_Model.pkl")