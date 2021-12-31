
import pandas as pd

df = pd.read_csv('Data/Data.csv', encoding="ISO-8859-1")
df.head()

train = df[df['Date'] < '20150101']
test = df[df['Date']> '20141231']

# removing punctuations
data = train.iloc[:,2:27]
data.replace("[^a-zA-Z]"," ", regex = True, inplace = True)

# Renaming the columns name
list1 = [i for i in range(25)]
new_index = [str(i) for i in list1]
data.columns = new_index

# Converting the headlines lower 
for index in new_index:
    data[index] = data[index].str.lower()

# coverting into paragraph using headling
headlines = []
for row in range (0,len(data.index)):
    headlines.append(' '.join(str(x) for x in data.iloc[row,0:25]))

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

# Bag of Words
CountVector = CountVectorizer(ngram_range=(2,2))
traindataset = CountVector.fit_transform(headlines)

# implementing RandomForest Classifier
randomclassifier = RandomForestClassifier(n_estimators=200,criterion='entropy')
randomclassifier.fit(traindataset, train['Label'])

# predict for the Test Dataset
test_transform = []
for row in range(0,len(test.index)):
    test_transform.append(' '.join(str(x) for x in test.iloc[row,2:27]))
test_dataset = CountVector.transform(test_transform)
predictions = randomclassifier.predict(test_dataset)

# checking the accuracy of the model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

matrix = confusion_matrix(test['Label'],predictions)
print(matrix)

score = accuracy_score(test['Label'],predictions)
print(score)

report = classification_report(test['Label'],predictions)
print(report)