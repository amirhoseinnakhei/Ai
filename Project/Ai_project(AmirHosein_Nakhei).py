from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score , precision_score , recall_score

knn = KNeighborsClassifier()
svm = SVC()
Dtree = DecisionTreeClassifier()

estimator = [('knn' , knn) , ('svm' , svm) , ('Dtree' , Dtree)]

Data = load_breast_cancer()
X , y = Data['data'] , Data['target']

scaler = MinMaxScaler()
sc_data = scaler.fit_transform(X)

x_train , x_test , y_train , y_test = train_test_split(
    sc_data , y , test_size=0.3 , random_state=50
)

vote = VotingClassifier(
    estimators = estimator , voting = 'hard'
)

vote.fit(x_train,y_train)
vote_value = vote.predict(x_test)

accuracy = accuracy_score(y_test , vote_value)
precision = precision_score(y_test , vote_value , average = 'weighted')
recall = recall_score(y_test , vote_value , average = 'weighted')

print(f'vote_value is : {vote_value}')
print()
print('==================================================================================================')
print()
print(f'accuracy : {accuracy} , precision : {precision} , recall : {recall}')
