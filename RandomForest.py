

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import normalize
import warnings
warnings.filterwarnings("ignore")
import numpy as np


rand_st = 17
conf_mat = np.zeros((30, 30), dtype=np.int)
my_data = np.genfromtxt('leaf/leaf.csv', delimiter=',')
shuffled = []
while len(my_data) > 0:
    for n in range(1,37):
        for i in range(len(my_data)):
            if int(my_data[i,0])==n:
                shuffled.append(my_data.take(i, axis=0))
                my_data = np.delete(my_data, i, axis=0)
                break
                
shuffled = np.array(shuffled)
y = shuffled[:,:1].astype(int).ravel()
X = shuffled[:,2:]



clf = RandomForestClassifier(random_state=rand_st)
scoring = ['precision_macro', 'recall_macro']
scores = cross_validate(clf, X, y, cv=10, scoring=scoring)
print("Média da acurácia:",scores['test_precision_macro'].mean())

norm_X = X
norm_X[:,1] = normalize(X[:,1].reshape((-1,1)), axis=0).reshape((-1,))
norm_X[:,-1] = normalize(X[:,-1].reshape((-1,1)), axis=0).reshape((-1,))
n_clf = RandomForestClassifier(random_state=rand_st)
scoring = ['precision_macro', 'recall_macro']
n_scores = cross_validate(n_clf, norm_X, y, cv=10, scoring=scoring)
print("Média da acurácia com corpus normalizado:",n_scores['test_precision_macro'].mean())



test = shuffled[0:34]
train = shuffled[34:]
y_train = train[:,:1].astype(int).ravel()
X_train = train[:,2:]
y_test = test[:,:1].astype(int).ravel()
X_test = test[:,2:]
u_clf = RandomForestClassifier(random_state=rand_st)
u_clf.fit(X_train, y_train)
y_pred = u_clf.predict(X_test)
print("Primeira fatia:")
print(classification_report(y_test, y_pred)) 
print()
print()
conf_mat += confusion_matrix(y_test, y_pred, labels=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36])



test = shuffled[34:68]
train = np.concatenate((shuffled[0:34],shuffled[68:]), axis=0)
y_train = train[:,:1].astype(int).ravel()
X_train = train[:,2:]
y_test = test[:,:1].astype(int).ravel()
X_test = test[:,2:]
u_clf = RandomForestClassifier(random_state=rand_st)
u_clf.fit(X_train, y_train)
y_pred = u_clf.predict(X_test)
print("Segunda fatia:")
print(classification_report(y_test, y_pred)) 
print()
print()
conf_mat += confusion_matrix(y_test, y_pred, labels=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36])



test = shuffled[68:102]
train = np.concatenate((shuffled[0:68],shuffled[102:]), axis=0)
y_train = train[:,:1].astype(int).ravel()
X_train = train[:,2:]
y_test = test[:,:1].astype(int).ravel()
X_test = test[:,2:]
u_clf = RandomForestClassifier(random_state=rand_st)
u_clf.fit(X_train, y_train)
y_pred = u_clf.predict(X_test)
print("Terceira fatia:")
print(classification_report(y_test, y_pred)) 
print()
print()
conf_mat += confusion_matrix(y_test, y_pred, labels=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36])


test = shuffled[102:136]
train = np.concatenate((shuffled[0:102],shuffled[136:]), axis=0)
y_train = train[:,:1].astype(int).ravel()
X_train = train[:,2:]
y_test = test[:,:1].astype(int).ravel()
X_test = test[:,2:]
u_clf = RandomForestClassifier(random_state=rand_st)
u_clf.fit(X_train, y_train)
y_pred = u_clf.predict(X_test)
print("Quarta fatia:")
print(classification_report(y_test, y_pred)) 
print()
print()
conf_mat += confusion_matrix(y_test, y_pred, labels=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36])


test = shuffled[136:170]
train = np.concatenate((shuffled[0:136],shuffled[170:]), axis=0)
y_train = train[:,:1].astype(int).ravel()
X_train = train[:,2:]
y_test = test[:,:1].astype(int).ravel()
X_test = test[:,2:]
u_clf = RandomForestClassifier(random_state=rand_st)
u_clf.fit(X_train, y_train)
y_pred = u_clf.predict(X_test)
print("Quinta fatia:")
print(classification_report(y_test, y_pred)) 
print()
print()
conf_mat += confusion_matrix(y_test, y_pred, labels=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36])


test = shuffled[170:204]
train = np.concatenate((shuffled[0:170],shuffled[204:]), axis=0)
y_train = train[:,:1].astype(int).ravel()
X_train = train[:,2:]
y_test = test[:,:1].astype(int).ravel()
X_test = test[:,2:]
u_clf = RandomForestClassifier(random_state=rand_st)
u_clf.fit(X_train, y_train)
y_pred = u_clf.predict(X_test)
print("Sexta fatia:")
print(classification_report(y_test, y_pred)) 
print()
print()
conf_mat += confusion_matrix(y_test, y_pred, labels=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36])


test = shuffled[204:238]
train = np.concatenate((shuffled[0:204],shuffled[238:]), axis=0)
y_train = train[:,:1].astype(int).ravel()
X_train = train[:,2:]
y_test = test[:,:1].astype(int).ravel()
X_test = test[:,2:]
u_clf = RandomForestClassifier(random_state=rand_st)
u_clf.fit(X_train, y_train)
y_pred = u_clf.predict(X_test)
print("Sétima fatia:")
print(classification_report(y_test, y_pred)) 
print()
print()
conf_mat += confusion_matrix(y_test, y_pred, labels=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36])


test = shuffled[238:272]
train = np.concatenate((shuffled[0:238],shuffled[272:]), axis=0)
y_train = train[:,:1].astype(int).ravel()
X_train = train[:,2:]
y_test = test[:,:1].astype(int).ravel()
X_test = test[:,2:]
u_clf = RandomForestClassifier(random_state=rand_st)
u_clf.fit(X_train, y_train)
y_pred = u_clf.predict(X_test)
print("Oitava fatia:")
print(classification_report(y_test, y_pred)) 
print()
print()
conf_mat += confusion_matrix(y_test, y_pred, labels=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36])


test = shuffled[272:306]
train = np.concatenate((shuffled[0:272],shuffled[306:]), axis=0)
y_train = train[:,:1].astype(int).ravel()
X_train = train[:,2:]
y_test = test[:,:1].astype(int).ravel()
X_test = test[:,2:]
u_clf = RandomForestClassifier(random_state=rand_st)
u_clf.fit(X_train, y_train)
y_pred = u_clf.predict(X_test)
print("Nona fatia:")
print(classification_report(y_test, y_pred)) 
print()
print() 
conf_mat += confusion_matrix(y_test, y_pred, labels=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36])


test = shuffled[306:]
train = shuffled[0:306]
y_train = train[:,:1].astype(int).ravel()
X_train = train[:,2:]
y_test = test[:,:1].astype(int).ravel()
X_test = test[:,2:]
u_clf = RandomForestClassifier(random_state=rand_st)
u_clf.fit(X_train, y_train)
y_pred = u_clf.predict(X_test)
print("Décima fatia:")
print(classification_report(y_test, y_pred)) 
print()
print()
conf_mat += confusion_matrix(y_test, y_pred, labels=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36])


hit_rate = np.zeros(30)
for i in range(30):
    sum = 0
    for j in range(30):
        sum += conf_mat[i][j]
    hit_rate[i] = conf_mat[i][i]/sum
 

for i in range(15):
    print("Precisão da classe", i+1,"=", hit_rate[i])
for i in range(15,30):
    print("Precisão da classe", i+7,"=", hit_rate[i])    

print("Média de precisão para todas as classes:", hit_rate.mean())

