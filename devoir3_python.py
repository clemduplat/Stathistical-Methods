from sklearn.model_selection import train_test_split
#Génération de données basé sur ma matricule
#utilisation du wage dataset
wage = load_data("Wage")
#print(wage.columns)
#Index(['year', 'age', 'maritl', 'race', 'education', 'region', 'jobclass','health', 'health_ins', 'logwage', 'wage'],dtype='object')
wage['health'] = wage['health'].apply(lambda x: 1 if x == "2. >=Very Good" else 0)
matricule = 2405979
np.random.seed(matricule)
train_data, valid_data = train_test_split(wage, test_size=500, random_state=matricule)
X_train = train_data[['age', 'wage']]
y_train = train_data['health']

#loocv: on divise les données de manière à ce que chaque observation soit utilisée comme un cas de test unique (les autres obs servent donc à entrâiner)
#-> on entraîne n-1 obs, on test avec 1 obs et ce processus est répeté n fois
# + : elle utilise toutes les données -> bonne estimation de l'erreur
# - : coûteuse si n est grand
# 5Fold: on divise les données en 5 sous-ensembles
# -> on entraine le modèle sur 4 sous-ensembles, on test sur un -> répète le processus 5 fois
# + : elle utilise moins de données que loocv -> moins coûteuse
# - : -> moins précis aussi
erreur_loocv = []
erreur_5fold = []
for k in range(1,181):
    knn = KNeighborsClassifier(n_neighbors=k)
    # loocv
    #https://www.statology.org/leave-one-out-cross-validation-in-python/
    #Provides train/test indices to split data in train/test sets. Each sample is used once as a test set (singleton) while the remaining samples form the training set.
    # Note: LeaveOneOut() is equivalent to KFold(n_splits=n) and LeavePOut(p=1) where n is the number of samples.
    loo = LeaveOneOut()
    loo_scores = cross_validate(knn, X_train, y_train, cv=loo,scoring='accuracy') 
    erreur_loocv.append(1-np.mean(loo_scores['test_score']))
    # 5fold Cv
    kf = KFold(n_splits=5, shuffle=True, random_state=matricule)
    kfold_scores = cross_validate(knn, X_train, y_train, cv=kf,scoring='accuracy')
    # kfold_error = 1 - kfold_scores.mean()
    erreur_5fold.append(1- np.mean(kfold_scores['test_score']))

plt.figure(figsize=(12, 6))
plt.plot([1/k for k in range(1, 181)], erreur_loocv, label='LOOCV')
plt.plot([1/k for k in range(1, 181)], erreur_5fold, label='5-Fold CV')
plt.xlabel('1/K')
plt.ylabel('Taux d\'erreur')
plt.title('Taux d\'erreur vs. 1/K')
plt.legend()
plt.show()
optimal_k_loocv = erreur_loocv.index(min(erreur_loocv)) + 1
optimal_k_kfold = erreur_5fold.index(min(erreur_5fold)) + 1

print(f"K optimal pour LOOCV: {optimal_k_loocv}")
print(f"K optimal pour 5-Fold CV: {optimal_k_kfold}")
#section 4.7.2 from ISLP website
#modèle 1 : b0 + b1*x1+b2*x2
#modèle 2 : b0 + b1*x1+b2*x2 +b3*x1^2 ¨+b4*x2^2
X_train_1 = sm.add_constant(X_train)
X_train_2 = X_train.copy()
#on inclut les termes quadratiques
X_train_2['age^2']= X_train['age']**2
X_train_2['wage^2']= X_train['wage']**2
X_train_2 = sm.add_constant(X_train_2)
loo = LeaveOneOut()
kf = KFold(n_splits=5, shuffle=True, random_state=matricule)

def mean_error(X, y, cv):
    errors = []
    #pour chacun des fold on va calucler l'erreur puis prendre la moyenne
    for i, j in cv.split(X):
        X_train, X_test= X.iloc[i],X.iloc[j]
        y_train, y_test= y.iloc[i],y.iloc[j]
        #from ISLP: The syntax of sm.GLM() is similar to that of sm.OLS(), except that
        # we must pass in the argument family=sm.families.Binomial() in order to
        # tell statsmodels to run a logistic regression rather than some other type of
        # generalized linear model.
        #on fit un logistique régression comme dans le livre
        glm = sm.GLM(y_train, X_train, family=sm.families.Binomial())
        result = glm.fit()
        probs= result.predict(X_test)>0.5 #est-ce que ça va être utile de changer ce 0.5 ? 
        #voir si c'est classifié de la même manière
        error = np.mean(probs!=y_test)
        errors.append(error)
    return np.mean(errors)
#enft j'aurais juste pu utiliser à nv cross_validate mais pas sûre si il va bien faire la logistique régression 
#si je précise pas sm.families.Binomial()
model1_loo = mean_error(X_train_1, y_train, loo)
model1_kf = mean_error(X_train_1, y_train, kf)
model2_loo = mean_error(X_train_2, y_train, loo)
model2_kf = mean_error(X_train_2, y_train, kf)
print("M1 LOOCV", model1_loo)
print("M1 5-Fold", model1_kf)
print("M2 LOOCV", model2_loo)
print("M2 5-Fold", model2_kf)

#section 4.7.3 et 4.7.4
# Linear Discriminant Analysis (LDA)
# SincetheLDAestimatorautomaticallyaddsanintercept,weshouldre
# movethecolumncorrespondingtotheinterceptinbothX_trainandX_test.

loo = LeaveOneOut()
kf = KFold(n_splits=5, shuffle=True, random_state=matricule)
lda_model = LDA(store_covariance=True)
qda_model = QDA(store_covariance=True)
#on va faire pareil que l'exo précedent mais sans binomial
#(à nv je pense qu'on aurait pu utiliser cross_validate mais bon)
def mean_error(model, X, y, cv):
    errors = []
    #pour chaque fold
    for i, j in cv.split(X):
        model.fit(X.iloc[i], y.iloc[i])
        predictions = model.predict(X.iloc[j])
        error = np.mean(predictions!=y.iloc[j])
        errors.append(error)
    return np.mean(errors)

lda_loocv_error = mean_error(lda_model, X_train, y_train, loo)
lda_kfold_error = mean_error(lda_model, X_train, y_train, kf)

qda_loocv_error = mean_error(qda_model, X_train, y_train, loo)
qda_kfold_error = mean_error(qda_model, X_train, y_train, kf)

print("LDA LOOCV ", lda_loocv_error)
print("LDA 5-Fold ", lda_kfold_error)
print("QDA LOOCV ", qda_loocv_error)
print("QDA 5-Fold ", qda_kfold_error)

# comme au devoir 2
##source: https://ogrisel.github.io/scikit-learn.org/sklearn-tutorial/auto_examples/neighbors/plot_classification.html
# matricule = 2405979
#je vais redefinir X et y avec .values sinon j'ai des erreurs après pcq je prends pas les values
X = train_data[['age', 'wage']].values
y = train_data['health'].values

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00'])
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
#arrange prend trop de mémoire donc je remplace en utilisant linspace
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))

#on fait comme au devoir 2 mais avec nos méthodes choisies
#knn, ici j'utilise le 72 que j'ai choisi à la question 1 a)
k = 72
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X, y)
Z_knn = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z_knn = Z_knn.reshape(xx.shape)
log_model = LogisticRegression()
log_model.fit(X, y)
Z_log = log_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z_log = Z_log.reshape(xx.shape)
lda = LinearDiscriminantAnalysis()  
lda.fit(X, y)
Z_lda = lda.predict(np.c_[xx.ravel(), yy.ravel()])
Z_lda = Z_lda.reshape(xx.shape)
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z_knn, cmap=cmap_light, alpha=0.3)
plt.contour(xx, yy, Z_log, levels=[0.5], colors='blue')
plt.contour(xx, yy, Z_lda, levels=[0.5], colors='green')
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, s= 5)
plt.title("Séparation des classes avec les méthodes : KNN, Régression Logistique, LDA")
plt.xlabel('Age')
plt.ylabel('Wage')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.show()
knn_prediction = knn.predict(X)
print("Matrice de confusion pour KNN :")
c1 = confusion_table(y, knn_prediction)
print(c1)
print(classification_report(y, knn_prediction))
pred_log = log_model.predict(X)
print("Matrice de confusion pour régression logistique:")
c2 = confusion_table(y, pred_log)
print(c2)
print(classification_report(y, pred_log))
pred_lda = lda.predict(X)
print("Matrice de confusion pour LDA :")
c3 = confusion_table(y, pred_lda)
print(c3)
print(classification_report(y, pred_lda))
# utiliser iris dataset qui est dans sklearn
iris = load_iris()
X = iris.data  

# on veut calculer theta_prime qui est donné comme
#l'espérance du minimum entre : X2 + log(X1), X1 + X3 − 2X4, exp {−|X1 − X4|} , X2 + 3X3
#on en fait une fonction au cas ou si on veut le réutiliser plus tard
def theta_estimation(data):
    return np.mean(np.min([
        data[:, 1] + np.log(data[:, 0]),       
        data[:, 0] + data[:, 2] - 2 * data[:, 3], 
        np.exp(-np.abs(data[:, 0] - data[:, 3])), 
        data[:, 1] + 3 * data[:, 2]              
    ], axis=0))
theta_hat=theta_estimation(X)  
print(theta_hat)
# from section 5_3_3 from ISLP website
def boot_SE(func, data, B=3500, seed=0):
    rng = np.random.default_rng(seed)
    first_, second_ = 0, 0
    n = data.shape[0]
    for _ in range(B):
        idx = rng.choice(n, n, replace=True)
        value = func(data[idx])
        first_ += value
        second_ += value**2
    standard_error = np.sqrt(second_ / B - (first_ / B)**2)
    return standard_error
theta_original = theta_estimation(X)

# on veut utiliser le bootstrap avec n repetitions
#voir ISLP_website
repetitions=3500
bootstrap_estimates=np.zeros(repetitions)
np.random.seed(2405979)  
for i in range(repetitions):
    bootstrap_sample = X[np.random.choice(range(len(X)), size=len(X), replace=True)]
    bootstrap_estimates[i] = theta_estimation(bootstrap_sample)
bias = np.mean(bootstrap_estimates)-theta_original
std_error = boot_SE(theta_estimation, X, B=3500, seed=matricule)
print("Estimation bootstrap de θ:", np.mean(bootstrap_estimates))
print("Biais estimé:", bias)
print("Erreur-type estimée:", std_error)
#found this on internet:
#https://www.statsmodels.org/dev/generated/statsmodels.stats.weightstats.DescrStatsW.html
# tconfint_mean([alpha, alternative]) : two-sided confidence interval for weighted mean of data

output = DescrStatsW(bootstrap_estimates)
int_bas, int_haut = output.tconfint_mean(alpha=0.05, alternative='two-sided') 
print("Intervalle de confiance: ", [int_bas, int_haut])
