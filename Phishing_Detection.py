import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

# model selection
from sklearn.model_selection import train_test_split, cross_validate

# for classification
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, mean_squared_error

phidf = pd.read_csv("https://raw.githubusercontent.com/v3shabh/No-Phishing/main/train.csv")
test = pd.read_csv("https://raw.githubusercontent.com/v3shabh/No-Phishing/main/test.csv")

q1 = phidf.describe(include="all")
phidf.info()
phidf.shape
t1 = test.describe(include="all")
test.info()
test.shape

def unique_val(df):
    for val in df:
        print("Unique value in ", val, df[val].unique().shape)

unique_val(phidf)
unique_val(test)

drop_cols = ["ratio_intRedirection", "Unnamed: 0", "ratio_nullHyperlinks", "nb_or","ratio_intErrors","submit_email","sfh"]
phidf.drop(drop_cols, axis=1, inplace=True)
test.drop(drop_cols, axis=1, inplace=True)

cols = ["ip","nb_at","nb_qm","nb_tilde","nb_star","nb_dollar",\
       "nb_www","nb_dslash","https_token","punycode","port",\
       "tld_in_path","tld_in_subdomain","abnormal_subdomain",\
       "nb_subdomains","prefix_suffix","random_domain","shortening_service",\
       "path_extension","nb_external_redirection","domain_in_brand","brand_in_subdomain",\
       "brand_in_path","suspecious_tld","statistical_report","login_form","external_favicon",\
       "iframe","popup_window","onmouseover","right_clic","empty_title","domain_in_title",\
       "domain_with_copyright","whois_registered_domain","dns_record","google_index","status"]
phidf[cols] = phidf[cols].astype("object")
test[cols[0:-1]] = test[cols[0:-1]].astype("object")

def value_count_percent(df):
    for val in df:
        if df[val].dtypes == "object":
            print(df[val].value_counts() / df.shape[0])
value_count_percent(phidf)
value_count_percent(test)

phidf.replace({"zero":"0","one":"1","Zero":"0","One":"1"},inplace=True)
test.replace({"zero":"0","one":"1","Zero":"0","One":"1"},inplace=True)

quasi_constant_feat = []
for feature in phidf.columns:
    dominant = (phidf[feature].value_counts()/np.float(len(phidf))).sort_values(ascending= False).values[0]
    if dominant > 0.90:
        quasi_constant_feat.append(feature)
print(quasi_constant_feat)

cols2 = ['nb_at', 'nb_and', 'nb_tilde', 'nb_percent', 'nb_star',
         'nb_colon', 'nb_comma', 'nb_semicolumn', 'nb_dollar', 'nb_space',
         'nb_dslash', 'http_in_path', 'punycode', 'port', 'tld_in_path',
         'tld_in_subdomain', 'abnormal_subdomain', 'random_domain',
         'path_extension', 'nb_external_redirection', 'brand_in_subdomain',
         'brand_in_path', 'suspecious_tld', 'statistical_report', 'login_form',
         'iframe', 'popup_window', 'onmouseover', 'right_clic', 'whois_registered_domain', 'dns_record']
phidf.drop(cols2, axis=1, inplace=True)
test.drop(cols2, axis=1, inplace=True)
#basic stats
phidf.mean()
test.mean()
phidf.median()
phidf.mode()
phidf.std()
q2 = phidf.skew()
q3 = test.skew()

#dealing null values
phidf.isnull().any().sum()
test.isnull().any().sum()

sns.countplot(phidf["status"])

phidf.select_dtypes("float64").hist()
plt.tight_layout()

phidf.select_dtypes("float64").plot(kind="density", subplots=True, sharex=False)
phidf.select_dtypes("int64").plot(kind="density", subplots=True, sharex=False)

phidf.select_dtypes("float64").plot(kind="box",subplots=True,sharex=False,sharey=False)
phidf.select_dtypes("int64").plot(kind="box",subplots=True,sharex=False,sharey=False)

for val in phidf:
    print(phidf[val].value_counts()/phidf.shape[0])
phidf.dtypes
def outlier3sd(q):
    m = q.mean()
    sd = q.std()
    lc = m - (3*sd)
    uc = m + (3*sd)
    n = [index for index, value in enumerate(q) if value > uc or value < lc]
    q[n] = q.mean()
    print("Outliers indexes of ",q , "is", n)
    print(len(n)/phidf.shape[0])

cols3 = ['length_url', 'length_hostname','nb_dots',
         'nb_qm', 'nb_eq', 'nb_underscore', 'nb_slash', 'nb_www', 'nb_com',
         'ratio_digits_url', 'ratio_digits_host', 'nb_subdomains',
         'prefix_suffix', 'shortening_service', 'nb_redirection',
         'length_words_raw', 'char_repeat', 'shortest_words_raw',
         'shortest_word_host', 'shortest_word_path', 'longest_words_raw',
         'longest_word_host', 'longest_word_path', 'avg_words_raw',
         'avg_word_host', 'avg_word_path', 'phish_hints', 'domain_in_brand',
         'nb_hyperlinks', 'ratio_intHyperlinks', 'ratio_extHyperlinks',
         'nb_extCSS', 'ratio_extRedirection', 'ratio_extErrors',
         'links_in_tags', 'ratio_intMedia', 'ratio_extMedia','safe_anchor',
         'domain_registration_length', 'domain_age','web_traffic','page_rank']

def imputer(colname,df):
    for val in colname:
        print(val)
        outlier3sd(df[val])

imputer(cols3, phidf)
imputer(cols3, test)

phidf.skew()
test.skew()

for i in cols3:
    phidf[i] = (phidf[i] ** (1 / 2))

for i in cols3:
    test[i] = (test[i] ** (1 / 2))

cormat = phidf.corr()
sns.heatmap(cormat)
plt.tight_layout()

target = phidf["status"]
phidf.drop("status", axis=1, inplace=True)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
phidf[phidf.select_dtypes(include=['object']).columns] = phidf[phidf.select_dtypes(include=['object']).columns].apply(le.fit_transform)
test[test.select_dtypes(include=['object']).columns] = test[test.select_dtypes(include=['object']).columns].apply(le.fit_transform)
"""#train and test
x_train,x_test,y_train,y_test = train_test_split(phidf, target, random_state= 10, test_size= 0.3)
x_train.shape
x_test.shape

from sklearn.linear_model import LogisticRegression
le = LogisticRegression(max_iter=200)
le.fit(x_train,y_train)
predictions = le.predict(x_test)
confusion_matrix(y_test, predictions)
accuracy_score(y_test, predictions)


ad = AdaBoostClassifier()
ad.fit(x_train,y_train)
predictions_ad = ad.predict(x_test)
confusion_matrix(y_test, predictions_ad)
accuracy_score(y_test,predictions_ad)

gd = GradientBoostingClassifier()
gd.fit(x_train,y_train)
predictions_gd = gd.predict(x_test)
confusion_matrix(y_test,predictions_gd)
accuracy_score(y_test,predictions_gd)"""

phidf["domain_registration_length"].fillna((phidf["domain_registration_length"].mean()), inplace= True)
phidf["domain_age"].fillna((phidf["domain_age"].mean()), inplace= True)
test["domain_registration_length"].fillna((test["domain_registration_length"].mean()), inplace= True)
test.isnull().any()

gd = GradientBoostingClassifier(learning_rate=0.5)
gd.fit(phidf,target)
predictions = gd.predict(test)

res = pd.DataFrame(predictions) #preditcions are nothing but the final predictions of your model on input features of your new unseen test data
res.index = test.index # its important for comparison. Here "test_new" is your new test dataset
res.columns = ["prediction"]
res.to_csv("D:/ML/Dphi_Tech/prediction_results4.csv")      # the csv file will be saved locally on the same location where this notebook is located.

from sklearn.model_selection import GridSearchCV
ad = AdaBoostClassifier()
#Creating a grid of
grid_params = {'n_estimators': [100,200,300,400,500,600,700,800]}#Building a 3 fold CV GridSearchCV
grid_object = GridSearchCV(estimator = ad, param_grid = grid_params, scoring = 'accuracy', cv = 3, n_jobs = -1)
#Fitting the grid to the training
grid_object.fit(phidf, target)#Extracting the best
grid_object.best_params_
ad = AdaBoostClassifier(n_estimators=1200)
ad.fit(phidf,target)
predictions_ad = ad.predict(test)

res = pd.DataFrame(predictions_ad) #preditcions are nothing but the final predictions of your model on input features of your new unseen test data
res.index = test.index # its important for comparison. Here "test_new" is your new test dataset
res.columns = ["prediction"]
res.to_csv("D:/ML/Dphi_Tech/prediction_results15.csv")

rf = RandomForestClassifier(random_state = 42, n_estimators=1000)
rf.fit(phidf,target)
predictions_rf = rf.predict(test)

res = pd.DataFrame(predictions_rf) #preditcions are nothing but the final predictions of your model on input features of your new unseen test data
res.index = test.index # its important for comparison. Here "test_new" is your new test dataset
res.columns = ["prediction"]
res.to_csv("D:/ML/Dphi_Tech/prediction_results14.csv")

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(phidf,target)
pred_knn = knn.predict(test)

res = pd.DataFrame(pred_knn) #preditcions are nothing but the final predictions of your model on input features of your new unseen test data
res.index = test.index # its important for comparison. Here "test_new" is your new test dataset
res.columns = ["prediction"]
