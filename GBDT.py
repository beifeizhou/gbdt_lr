import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import (RandomTreesEmbedding, RandomForestClassifier,
                              GradientBoostingClassifier)
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt

path = ''
data = pd.read_csv(path+'feature_score.csv',  header=None)

X = data[range(3000)]
y = data[[3000]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train, X_train_lr, y_train, y_train_lr = train_test_split(X_train, y_train, test_size=0.5)

n_estimator = 10
# Unsupervised transformation based on totally random trees
rt = RandomTreesEmbedding(max_depth=3, n_estimators=n_estimator,
    random_state=0)
rt_lm = LogisticRegression()
pipeline = make_pipeline(rt, rt_lm)
pipeline.fit(X_train, y_train)
y_pred_rt = pipeline.predict_proba(X_test)[:, 1]
fpr_rt_lm, tpr_rt_lm, _ = roc_curve(y_test, y_pred_rt)

# Supervised transformation based on random forests
rf = RandomForestClassifier(max_depth=3, n_estimators=n_estimator)
rf_enc = OneHotEncoder()
rf_lm = LogisticRegression()
rf.fit(X_train, y_train)
rf_enc.fit(rf.apply(X_train))
rf_lm.fit(rf_enc.transform(rf.apply(X_train_lr)), y_train_lr)

y_pred_rf_lm = rf_lm.predict_proba(rf_enc.transform(rf.apply(X_test)))[:, 1]
fpr_rf_lm, tpr_rf_lm, _ = roc_curve(y_test, y_pred_rf_lm)

grd = GradientBoostingClassifier(n_estimators=n_estimator)
grd_enc = OneHotEncoder()
grd_lm = LogisticRegression()
grd.fit(X_train, y_train)
grd_enc.fit(grd.apply(X_train)[:, :, 0])
grd_lm.fit(grd_enc.transform(grd.apply(X_train_lr)[:, :, 0]), y_train_lr)

y_pred_grd_lm = grd_lm.predict_proba(
            grd_enc.transform(grd.apply(X_test)[:, :, 0]))[:, 1]
fpr_grd_lm, tpr_grd_lm, _ = roc_curve(y_test, y_pred_grd_lm)

# The gradient boosted model by itself
y_pred_grd = grd.predict_proba(X_test)[:, 1]
fpr_grd, tpr_grd, _ = roc_curve(y_test, y_pred_grd)


# The random forest model by itself
y_pred_rf = rf.predict_proba(X_test)[:, 1]
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_rf)
a = grd_enc.transform(grd.apply(X_train_lr)[:, :, 0])

#lr only 
enc = OneHotEncoder(handle_unknown='ignore')
lm = LogisticRegression()
enc.fit(X_train)
lm.fit(enc.transform(X_train_lr), y_train_lr)
y_pred_lm = lm.predict_proba(enc.transform(X_test))
fpr_lm, tpr_lm, _ = roc_curve(y_test, y_pred_rf_lm)

#enc-gbdt-enc-lr
enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(X_train)
X_train_lr_enc = enc.transform(X_train_lr)
X_train_enc = enc.transform(X_train)
X_test_enc = enc.transform(X_test)

grd = GradientBoostingClassifier(n_estimators=n_estimator)
grd_enc = OneHotEncoder()
grd_lm = LogisticRegression()
grd.fit(X_train_enc, y_train)
grd_enc.fit(grd.apply(X_train_enc)[:, :, 0])
grd_lm.fit(grd_enc.transform(grd.apply(X_train_lr_enc)[:, :, 0]), y_train_lr)

y_pred_grd_lm_enc = grd_lm.predict_proba(
            grd_enc.transform(grd.apply(X_test_enc)[:, :, 0]))[:, 1]
fpr_grd_lm_enc, tpr_grd_lm_enc, _ = roc_curve(y_test, y_pred_grd_lm_enc)

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_rt_lm, tpr_rt_lm, label='RT + LR')
plt.plot(fpr_rf, tpr_rf, label='RF')
plt.plot(fpr_rf_lm, tpr_rf_lm, label='RF + LR')
plt.plot(fpr_grd, tpr_grd, label='GBT')
plt.plot(fpr_grd_lm, tpr_grd_lm, label='GBT + LR')
plt.plot(fpr_grd_lm_enc, tpr_grd_lm_enc, label='ENC + GBT + LR')
plt.plot(fpr_lm, tpr_lm, label='LR')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()

########
y_pred = lm.predict(enc.transform(X_test))
a = y_test.values.flatten()-y_pred
print 'The precision is: '+str((len(a)-np.count_nonzero(a))/float(len(a)))
print y_pred_lm
