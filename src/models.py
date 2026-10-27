"""Сравнение моделей по времени; все преобразования обучаются внутри Pipeline."""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score,roc_auc_score,brier_score_loss,log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from src.core import NUMERIC,CATEGORICAL,priority_indices,wilson


def make_model(kind):
    numeric=Pipeline([("missing",SimpleImputer(strategy="median")),("scale",StandardScaler())])
    categories=OneHotEncoder(handle_unknown="ignore",sparse_output=(kind=="logistic"))
    transform=ColumnTransformer([("numeric",numeric,NUMERIC),("categories",categories,CATEGORICAL)],
                                sparse_threshold=1 if kind=="logistic" else 0)
    classifier=(
        LogisticRegression(C=.3,max_iter=1000,random_state=42)
        if kind=="logistic" else
        HistGradientBoostingClassifier(max_iter=120,max_leaf_nodes=15,min_samples_leaf=100,
            learning_rate=.07,l2_regularization=10,early_stopping=False,random_state=42)
    )
    return Pipeline([("features",transform),("model",classifier)])


def metrics(y, probabilities, ids=None, fraction=.2, constant=False):
    y=np.asarray(y,dtype=int)
    p=np.clip(np.asarray(probabilities,dtype=float),1e-7,1-1e-7)
    picked=priority_indices(p,fraction,ids)
    prevalence=float(y.mean())
    if constant:
        precision=prevalence
        recall=len(picked)/len(y)
        successes=None
        low,high=wilson(int(y.sum()),len(y))
    else:
        successes=int(y[picked].sum())
        precision=float(y[picked].mean()) if len(picked) else 0
        recall=float(successes/y.sum()) if y.sum() else 0
        low,high=wilson(successes,len(picked))
    return dict(n=len(y),late=int(y.sum()),prevalence=prevalence,
        average_precision=float(average_precision_score(y,p)),
        roc_auc=float(roc_auc_score(y,p)),brier=float(brier_score_loss(y,p)),
        log_loss=float(log_loss(y,p)),priority_n=len(picked),
        precision_at_budget=precision,recall_at_budget=recall,lift_at_budget=precision/prevalence,
        precision_low=low,precision_high=high,priority_late=successes)


def paired_bootstrap(y, p, rule_scores, repeats=300, ids=None):
    """Парный bootstrap прироста полноты при одинаковом бюджете; модели фиксированы."""
    rng=np.random.default_rng(42)
    y,p,rule_scores=np.asarray(y),np.asarray(p),np.asarray(rule_scores)
    secondary=np.arange(len(y)) if ids is None else np.asarray(ids)
    estimates=[]
    for _ in range(repeats):
        sample=rng.integers(0,len(y),len(y))
        yy=y[sample]
        if not yy.sum():
            continue
        a=priority_indices(p[sample],ids=secondary[sample])
        b=priority_indices(rule_scores[sample],ids=secondary[sample])
        estimates.append(float((yy[a].sum()-yy[b].sum())/yy.sum()))
    return dict(repeats=len(estimates),low=float(np.quantile(estimates,.025)),
                high=float(np.quantile(estimates,.975)))
