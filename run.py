"""Полный воспроизводимый анализ Olist: SQL → качество сервиса → временная проверка ML."""
import os
os.environ.setdefault("OMP_NUM_THREADS","1")
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
from pathlib import Path
import hashlib,importlib.metadata,json
import duckdb
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import precision_recall_curve
from scripts.download_data import main as download
from src.core import features,timing_mask,previous_order_counts,MODEL_LABELS,FEATURE_LABELS,priority_indices,wilson
from src.models import make_model,metrics,paired_bootstrap

ROOT=Path(__file__).resolve().parent
TABLES=ROOT/"reports/tables"
AS_OF=pd.Timestamp("2018-10-18")


def dump(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")


def load_sources():
    names={
        "orders":"olist_orders_dataset.csv","items":"olist_order_items_dataset.csv",
        "customers":"olist_customers_dataset.csv","sellers":"olist_sellers_dataset.csv",
        "products":"olist_products_dataset.csv","reviews":"olist_order_reviews_dataset.csv",
        "payments":"olist_order_payments_dataset.csv","translation":"product_category_name_translation.csv",
    }
    sources={name:pd.read_csv(ROOT/"data/raw"/file) for name,file in names.items()}
    for col in ["order_purchase_timestamp","order_approved_at","order_delivered_carrier_date",
                "order_delivered_customer_date","order_estimated_delivery_date"]:
        sources["orders"][col]=pd.to_datetime(sources["orders"][col],errors="raise")
    for col in ["review_creation_date","review_answer_timestamp"]:
        sources["reviews"][col]=pd.to_datetime(sources["reviews"][col],errors="raise")
    return sources


def make_cube(frame):
    rows=[]
    for (month,state),g in frame.groupby(["month","customer_state"],dropna=False):
        valid=g[g.timing_valid]
        late=valid[valid.late]
        reviewed=valid[valid.review_answer_timestamp.ge(valid.order_delivered_customer_date)]
        late_reviews=reviewed[reviewed.late]
        timely_reviews=reviewed[~reviewed.late]
        rows.append(dict(month=month,state=state,orders=len(g),
            delivered=len(valid),late=len(late),canceled=int(g.order_status.eq("canceled").sum()),
            unavailable=int(g.order_status.eq("unavailable").sum()),
            delivery_days_sum=float(valid.delivery_days.sum()),
            late_delay_days_sum=float(late.delay_days.sum()),
            items_value=float(valid.items_value.sum()),freight_value=float(valid.freight_value.sum()),
            reviewed_late=len(late_reviews),low_late=int(late_reviews.review_score.le(2).sum()),
            reviewed_timely=len(timely_reviews),low_timely=int(timely_reviews.review_score.le(2).sum())))
    return pd.DataFrame(rows)


def segment_table(frame,key,minimum=100):
    valid=frame[frame.timing_valid]
    result=valid.groupby(key,dropna=False).agg(delivered=("order_id","size"),late=("late","sum"),
        delivery_days=("delivery_days","mean")).reset_index()
    result=result[result.delivered>=minimum].copy()
    result["late_rate"]=result.late/result.delivered
    bounds=[wilson(int(r.late),int(r.delivered)) for r in result.itertuples()]
    result["low"]=[x[0] for x in bounds]
    result["high"]=[x[1] for x in bounds]
    return result.sort_values(["late_rate",key],ascending=[False,True])


def main():
    TABLES.mkdir(parents=True,exist_ok=True)
    (ROOT/"data/processed").mkdir(parents=True,exist_ok=True)
    download()
    raw=load_sources()
    for name,key in [("orders","order_id"),("customers","customer_id"),("products","product_id"),
                     ("sellers","seller_id"),("translation","product_category_name")]:
        if raw[name][key].duplicated().any():
            raise ValueError(f"Неуникальный ключ: {name}.{key}")
    if len(raw["orders"])!=99441 or len(raw["items"])!=112650:
        raise ValueError("Изменился зафиксированный набор Olist версии 2.")
    if raw["items"][["price","freight_value"]].lt(0).any().any():
        raise ValueError("Отрицательные суммы в товарах.")
    foreign_keys={
        "items_to_orders":int((~raw["items"].order_id.isin(raw["orders"].order_id)).sum()),
        "items_to_products":int((~raw["items"].product_id.isin(raw["products"].product_id)).sum()),
        "orders_to_customers":int((~raw["orders"].customer_id.isin(raw["customers"].customer_id)).sum()),
        "items_to_sellers":int((~raw["items"].seller_id.isin(raw["sellers"].seller_id)).sum()),
    }
    if any(foreign_keys.values()):
        raise ValueError(f"Нарушены внешние ключи: {foreign_keys}")
    con=duckdb.connect(str(ROOT/"data/processed/analytics.duckdb"))
    for name,frame in raw.items():
        con.register("raw_"+name,frame)
    con.execute((ROOT/"sql/01_order_mart.sql").read_text(encoding="utf-8-sig"))
    mart=con.sql("SELECT * FROM order_mart ORDER BY order_purchase_timestamp,order_id").df()
    if len(mart)!=len(raw["orders"]) or not mart.order_id.is_unique:
        raise ValueError("JOIN размножил заказы.")
    for col,original in [("items_value",raw["items"].price.sum()),
                         ("freight_value",raw["items"].freight_value.sum()),
                         ("paid_value",raw["payments"].payment_value.sum())]:
        if not np.isclose(mart[col].sum(),original,atol=.01,rtol=0):
            raise ValueError(f"Не сошлась сумма: {col}")
    mart["timing_valid"]=timing_mask(mart)
    mart["late"]=mart.delay_days.gt(0)&mart.timing_valid
    mart["seller_orders_prev_7d"]=previous_order_counts(mart)
    mart["month"]=mart.order_purchase_timestamp.dt.strftime("%Y-%m")
    cohort=mart[mart.order_purchase_timestamp.ge("2017-01-01")&
                mart.order_purchase_timestamp.lt("2018-09-01")].copy()
    con.register("analysis_mart",cohort)
    monthly=con.sql((ROOT/"sql/02_monthly_metrics.sql").read_text(encoding="utf-8-sig")).df()
    sellers=con.sql((ROOT/"sql/03_seller_watchlist.sql").read_text(encoding="utf-8-sig")).df()
    cube=make_cube(cohort)
    states=segment_table(cohort,"customer_state",200)
    categories=segment_table(cohort,"category",100)
    valid=cohort[cohort.timing_valid]
    quality=dict(source_rows={k:len(v) for k,v in raw.items()},mart_rows=len(mart),
        mart_unique_orders=mart.order_id.nunique(),foreign_keys=foreign_keys,
        extra_review_records=int(raw["reviews"].order_id.duplicated().sum()),
        delivered_missing_dates=int((mart.order_status.eq("delivered")&~mart.timing_valid).sum()),
        orders_without_items=int(mart.items_count.isna().sum()),
        reconciliation=dict(items_value=float(mart.items_value.sum()),freight_value=float(mart.freight_value.sum()),
                            paid_value=float(mart.paid_value.sum())),
        snapshot_lower_bound=str(raw["orders"].order_purchase_timestamp.max()),
        as_of=str(AS_OF.date()),cohort_start="2017-01-01",cohort_end_exclusive="2018-09-01")
    overview=dict(orders=len(cohort),delivered=len(valid),late=int(valid.late.sum()),
        late_rate=float(valid.late.mean()),canceled=int(cohort.order_status.eq("canceled").sum()),
        canceled_rate=float(cohort.order_status.eq("canceled").mean()),
        timing_excluded=int((cohort.order_status.eq("delivered")&~cohort.timing_valid).sum()),
        delivered_share=float(len(valid)/len(cohort)),delivery_median=float(valid.delivery_days.median()),
        delivery_p95=float(valid.delivery_days.quantile(.95)),
        mean_delay_if_late=float(valid.loc[valid.late,"delay_days"].mean()),
        items_value_delivered=float(valid.items_value.sum()),
        freight_value_delivered=float(valid.freight_value.sum()),
        multi_seller_share=float(valid.sellers_count.gt(1).mean()))
    reviewed=valid[valid.review_answer_timestamp.ge(valid.order_delivered_customer_date)&valid.review_score.notna()]
    review_rows=[]
    for flag,label in [(False,"В срок"),(True,"С опозданием")]:
        g=reviewed[reviewed.late.eq(flag)]
        low,high=wilson(int(g.review_score.le(2).sum()),len(g))
        review_rows.append(dict(group=label,n=len(g),low_reviews=int(g.review_score.le(2).sum()),
            low_rate=float(g.review_score.le(2).mean()),mean_score=float(g.review_score.mean()),low=low,high=high))
    reviews=pd.DataFrame(review_rows)
    quality["reviews_before_delivery"]=int((valid.review_answer_timestamp.lt(valid.order_delivered_customer_date)).sum())
    quality["reviews_in_comparison"]=len(reviewed)
    cohort.to_csv(ROOT/"data/processed/order_mart.csv",index=False)
    tables={"monthly":monthly,"seller_watchlist":sellers,"states":states,"categories":categories,
            "operational_cube":cube,"review_comparison":reviews}
    for name,table in tables.items():
        table.to_csv(TABLES/(name+".csv"),index=False)
    # Решения фиксируются до обучения и до просмотра качества на тесте.
    protocol=dict(seed=42,primary_metric="average_precision",priority_fraction=.2,
        target="Опоздание по календарной дате среди доставленных заказов с валидными датами",
        candidates=["logistic","boost"],train_purchase_start="2017-01-01",
        train_purchase_end="2017-11-01",train_labels_before="2018-01-01",
        validation_purchase_start="2018-01-01",validation_purchase_end="2018-03-01",
        validation_labels_before="2018-05-01",final_fit_purchase_end="2018-03-01",
        final_fit_labels_before="2018-05-01",test_purchase_start="2018-05-01",
        test_purchase_end="2018-07-01",test_labels_before=str(AS_OF.date()),
        rule="Приоритет заказам с более коротким обещанным сроком",
        predictor_assumptions=["Обещанная дата соответствует показанной при оформлении",
            "Состав корзины, назначенные продавцы, цена доставки и каталог известны при оформлении"])
    dump(ROOT/"reports/protocol.json",protocol)
    eligible=mart[mart.timing_valid&mart.items_count.notna()&mart.promise_days.ge(0)].copy()
    def select(start,end,labels_before):
        return eligible[eligible.order_purchase_timestamp.ge(start)&eligible.order_purchase_timestamp.lt(end)&
                        eligible.order_delivered_customer_date.lt(labels_before)].copy()
    train=select("2017-01-01","2017-11-01","2018-01-01")
    val=select("2018-01-01","2018-03-01","2018-05-01")
    test=select("2018-05-01","2018-07-01",AS_OF)
    fit=select("2017-01-01","2018-03-01","2018-05-01")
    for name,g in [("train",train),("validation",val),("final_fit",fit),("test",test)]:
        print(f"Выборка {name}: {len(g)} заказов, {int(g.late.sum())} опозданий.",flush=True)
    if set(fit.order_id)&set(test.order_id):
        raise ValueError("Обучение пересекается с тестом.")
    xtrain,xval,xtest,xfit=map(features,[train,val,test,fit])
    ytrain,yval,ytest,yfit=[g.late.astype(int).to_numpy() for g in [train,val,test,fit]]
    candidates={}
    validation=[]
    for name in protocol["candidates"]:
        print("Обучение: "+MODEL_LABELS[name],flush=True)
        model=make_model(name).fit(xtrain,ytrain)
        probabilities=model.predict_proba(xval)[:,1]
        validation.append(dict(model=name,label=MODEL_LABELS[name],**metrics(yval,probabilities,val.order_id)))
        candidates[name]=model
    selection=pd.DataFrame(validation)
    chosen=str(selection.sort_values(["average_precision","model"],ascending=[False,True]).iloc[0].model)
    lock={**protocol,"selected":chosen,"validation_metrics":validation,
          "features":list(xtrain.columns),"test_used_for_selection":False}
    dump(ROOT/"reports/selection_lock.json",lock)
    selection.to_csv(TABLES/"model_selection.csv",index=False)
    # Важность признаков описательная, считается на валидации и не определяет выбор модели.
    importance=permutation_importance(candidates[chosen],xval,yval,scoring="average_precision",
                                     n_repeats=3,random_state=42,n_jobs=1)
    imp=pd.DataFrame(dict(feature=xval.columns,importance=importance.importances_mean,
                         std=importance.importances_std))
    imp["label"]=imp.feature.map(FEATURE_LABELS)
    imp=imp.sort_values("importance",ascending=False)
    imp.to_csv(TABLES/"feature_importance.csv",index=False)
    print("Переобучение выбранной модели до начала тестового периода.",flush=True)
    final=make_model(chosen).fit(xfit,yfit)
    probabilities=final.predict_proba(xtest)[:,1]
    # Положительное монотонное преобразование сохраняет ранжирование бизнес-правила.
    rule_scores=1/(1+test.promise_days.to_numpy(float))
    prior=np.repeat(yfit.mean(),len(test))
    model_metrics=[]
    for name,p in [("prior",prior),("short_promise",rule_scores),(chosen,probabilities)]:
        row=dict(model=name,label=MODEL_LABELS[name],**metrics(ytest,p,test.order_id,constant=name=="prior"))
        if name=="short_promise":
            row["brier"]=row["log_loss"]=None  # Ранговое правило не является оценкой вероятности.
        model_metrics.append(row)
    primary=model_metrics[-1]
    rule_metric=model_metrics[1]
    comparison=dict(recall_difference=primary["recall_at_budget"]-rule_metric["recall_at_budget"],
                    **paired_bootstrap(ytest,probabilities,rule_scores,ids=test.order_id))
    capacity=[]
    for q in [.05,.1,.2,.3,.5]:
        for name,p in [(chosen,probabilities),("short_promise",rule_scores),("prior",prior)]:
            row=dict(model=name,label=MODEL_LABELS[name],coverage=q,
                **metrics(ytest,p,test.order_id,q,constant=name=="prior"))
            if name=="short_promise":
                row["brier"]=row["log_loss"]=None
            capacity.append(row)
    pd.DataFrame(model_metrics).to_csv(TABLES/"test_model_metrics.csv",index=False)
    pd.DataFrame(capacity).to_csv(TABLES/"test_capacity.csv",index=False)
    export=test[["order_id","order_purchase_timestamp","order_delivered_customer_date","late","promise_days"]].copy()
    export["probability"]=probabilities
    export["rule_score"]=rule_scores
    export.to_csv(ROOT/"data/processed/test_predictions.csv",index=False)
    split_rows=[]
    for name,g in [("train",train),("validation",val),("test",test),("final_fit",fit)]:
        part=g[["order_id","order_purchase_timestamp","order_delivered_customer_date"]].copy()
        part["split"]=name
        split_rows.append(part)
    pd.concat(split_rows).to_csv(ROOT/"data/processed/model_splits.csv",index=False)
    calibration=pd.DataFrame({"probability":probabilities,"late":ytest})
    calibration["bin"]=pd.qcut(calibration.probability,10,duplicates="drop")
    calibration=calibration.groupby("bin",observed=True).agg(n=("late","size"),
        predicted=("probability","mean"),observed=("late","mean")).reset_index(drop=True)
    calibration.to_csv(TABLES/"calibration.csv",index=False)
    pr_curves=[]
    for name,p in [(chosen,probabilities),("short_promise",rule_scores)]:
        precision,recall,_=precision_recall_curve(ytest,p)
        take=np.unique(np.linspace(0,len(precision)-1,120).astype(int))
        pr_curves.append(dict(model=name,label=MODEL_LABELS[name],
            precision=precision[take].tolist(),recall=recall[take].tolist()))
    status=cohort.order_status.value_counts().rename_axis("status").reset_index(name="orders")
    status.to_csv(TABLES/"order_statuses.csv",index=False)
    result=dict(title="ОБЕЩАЛИ ДОСТАВИТЬ",quality=quality,overview=overview,
        monthly=monthly.to_dict("records"),states=states.to_dict("records"),
        categories=categories.to_dict("records"),sellers=sellers.to_dict("records"),
        reviews=review_rows,cube=cube.to_dict("records"),status=status.to_dict("records"),
        protocol=protocol,selection=lock,primary=primary,model_metrics=model_metrics,
        capacity=capacity,comparison_vs_rule=comparison,calibration=calibration.to_dict("records"),
        feature_importance=imp.to_dict("records"),pr_curves=pr_curves,
        split_counts={name:len(g) for name,g in [("train",train),("validation",val),("test",test),("final_fit",fit)]},
        split_late={name:int(g.late.sum()) for name,g in [("train",train),("validation",val),("test",test),("final_fit",fit)]},
        versions={name:importlib.metadata.version(name) for name in
                  ["numpy","pandas","scikit-learn","scipy","duckdb","matplotlib"]})
    dump(ROOT/"reports/results.json",result)
    con.close()
    from src.reporting import build_reports
    build_reports(ROOT,result)
    print("Готово. Опоздания:",round(overview["late_rate"]*100,2),
          "%; тестовый AP:",round(primary["average_precision"],4),flush=True)


if __name__=="__main__":
    main()
