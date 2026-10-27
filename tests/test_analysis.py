"""Проверки реальных рисков анализа: JOIN, календарные сроки, прошлое и бюджет."""
from pathlib import Path
import json
import duckdb
import numpy as np
import pandas as pd
import pytest
from src.core import features,timing_mask,previous_order_counts,priority_indices,wilson,FEATURES
from src.models import make_model,metrics

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture
def synthetic_sources():
    orders=pd.DataFrame({
        "order_id":["a","b"],"customer_id":["c1","c2"],"order_status":["delivered","canceled"],
        "order_purchase_timestamp":pd.to_datetime(["2018-01-01 10:00","2018-01-02 10:00"]),
        "order_estimated_delivery_date":pd.to_datetime(["2018-01-05","2018-01-05"]),
        "order_delivered_customer_date":pd.to_datetime(["2018-01-05 22:00",None]),
    })
    return dict(
        orders=orders,
        customers=pd.DataFrame({"customer_id":["c1","c2"],"customer_state":["SP","RJ"],"customer_city":["x","y"]}),
        items=pd.DataFrame({"order_id":["a","a","b"],"order_item_id":[1,2,1],
            "product_id":["p1","p2","p1"],"seller_id":["s1","s2","s1"],
            "price":[10.,20.,5.],"freight_value":[1.,2.,.5]}),
        products=pd.DataFrame({"product_id":["p1","p2"],"product_category_name":["cat","cat"],
            "product_weight_g":[100,200],"product_length_cm":[10,20],
            "product_height_cm":[10,20],"product_width_cm":[10,20]}),
        sellers=pd.DataFrame({"seller_id":["s1","s2"],"seller_state":["SP","RJ"]}),
        translation=pd.DataFrame({"product_category_name":["cat"],"product_category_name_english":["test"]}),
        payments=pd.DataFrame({"order_id":["a","a","b"],"payment_value":[12.,21.,5.5]}),
        reviews=pd.DataFrame({"order_id":["a","a","b"],"review_id":["r1","r2","r3"],"review_score":[5,1,2],
            "review_answer_timestamp":pd.to_datetime(["2018-01-06","2018-01-07","2018-01-08"]),
            "review_creation_date":pd.to_datetime(["2018-01-06","2018-01-07","2018-01-08"])}),
    )


def test_sql_joins_preserve_orders_and_money(synthetic_sources):
    con=duckdb.connect()
    for name,frame in synthetic_sources.items():
        con.register("raw_"+name,frame)
    con.execute((ROOT/"sql/01_order_mart.sql").read_text(encoding="utf-8-sig"))
    result=con.sql("SELECT * FROM order_mart ORDER BY order_id").df()
    assert len(result)==2 and result.order_id.is_unique
    assert result.items_value.sum()==pytest.approx(35)
    assert result.freight_value.sum()==pytest.approx(3.5)
    assert result.paid_value.sum()==pytest.approx(38.5)
    a=result.iloc[0]
    assert a.review_score==1 and a.review_records==2
    assert a.items_count==2 and a.sellers_count==2
    con.close()


def test_promised_day_is_not_midnight_deadline(synthetic_sources):
    con=duckdb.connect()
    for name,frame in synthetic_sources.items():
        con.register("raw_"+name,frame)
    con.execute((ROOT/"sql/01_order_mart.sql").read_text(encoding="utf-8-sig"))
    result=con.sql("SELECT * FROM order_mart ORDER BY order_id").df()
    assert result.iloc[0].delay_days==0
    assert timing_mask(result).tolist()==[True,False]
    con.close()


def test_bad_dates_and_cancellation_are_not_on_time(synthetic_sources):
    frame=synthetic_sources["orders"].copy()
    frame.loc[0,"order_delivered_customer_date"]=pd.Timestamp("2017-12-31")
    assert not timing_mask(frame).any()


def test_seller_history_does_not_see_future_or_same_timestamp():
    frame=pd.DataFrame({"main_seller_id":["a"]*4,
        "order_purchase_timestamp":pd.to_datetime(["2018-01-01","2018-01-02","2018-01-02","2018-01-20"])})
    assert previous_order_counts(frame).tolist()==[0,1,1,0]
    longer=pd.concat([frame,pd.DataFrame({"main_seller_id":["a"],
        "order_purchase_timestamp":[pd.Timestamp("2018-02-01")]})],ignore_index=True)
    assert previous_order_counts(longer).iloc[:4].tolist()==[0,1,1,0]


def feature_fixture(n=60):
    frame=pd.DataFrame({col:np.ones(n) for col in [
        "promise_days","items_count","sellers_count","categories_count","weight_kg","volume_litres",
        "purchase_hour","purchase_weekday","purchase_month","same_state","seller_orders_prev_7d"]})
    frame["items_value"]=np.arange(n)+10
    frame["freight_value"]=5
    frame["customer_state"]="SP"
    frame["seller_state"]="SP"
    frame["category"]="test"
    return frame


def test_future_outcomes_cannot_change_features():
    frame=feature_fixture()
    before=features(frame)
    frame["order_status"]="canceled"
    frame["review_score"]=1
    frame["delay_days"]=999
    frame["paid_value"]=100000
    frame["order_delivered_customer_date"]=pd.Timestamp("2099-01-01")
    pd.testing.assert_frame_equal(before,features(frame))
    assert list(before.columns)==FEATURES


def test_pipeline_handles_missing_and_unseen_categories():
    frame=feature_fixture()
    frame.loc[0,"weight_kg"]=np.nan
    x=features(frame)
    model=make_model("logistic").fit(x,np.tile([0,1],30))
    future=x.iloc[:2].copy()
    future["customer_state"]="UNSEEN"
    prediction=model.predict_proba(future)[:,1]
    assert np.isfinite(prediction).all()
    assert ((prediction>=0)&(prediction<=1)).all()


def test_budget_and_tie_break_are_deterministic():
    chosen=priority_indices([.9,.9,.2,.1],.5,["b","a","c","d"])
    assert chosen.tolist()==[1,0]
    assert len(priority_indices(np.ones(103),.2))==20
    with pytest.raises(ValueError):
        priority_indices([1,2],1.1)


def test_metrics_use_same_contact_budget():
    y=np.array([1,0,0,1,0,0,0,0,0,0])
    score=np.array([.9,.8,.7,.6,.5,.4,.3,.2,.1,.05])
    observed=metrics(y,score,fraction=.2)
    assert observed["priority_n"]==2
    assert observed["precision_at_budget"]==pytest.approx(.5)
    assert observed["recall_at_budget"]==pytest.approx(.5)
    random=metrics(y,np.full(10,.2),fraction=.2,constant=True)
    assert random["recall_at_budget"]==pytest.approx(.2)
    assert random["precision_at_budget"]==pytest.approx(.2)


def test_wilson_zero_and_small_samples():
    assert wilson(0,0)==(None,None)
    low,high=wilson(0,10)
    assert abs(low)<1e-12 and 0<high<1
    low,high=wilson(10,10)
    assert 0<low<1 and high==pytest.approx(1)


@pytest.mark.skipif(not (ROOT/"data/processed/test_predictions.csv").exists(),reason="Сначала выполните python run.py")
def test_saved_temporal_split_and_independent_test_metrics():
    splits=pd.read_csv(ROOT/"data/processed/model_splits.csv",parse_dates=[
        "order_purchase_timestamp","order_delivered_customer_date"])
    fit=splits[splits.split=="final_fit"]
    test=splits[splits.split=="test"]
    assert not set(fit.order_id)&set(test.order_id)
    assert fit.order_purchase_timestamp.max()<pd.Timestamp("2018-03-01")
    assert fit.order_delivered_customer_date.max()<pd.Timestamp("2018-05-01")
    assert test.order_purchase_timestamp.min()>=pd.Timestamp("2018-05-01")
    pred=pd.read_csv(ROOT/"data/processed/test_predictions.csv")
    result=json.loads((ROOT/"reports/results.json").read_text(encoding="utf-8"))
    picked=np.lexsort((pred.order_id.to_numpy(),-pred.probability.to_numpy()))[:int(.2*len(pred))]
    assert pred.iloc[picked].late.mean()==pytest.approx(result["primary"]["precision_at_budget"])
    assert pred.iloc[picked].late.sum()/pred.late.sum()==pytest.approx(result["primary"]["recall_at_budget"])
    assert result["selection"]["test_used_for_selection"] is False


@pytest.mark.skipif(not (ROOT/"reports/results.json").exists(),reason="Сначала выполните python run.py")
def test_dashboard_cube_reconciles_to_report():
    result=json.loads((ROOT/"reports/results.json").read_text(encoding="utf-8"))
    cube=pd.DataFrame(result["cube"])
    assert cube[["month","state"]].duplicated().sum()==0
    for field in ["orders","delivered","late","canceled"]:
        assert cube[field].sum()==result["overview"][field]
    assert sum(row["orders"] for row in result["status"])==result["overview"]["orders"]
