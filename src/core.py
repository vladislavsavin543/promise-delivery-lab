"""Определения метрик и признаки, доступные в момент оформления заказа."""
import numpy as np
import pandas as pd

NUMERIC = [
    "promise_days","log_items_value","log_freight_value","freight_share",
    "items_count","sellers_count","categories_count","weight_kg","volume_litres",
    "purchase_hour","purchase_weekday","purchase_month","same_state",
    "log_seller_orders_prev_7d",
]
CATEGORICAL = ["customer_state","seller_state","category"]
FEATURES = NUMERIC+CATEGORICAL
FEATURE_LABELS = {
    "promise_days":"Обещанный срок","log_items_value":"Стоимость товаров",
    "log_freight_value":"Стоимость доставки","freight_share":"Доля доставки в сумме",
    "items_count":"Число позиций","sellers_count":"Число продавцов",
    "categories_count":"Число категорий","weight_kg":"Вес товаров",
    "volume_litres":"Объём товаров","purchase_hour":"Час заказа",
    "purchase_weekday":"День недели","purchase_month":"Месяц заказа",
    "same_state":"Покупатель и продавец в одном штате",
    "log_seller_orders_prev_7d":"Заказы продавца за предыдущие 7 дней",
    "customer_state":"Штат покупателя","seller_state":"Штат продавца","category":"Категория товаров",
}
MODEL_LABELS={"prior":"Постоянный риск","short_promise":"Короткий обещанный срок",
              "logistic":"Логистическая регрессия","boost":"Градиентный бустинг"}
STATUS_LABELS={"delivered":"Доставлен","canceled":"Отменён","shipped":"Передан в доставку",
    "unavailable":"Недоступен","invoiced":"Выставлен счёт","processing":"Обрабатывается",
    "created":"Создан","approved":"Подтверждён"}
CATEGORY_LABELS={
    "bed_bath_table":"Постель, ванна и стол","health_beauty":"Красота и здоровье",
    "sports_leisure":"Спорт и досуг","furniture_decor":"Мебель и декор",
    "computers_accessories":"Компьютеры и аксессуары","housewares":"Товары для дома",
    "watches_gifts":"Часы и подарки","telephony":"Телефония","garden_tools":"Садовые инструменты",
    "auto":"Автотовары","toys":"Игрушки","cool_stuff":"Оригинальные товары",
    "perfumery":"Парфюмерия","baby":"Детские товары","electronics":"Электроника",
    "stationery":"Канцелярия","fashion_bags_accessories":"Сумки и аксессуары",
    "pet_shop":"Товары для животных","office_furniture":"Офисная мебель",
    "luggage_accessories":"Багаж и аксессуары","consoles_games":"Игровые приставки",
    "construction_tools_construction":"Строительные инструменты",
    "home_appliances":"Бытовая техника","musical_instruments":"Музыкальные инструменты",
    "small_appliances":"Мелкая бытовая техника","unknown":"Категория не указана",
}


def timing_mask(frame):
    """Дата обещания задаёт календарный день; доставка в тот же день своевременна."""
    return (
        frame.order_status.eq("delivered")
        & frame.order_delivered_customer_date.notna()
        & frame.order_estimated_delivery_date.notna()
        & frame.order_purchase_timestamp.notna()
        & frame.order_delivered_customer_date.ge(frame.order_purchase_timestamp)
        & frame.order_estimated_delivery_date.dt.normalize().ge(
            frame.order_purchase_timestamp.dt.normalize())
    )


def previous_order_counts(frame, days=7):
    """Поток прошлых покупок, без исходов. Заказы с равным временем не видят друг друга."""
    result = pd.Series(0,index=frame.index,dtype="int64")
    for _,group in frame.dropna(subset=["main_seller_id"]).groupby("main_seller_id",sort=False):
        times=group.order_purchase_timestamp.to_numpy(dtype="datetime64[ns]")
        sorted_times=np.sort(times)
        left=np.searchsorted(sorted_times,times-np.timedelta64(days,"D"),side="left")
        right=np.searchsorted(sorted_times,times,side="left")
        result.loc[group.index]=right-left
    return result


def features(frame):
    """Явный список исключает даты доставки, статусы, отзывы и будущие платежи."""
    x=frame.copy()
    x["log_items_value"]=np.log1p(x.items_value.clip(lower=0))
    x["log_freight_value"]=np.log1p(x.freight_value.clip(lower=0))
    x["freight_share"]=x.freight_value/(x.items_value+x.freight_value).replace(0,np.nan)
    x["log_seller_orders_prev_7d"]=np.log1p(x.seller_orders_prev_7d)
    x=x[FEATURES].copy()
    for col in CATEGORICAL:
        x[col]=x[col].fillna("unknown").astype(str)
    return x.replace([np.inf,-np.inf],np.nan)


def wilson(successes, n, z=1.96):
    if not n:
        return (None,None)
    p=successes/n
    center=(p+z*z/(2*n))/(1+z*z/n)
    half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return (float(center-half),float(center+half))


def priority_indices(scores, fraction=.2, ids=None):
    n=len(scores)
    if not 0<=fraction<=1:
        raise ValueError("Доля приоритета должна быть между 0 и 1.")
    secondary=np.arange(n) if ids is None else np.asarray(ids)
    return np.lexsort((secondary,-np.asarray(scores)))[:int(np.floor(n*fraction))]
