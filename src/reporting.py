"""Графики, отчёт и автономный дашборд на русском."""
import json
import numpy as np
import pandas as pd
from src.core import CATEGORY_LABELS,MODEL_LABELS


def build_reports(root,r):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,
        "axes.spines.right":False,"axes.titleweight":"bold","figure.facecolor":"#f4f6f4",
        "axes.facecolor":"#f4f6f4","savefig.facecolor":"#f4f6f4"})
    folder=root/"reports/figures"
    folder.mkdir(exist_ok=True)
    def save(fig,name):
        fig.tight_layout()
        fig.savefig(folder/name,dpi=160,bbox_inches="tight")
        plt.close(fig)
    monthly=pd.DataFrame(r["monthly"])
    fig,ax=plt.subplots(figsize=(11,4.5))
    ax.plot(np.arange(len(monthly)),monthly.late_rate*100,"o-",color="#d66d3e")
    ax.set_xticks(np.arange(len(monthly))[::2],monthly.month.iloc[::2],rotation=35,ha="right")
    ax.set(ylabel="Опоздания среди доставленных, %",title="Динамика нарушенных сроков")
    ax.grid(axis="y",alpha=.2)
    save(fig,"monthly_lateness.png")
    states=pd.DataFrame(r["states"]).sort_values("late_rate").tail(10)
    fig,ax=plt.subplots(figsize=(9,5))
    ax.errorbar(states.late_rate*100,np.arange(len(states)),
        xerr=[(states.late_rate-states.low)*100,(states.high-states.late_rate)*100],
        fmt="o",color="#137d78",capsize=4)
    ax.set_yticks(np.arange(len(states)),[f"{a.customer_state} · n={a.delivered:,}" for a in states.itertuples()])
    ax.set(xlabel="Опоздания, %; индивидуальный 95% ДИ Уилсона",title="Штаты покупателей с высокой долей опозданий")
    save(fig,"state_lateness.png")
    reviews=pd.DataFrame(r["reviews"])
    fig,ax=plt.subplots(figsize=(8,4.5))
    ax.bar(reviews.group,reviews.low_rate*100,color=["#137d78","#d66d3e"],width=.5)
    ax.errorbar(reviews.group,reviews.low_rate*100,
        yerr=[(reviews.low_rate-reviews.low)*100,(reviews.high-reviews.low_rate)*100],
        fmt="none",color="#30353d",capsize=5)
    ax.set(ylabel="Отзывы на 1–2 звезды, %",title="Связь отзывов и опозданий; не причинный эффект")
    save(fig,"reviews_and_delay.png")
    fig,ax=plt.subplots(figsize=(9,4.7))
    for row in r["pr_curves"]:
        ax.plot(row["recall"],row["precision"],lw=2,label=row["label"])
    ax.axhline(r["primary"]["prevalence"],color="#939b9f",ls="--",label="Доля опозданий")
    ax.set(xlabel="Полнота: доля найденных опозданий",ylabel="Точность: опоздания в приоритете",
           title="Проверка ранжирования на будущих заказах")
    ax.legend(frameon=False,fontsize=9)
    save(fig,"precision_recall.png")
    cal=pd.DataFrame(r["calibration"])
    fig,ax=plt.subplots(figsize=(7,5))
    ax.plot(cal.predicted,cal.observed,"o-",color="#137d78")
    limit=max(cal.predicted.max(),cal.observed.max())*1.1
    ax.plot([0,limit],[0,limit],"--",color="#939b9f")
    ax.set(xlabel="Средний прогноз в группе",ylabel="Фактическая доля опозданий",
           title="Калибровка модели на тесте")
    save(fig,"calibration.png")
    capacity=pd.DataFrame(r["capacity"])
    fig,ax=plt.subplots(figsize=(9,4.5))
    for name,g in capacity.groupby("model",sort=False):
        ax.plot(g.coverage*100,g.recall_at_budget*100,"o-",label=MODEL_LABELS[name])
    ax.set(xlabel="Доля заказов в приоритете, %",ylabel="Найдено фактических опозданий, %",
           title="Ресурс проверки: одинаковый бюджет для всех")
    ax.legend(frameon=False)
    save(fig,"priority_capacity.png")
    o,m,d=r["overview"],r["primary"],r["comparison_vs_rule"]
    p=lambda v:f"{v*100:.1f}%".replace(".",",")
    n=lambda v:f"{v:,.0f}".replace(","," ")
    verdict=("На тесте модель при 20% бюджете превосходит правило короткого срока."
        if d["low"]>0 else "Преимущество модели перед правилом короткого срока при 20% бюджете не установлено."
        if d["high"]>=0 else "При 20% бюджете модель уступила простому правилу.")
    rows=[]
    for a in r["model_metrics"]:
        brier="—" if a["brier"] is None else f'{a["brier"]:.4f}'
        rows.append(f'| {a["label"]} | {a["average_precision"]:.3f} | {a["roc_auc"]:.3f} | '
                    f'{p(a["precision_at_budget"])} | {p(a["recall_at_budget"])} | {brier} |')
    modelrows="\n".join(rows)
    state_rows="\n".join(f'| {a["customer_state"]} | {a["delivered"]} | {p(a["late_rate"])} |' for a in r["states"][:8])
    category_rows="\n".join(f'| {CATEGORY_LABELS.get(a["category"],a["category"])} | {a["delivered"]} | {p(a["late_rate"])} |' for a in r["categories"][:8])
    split_rows="\n".join(f'| {label} | {r["split_counts"][name]} | {r["split_late"][name]} |'
        for name,label in [("train","Подбор: обучение"),("validation","Валидация"),("final_fit","Финальное обучение"),("test","Тест")])
    report=f"""# Обещали доставить — результаты исследования

## Решение для бизнеса

Основная когорта: январь 2017 — август 2018, **{n(o["orders"])} заказов**.
Валидная фактическая доставка: **{n(o["delivered"])}**.
Из них **{n(o["late"])} ({p(o["late_rate"])})** доставлены позже обещанной календарной даты.

Отменено **{n(o["canceled"])} ({p(o["canceled_rate"])})** от всех заказов когорты.
Отмена не считается ни своевременной доставкой, ни доставленным опозданием.
Медиана доставки — {o["delivery_median"]:.1f} дня, 95-й процентиль — {o["delivery_p95"]:.1f} дня.

{verdict}
В 20% тестовых заказов с максимальным прогнозом найдены **{p(m["recall_at_budget"])} опозданий**.
Точность приоритета — **{p(m["precision_at_budget"])}**, доля опозданий на тесте — {p(m["prevalence"])}.
Это качество ранжирования, а не доказательство пользы обзвона или изменения логистики.

## Границы применимости

Модель оценена **среди доставленных заказов**, а не среди всех будущих покупок.
Будущая доставка неизвестна при оформлении: для внедрения нужна проверка на всех новых заказах,
включая отмены и ещё не закрытые доставки.

Обещанная дата, цена доставки, продавцы и каталог предполагаются доступными при покупке.
Истории изменения этих полей в наборе нет. До подтверждения доступности признаков
в нужный момент модель нельзя считать готовой к эксплуатации.
Данные 2017–2018 годов не гарантируют качество на современном маркетплейсе.

## Как считаются сроки

Опоздание: DATE(фактическая доставка) > DATE(обещанная доставка).
Доставка в любое время обещанного дня своевременна: полночь в поле даты не является
реальным обещанием получить товар к 00:00.
Из доставленных заказов когорты исключено по качеству дат: {o["timing_excluded"]}.
Пустые и противоречивые даты не заменяются нулевым опозданием.

![Динамика опозданий](figures/monthly_lateness.png)

## Отзывы и опоздания

По заказу берётся последний ответ на отзыв; равные даты разрешаются стабильным порядком.
В сравнение входят только ответы не раньше доставки.
Повторных записей относительно order_id: {r["quality"]["extra_review_records"]}.
Ответов до доставки исключено: {r["quality"]["reviews_before_delivery"]}.

| Доставка | Отзывов | Оценки 1–2 | Средняя оценка |
|---|---:|---:|---:|
| В срок | {r["reviews"][0]["n"]} | {p(r["reviews"][0]["low_rate"])} | {r["reviews"][0]["mean_score"]:.2f} |
| С опозданием | {r["reviews"][1]["n"]} | {p(r["reviews"][1]["low_rate"])} | {r["reviews"][1]["mean_score"]:.2f} |

Это наблюдаемая связь, не причинный эффект. Различаются состав товаров, регионы,
ожидания, продавцы и те, кто оставил отзыв. Потерянная прибыль не оценивалась.

![Отзывы](figures/reviews_and_delay.png)

## Где разбирать процессы

Штаты с минимум 200 доставленными заказами, по наблюдаемой доле:

| Штат | Доставлено | Опоздания |
|---|---:|---:|
{state_rows}

![Штаты](figures/state_lateness.png)

Категории с минимум 100 доставками. Для смешанных корзин берётся категория
с максимальной суммарной стоимостью, а не категория каждой позиции:

| Категория | Доставлено | Опоздания |
|---|---:|---:|
{category_rows}

[Список продавцов для проверки](tables/seller_watchlist.csv) использует только заказы
с одним продавцом и минимум 100 доставками. Ранжирование — по нижней границе
индивидуального интервала Уилсона. Доля многопродавцовых заказов: {p(o["multi_seller_share"])}.
Рейтинг исследовательский: различия в составе заказов и множественные сравнения не устранены.
Общую задержку многопродавцового заказа не приписываем одному продавцу.

## Проверка модели по времени

- Подбор: покупки 2017-01-01 — 2017-10-31, доставка известна до 2018-01-01.
- Валидация: покупки 2018-01-01 — 2018-02-28, исходы известны до 2018-05-01.
- Финальное обучение: покупки до 2018-03-01, исходы до 2018-05-01.
- Тест: покупки 2018-05-01 — 2018-06-30, исходы до 2018-10-18.
- Между этапами оставлены периоды дозревания исходов. Доставки после границы
  исключаются, что дополнительно ограничивает оцениваемую совокупность.
- 2018-10-18 — техническая граница после последней записи; дата выгрузки статусов неизвестна.

| Выборка | Заказов | Опозданий |
|---|---:|---:|
{split_rows}

Кандидаты: логистическая регрессия и градиентный бустинг.
Выбор по average precision на валидации: **{MODEL_LABELS[r["selection"]["selected"]]}**.
Пропуски и преобразования обучаются внутри Pipeline.
Решение сохранено в selection_lock.json до тестовой оценки.
Финальная модель обучается заново на доступном прошлом; тест не используется для подбора.

Признаки: срок обещания, корзина, цена и стоимость доставки, вес, объём, регионы,
категория, календарные признаки и прошлый поток заказов продавца за 7 дней.
Этот поток использует только покупки строго раньше текущего времени, без их исходов.
Заказы с равными временными метками не видят друг друга.
Фактические даты, статус, отзыв, shipping_limit_date и итоги оплаты исключены.

| Стратегия | AP | ROC-AUC | Точность @20% | Полнота @20% | Brier |
|---|---:|---:|---:|---:|---:|
{modelrows}

У рангового правила короткого срока нет вероятностной интерпретации — Brier не показывается.
Постоянный риск — частота опозданий финального обучения; случайный приоритет усреднён аналитически.
Бюджет равен floor(q × n); при равных прогнозах используется порядок анонимных order_id.

![Ранжирование](figures/precision_recall.png)

Разница полноты модели и правила при 20% бюджете: **{d["recall_difference"]*100:+.1f} п.п.**
Парный bootstrap-интервал: **[{d["low"]*100:+.1f}; {d["high"]*100:+.1f}] п.п.**
300 повторов, seed=42. Он условен на обученных моделях и не учитывает переобучение
и зависимость заказов продавца во времени; это не гарантия на новый период.

![Ресурс проверки](figures/priority_capacity.png)
![Калибровка](figures/calibration.png)

Важность признаков вычислена перемешиванием на валидации, только для описания.
Важный предиктор не обязательно является причиной.

## Качество витрины

После всех JOIN — {r["quality"]["mart_rows"]} строк и столько же уникальных order_id.
Товары, отзывы и платежи предварительно агрегированы до одного заказа.
Суммы сверены с исходными таблицами: товары {r["quality"]["reconciliation"]["items_value"]:.2f},
доставка {r["quality"]["reconciliation"]["freight_value"]:.2f},
платежи {r["quality"]["reconciliation"]["paid_value"]:.2f} BRL.
Эти суммы не означают чистую выручку или прибыль.
Заказов без товарных строк: {r["quality"]["orders_without_items"]}.
Проверены уникальность справочников, внешние ключи и неотрицательные суммы.

## Следующий шаг

Разобрать причины задержек в крупных проблемных сегментах.
Подтвердить доступность признаков при покупке и проверить модель на новой полной когорте.
Пользу обзвона или ускорения оценивать отдельным контролируемым экспериментом.

## Источник

Olist, Brazilian E-Commerce Public Dataset, версия 2:
[официальная страница](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce).
Данные и производные отчёты, графики, агрегаты — CC BY-NC-SA 4.0, атрибуция Olist.
Код проекта — MIT. Клиентские выгрузки исключены из сайта, Git и архива.

[Дашборд](dashboard.html) · [Словарь таблиц](tables/README_RU.md) · [Методика](../METHODOLOGY_RU.md)
"""
    (root/"reports/REPORT_RU.md").write_text(report,encoding="utf-8")
    html=(root/"src/dashboard.html").read_text(encoding="utf-8-sig")
    html=html.replace("__RESULTS__",json.dumps(r,ensure_ascii=False,allow_nan=False).replace("</","<\\/"))
    (root/"reports/dashboard.html").write_text(html,encoding="utf-8")
