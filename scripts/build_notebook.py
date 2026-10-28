"""Сборка выполненного учебного ноутбука на русском."""
import base64,contextlib,io,json,os,sys
from pathlib import Path
import nbformat
ROOT=Path(__file__).resolve().parents[1]


def main():
    cells=[]
    environment={}
    def md(text):
        cells.append(nbformat.v4.new_markdown_cell(text))
    def code(text):
        stream=io.StringIO()
        with contextlib.redirect_stdout(stream):
            exec(compile(text,"<notebook>","exec"),environment)
        cell=nbformat.v4.new_code_cell(text)
        cell.execution_count=sum(c.cell_type=="code" for c in cells)+1
        cell.outputs=[nbformat.v4.new_output("stream",name="stdout",text=stream.getvalue())]
        cells.append(cell)
    md("# Обещали доставить\n\nSQL, качество сервиса и честная проверка модели риска опоздания. "
       "Данные Olist, версия 2, CC BY-NC-SA 4.0. Пересборка: **python run.py**. "
       "Ноутбук читает результаты и независимо проверяет ключевые расчёты.")
    code("""from pathlib import Path
import json
import numpy as np
import pandas as pd
ROOT=Path.cwd()
if not (ROOT/'run.py').exists():
    ROOT=ROOT.parent
R=json.loads((ROOT/'reports/results.json').read_text(encoding='utf-8'))
print('Заказов в источнике:',R['quality']['source_rows']['orders'])
print('Строк в SQL-витрине:',R['quality']['mart_rows'])
print('Уникальных заказов:',R['quality']['mart_unique_orders'])
print('Суммы после JOIN:',R['quality']['reconciliation'])""")
    md("## 1. Бизнес-вопрос и правильный знаменатель\n"
       "Где нарушаются обещания и можно ли заранее выделить риск? "
       "Сроки оцениваем среди доставленных заказов с валидными датами. "
       "Отмены считаем отдельно по всем заказам. День обещания включён в своевременную доставку.")
    code("""o=R['overview']
print('Основная когорта:',o['orders'])
print('Доставлено с валидными датами:',o['delivered'])
print('Опоздали:',o['late'])
print('Доля опозданий:',round(o['late']/o['delivered'],6))
print('Доля отмен:',round(o['canceled']/o['orders'],6))
assert np.isclose(o['late']/o['delivered'],o['late_rate'])""")
    md("## 2. SQL без размножения денег\n"
       "Один заказ может иметь несколько товаров, оплат и отзывов. Сначала сворачиваем "
       "каждую таблицу до заказа, затем соединяем. Файл sql/01_order_mart.sql содержит CTE, "
       "оконные функции выбора основного продавца, категории и последнего отзыва.")
    code("""import duckdb
con=duckdb.connect(str(ROOT/'data/processed/analytics.duckdb'),read_only=True)
checks=con.sql('''
    SELECT COUNT(*) AS rows,COUNT(DISTINCT order_id) AS unique_orders,
           SUM(items_value) AS items_value,SUM(paid_value) AS paid_value
    FROM order_mart
''').df()
checks.columns=['Строк','Уникальных заказов','Стоимость товаров, BRL','Оплачено, BRL']
print(checks.to_string(index=False))
con.close()
assert checks.iloc[0,0]==checks.iloc[0,1]""")
    md("## 3. Отзывы: связь не равна причине\n"
       "Сравниваем последний ответ, оставленный не раньше доставки. "
       "Отбор по отзывам и различия заказов ограничивают вывод. Причинный ущерб или прибыль не рассчитываем.")
    code("""reviews=pd.read_csv(ROOT/'reports/tables/review_comparison.csv')
print(reviews.rename(columns={'group':'Доставка','n':'Отзывов','low_rate':'Доля оценок 1–2',
    'mean_score':'Средняя оценка'})[['Доставка','Отзывов','Доля оценок 1–2','Средняя оценка']].to_string(index=False))
print('Повторных записей отзывов:',R['quality']['extra_review_records'])
print('Ответов до доставки исключено:',R['quality']['reviews_before_delivery'])""")
    md("## 4. Проверка по времени и доступность признаков\n"
       "Обучение — 2017 год, валидация — январь–февраль 2018, тест — май–июнь. "
       "Промежутки нужны для дозревания исходов. Перед тестом финально обучаемся на доступном прошлом. "
       "Обещание, корзина, назначенный продавец и каталог предполагаются известными при покупке; "
       "истории их изменений в наборе нет.")
    code("""splits=pd.read_csv(ROOT/'data/processed/model_splits.csv',
    parse_dates=['order_purchase_timestamp','order_delivered_customer_date'])
for name,label in [('train','Обучение подбора'),('validation','Валидация'),
                   ('final_fit','Финальное обучение'),('test','Тест')]:
    g=splits[splits.split==name]
    print(label,len(g),'заказов; покупки',g.order_purchase_timestamp.min(),'—',g.order_purchase_timestamp.max())
fit=splits[splits.split=='final_fit']
test=splits[splits.split=='test']
assert not set(fit.order_id)&set(test.order_id)
assert fit.order_delivered_customer_date.max()<test.order_purchase_timestamp.min()
print('Обучение не использует ещё неизвестные тестовые исходы: да')
print('Модель:',R['primary']['label'])
print('Тест использован для выбора:',R['selection']['test_used_for_selection'])""")
    md("## 5. Модель и простая альтернатива\n"
       "Кандидат выбирается по average precision на валидации. Сравниваем с постоянным риском "
       "и правилом короткого обещанного срока. Последнее задаёт ранжирование, но не вероятность. "
       "Поэтому Brier для правила не интерпретируем.")
    code("""m=pd.read_csv(ROOT/'reports/tables/test_model_metrics.csv')
print(m[['label','average_precision','roc_auc','precision_at_budget','recall_at_budget','brier']].rename(columns={
    'label':'Стратегия','average_precision':'AP','roc_auc':'ROC-AUC','precision_at_budget':'Точность при 20%',
    'recall_at_budget':'Полнота при 20%','brier':'Brier'}).to_string(index=False))""")
    md("## 6. Независимый пересчёт бюджета\n"
       "На тесте фиксированный бюджет 20%. При равных оценках порядок задаётся order_id. "
       "Проверяем расчёт напрямую, без функции расчёта метрик из проекта.")
    code("""h=pd.read_csv(ROOT/'data/processed/test_predictions.csv')
k=int(np.floor(.2*len(h)))
selected=np.lexsort((h.order_id.to_numpy(),-h.probability.to_numpy()))[:k]
precision=h.iloc[selected].late.mean()
recall=h.iloc[selected].late.sum()/h.late.sum()
print('Всего тестовых заказов:',len(h),'В приоритете:',k)
print('Опозданий в приоритете:',int(h.iloc[selected].late.sum()))
print('Точность:',round(precision,5),'Полнота:',round(recall,5))
assert np.isclose(precision,R['primary']['precision_at_budget'])
assert np.isclose(recall,R['primary']['recall_at_budget'])
print('Сохранённые результаты воспроизведены: да')""")
    image=base64.b64encode((ROOT/'reports/figures/priority_capacity.png').read_bytes()).decode()
    picture=nbformat.v4.new_code_cell("# График из воспроизводимого анализа: python run.py")
    picture.execution_count=sum(c.cell_type=="code" for c in cells)+1
    picture.outputs=[nbformat.v4.new_output("display_data",data={
        "image/png":image,"text/plain":"Ресурс проверки и доля найденных опозданий"})]
    cells.append(picture)
    md("## 7. Что проверить перед внедрением\n"
       "Модель оценена условно на доставленных заказах. Нужна полная новая когорта, "
       "подтверждение доступности полей в момент покупки, контроль калибровки и изменения частоты опозданий. "
       "Пользу реального вмешательства оцениваем отдельным экспериментом. "
       "Текущий bootstrap не учитывает переобучение и временную зависимость заказов продавца.")
    code("""print('Доля опозданий по выборкам:')
for name in ['train','validation','test']:
    print(name,round(R['split_late'][name]/R['split_counts'][name]*100,2),'%')
print('Разница полноты модели и правила:',R['comparison_vs_rule'])
print('Дашборд: reports/dashboard.html')
print('Полный отчёт: reports/REPORT_RU.md')""")
    notebook=nbformat.v4.new_notebook(cells=cells,metadata={
        "kernelspec":{"name":"python3","display_name":"Python 3","language":"python"},
        "language_info":{"name":"python","version":"3.12"}})
    nbformat.validate(notebook)
    nbformat.write(notebook,ROOT/"notebooks/01_delivery_analysis.ipynb")
    print("Выполненный ноутбук сохранён.")


if __name__=="__main__":
    os.chdir(ROOT)
    main()
