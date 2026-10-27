-- Для сравнения продавцов используем только заказы с одним продавцом.
-- Ранжируем нижнюю границу интервала Уилсона, а не шумную долю на малой выборке.
WITH counts AS (
 SELECT main_seller_id AS seller_id, MIN(seller_state) AS seller_state,
        COUNT(*) AS delivered, SUM((delay_days>0)::INTEGER) AS late
 FROM analysis_mart WHERE timing_valid AND sellers_count=1
 GROUP BY main_seller_id HAVING COUNT(*)>=100
), rates AS (SELECT *,late::DOUBLE/delivered AS late_rate FROM counts)
SELECT *, (late_rate+1.96*1.96/(2*delivered)-1.96*SQRT(
   late_rate*(1-late_rate)/delivered+1.96*1.96/(4*delivered*delivered)))/
   (1+1.96*1.96/delivered) AS wilson_low
FROM rates ORDER BY wilson_low DESC,seller_id;
