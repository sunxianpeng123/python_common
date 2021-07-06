# encoding: utf-8

"""
@author: sunxianpeng
@file: Validate.py
@time: 2021/7/6 14:42
"""
from sqlalchemy import *
from sqlalchemy.engine import create_engine
from sqlalchemy.schema import *
import pandas as pd
def read_presto_data(sql):
    # Presto
    engine = create_engine('presto://113.107.166.14:28080/')
    df = pd.read_sql(sql, engine)
    return df


if __name__ == '__main__':
    platform_id = 71
    cycle = "('" + "','".join(['2021-04-03', '2021-04-10', '2021-04-17', '2021-04-24']) + "')"
    cycle_first_datetime = "2021-03-28 00:00:00"
    cycle_last_datetime ="2021-04-24 23:59:59"
    sql_validate = """WITH tb_base AS (
    select *,
    	lag ( sales_number_add_sum_avg, 1, 0 ) over ( PARTITION BY platform_id, room_id ORDER BY statistics_date ASC ) AS sales_number_add_sum_avg_pre,
    	lag ( sales_price_add_avg, 1, 0 ) over ( PARTITION BY platform_id, room_id ORDER BY statistics_date ASC ) AS sales_price_add_avg_pre,
    	lag ( sales_number_add_sum_avg, 2, 0 ) over ( PARTITION BY platform_id, room_id ORDER BY statistics_date ASC ) AS sales_number_add_sum_avg_pre_2,
    	lag ( sales_price_add_avg, 2, 0 ) over ( PARTITION BY platform_id, room_id ORDER BY statistics_date ASC ) AS sales_price_add_avg_pre_2,
    	row_number ( ) over ( PARTITION BY platform_id, room_id ORDER BY statistics_date ASC ) AS rank
    from (
        SELECT
        	t2.platform_id,
        	t2.room_id,
        	t1.live_day_count,
        	t2.sales_number_add_sum,
        	t2.sales_price_add,
        	round((t2.sales_number_add_sum*1.0)/t1.live_day_count, 2 ) as sales_number_add_sum_avg,
        	round((t2.sales_price_add*1.0)/t1.live_day_count, 2 ) as sales_price_add_avg,
        	t2.statistics_date 
        FROM
        	(
        	(
            SELECT
            	platform_id,
            	room_id,
            	live_day_count,
            	cast ( statistics_date AS VARCHAR ) AS statistics_date 
            FROM
            	mysql149.anchor_live_schedule.anchor_week_statistics 
            WHERE
            	platform_id = {} 
            	AND cast ( statistics_date AS VARCHAR ) IN {} 
        	) t1
        	INNER JOIN (
            SELECT
            	platform_id,
            	room_id,
            	sales_number_add_sum,
            	sales_price_add,
            	cast ( statistics_date AS VARCHAR ) AS statistics_date 
            FROM
            	mysql149.anchor_goods_info.anchor_sales_week_info 
            WHERE
            	platform_id = {} 
            	AND cast ( statistics_date AS VARCHAR ) IN {} 
        ) t2 ON t1.platform_id = t2.platform_id AND t1.room_id = t2.room_id AND t1.statistics_date = t2.statistics_date )
    )
) ,
tb_mid_1 as (
    SELECT
    	platform_id,
    	room_id,
    	live_cycles,
    	sales_number_add_sum_avg,
    	sales_price_add_avg,
    	sales_number_add_sum_avg_pre,
    	sales_price_add_avg_pre,
    	sales_number_add_sum_avg_pre_2,
    	sales_price_add_avg_pre_2,
    	sales_number_add_sum_avg - sales_number_add_sum_avg_pre as sales_number_add_sum_avg_add,
    	sales_price_add_avg - sales_price_add_avg_pre as sales_price_add_avg_add,
    	sales_number_add_sum_avg_pre - sales_number_add_sum_avg_pre_2 as sales_number_add_sum_avg_add_2,
    	sales_price_add_avg_pre - sales_price_add_avg_pre_2 as sales_price_add_avg_add_2,
        statistics_date
    FROM(
    	SELECT
    	    t1.live_cycles,
    		t2.platform_id,
    		t2.room_id,
    		t2.sales_number_add_sum,
    		t2.sales_price_add,
    		t2.sales_number_add_sum_avg,
    		t2.sales_price_add_avg,
    		t2.sales_number_add_sum_avg_pre,
    		t2.sales_price_add_avg_pre,
    		t2.sales_number_add_sum_avg_pre_2,
    	    t2.sales_price_add_avg_pre_2,
    		t2.statistics_date
    	FROM
    		( SELECT platform_id, room_id, count(*) AS live_cycles FROM tb_base GROUP BY platform_id, room_id HAVING count( * ) >= 3) t1
    		INNER JOIN tb_base t2 ON t1.platform_id = t2.platform_id AND t1.room_id = t2.room_id ) order by statistics_date asc
),
tb_mid_2 as (
    SELECT 
        platform_id, 
        room_id, 
        sales_number_add_sum_avg_add, 
        sales_price_add_avg_add,
        sales_number_add_sum_avg_add_2,
        sales_price_add_avg_add_2, 
        statistics_date
    FROM (select *, row_number () over ( PARTITION BY platform_id, room_id ORDER BY statistics_date ASC) as rank from tb_mid_1) 
    where rank = 3 ORDER BY platform_id, room_id, statistics_date ASC
),
tb_mid_3_first_live_time as (
    SELECT
    	platform_id,
    	room_id,
    	first_live_time
    	FROM mysql149.anchor_live_schedule.anchor_base_info
    	WHERE platform_id = {} AND cast(first_live_time as varchar) >= '{}' AND cast(first_live_time as varchar)  <= '{}' 
),
tb_end as (
    SELECT 
        t1.platform_id, 
        t1.room_id, 
        t2.first_live_time,
        t1.sales_number_add_sum_avg_add, 
        t1.sales_price_add_avg_add,
        t1.sales_number_add_sum_avg_add_2,
        t1.sales_price_add_avg_add_2, 
        t1.statistics_date
    FROM (tb_mid_2 t1 INNER JOIN  tb_mid_3_first_live_time t2 ON t1.platform_id = t2.platform_id AND t1.room_id = t2.room_id )
),
tb_data as (
    select 
    	t3.platform_id,
    	t3.room_id,
    	t3.fansCount,
    	t3.fansCount_add,
    	t3.tyrant_count_sum,
    	t3.totalViewerMax,
    	t3.totalViewerSum,
    	t3.onlineViewerMax,
    	t3.dyValue_add,
    	t3.live_day_count,
    	t3.live_airtime_time,
        t4.sales_number_add_sum,
    	t4.sales_price_add,
    	t4.sales_price_average,
    	-- 销售效率（周销售金额/周直播时长）
    	if(live_airtime_time=0,0,round((t4.sales_price_add*1.0) / t3.live_airtime_time,2)) as sales_efficiency,
    	t4.goods_num,
    	t4.sales_number_week_add,
    	t4.sales_price_week_add,
    	t4.statistics_date
    from
    (SELECT
    	platform_id,
    	room_id,
    	fansCount,
    	fansCount_add,
    	tyrant_count_sum,
    	totalViewerMax,
    	totalViewerSum,
    	onlineViewerMax,
    	dyValue_add,
    	live_day_count,
    	live_airtime_time,
    	cast (statistics_date AS varchar )  as statistics_date
    FROM
    	mysql149.anchor_live_schedule.anchor_week_statistics 
    WHERE
    	platform_id = {} 
    	AND fansCount >= 500 and fansCount <= 50000 AND totalViewerMax >= 100   ) t3
    inner join(
    select 
    	t1.platform_id,
    	t1.room_id,
    	sales_number_add_sum,
    	sales_price_add,
    	sales_price_average,
    	goods_num,
    	cast ( t1.statistics_date AS varchar )  as statistics_date ,
    	sales_number_week_add,
    	sales_price_week_add
    	from
    (SELECT
    	platform_id,
    	room_id,
    	sales_number_add_sum,
    	sales_price_add,
    	if(sales_number_add_sum=0,0,round((sales_price_add*1.0) / sales_number_add_sum,2)) as sales_price_average,
    	goods_num,
    	cast ( statistics_date AS varchar )  as statistics_date
    FROM
    	mysql149.anchor_goods_info.anchor_sales_week_info 
    WHERE
    	platform_id = {} 
    	AND cast ( statistics_date AS varchar ) IN {}
    	) t1
    left join 
    (
    select 
    	platform_id,
    	room_id,
    	sales_number as sales_number_week_add,
    	sales_price as sales_price_week_add,
    	statistics_date
    from (
    select * from (SELECT
    	platform_id,
    	room_id,
    	sales_number_add_sum,
    	sales_price_add,
    	statistics_date,
    	sales_number_add_sum - lag ( sales_number_add_sum, 1, 0 ) over ( PARTITION BY platform_id, room_id ORDER BY statistics_date asc ) AS sales_number ,
        sales_price_add - lag ( sales_price_add, 1, 0 ) over ( PARTITION BY platform_id, room_id ORDER BY statistics_date asc  )  AS sales_price
    FROM
    	(
    	SELECT
    		platform_id,
    		room_id,
    		sales_number_add_sum,
    		sales_price_add,
    		cast ( statistics_date AS VARCHAR ) AS statistics_date 
    	FROM
    		mysql149.anchor_goods_info.anchor_sales_week_info 
    	WHERE
    		platform_id = {} 
    	AND cast ( statistics_date AS VARCHAR ) IN {} )
    ))) t2 on t1.platform_id = t2.platform_id and t1.room_id = t2.room_id and t1.statistics_date = t2.statistics_date
    ) t4 on t3.platform_id = t4.platform_id and t3.room_id = t4.room_id and t3.statistics_date = t4.statistics_date
)
SELECT	t1.first_live_time,
        t1.sales_number_add_sum_avg_add, 
        t1.sales_price_add_avg_add,
        t1.sales_number_add_sum_avg_add_2,
        t1.sales_price_add_avg_add_2, 
        t2.* 
FROM tb_end t1 INNER JOIN tb_data t2 ON t1.platform_id = t2.platform_id AND t1.room_id = t2.room_id AND t1.statistics_date = t2.statistics_date
    """.format(platform_id, cycle, platform_id, cycle, platform_id, cycle_first_datetime, cycle_last_datetime,platform_id, platform_id, cycle, platform_id, cycle)
    df = read_presto_data(sql_validate)
    print(df.head())