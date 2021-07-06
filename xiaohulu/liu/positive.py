# encoding: utf-8

"""
@author: sunxianpeng
@file: positive.py
@time: 2021/7/6 11:19
"""
from sqlalchemy import *
from sqlalchemy.engine import create_engine
from sqlalchemy.schema import *
import pandas as pd


def read_presto_data_test(sql):
    # Presto
    engine = create_engine(
        'presto://113.107.166.14:28080/')
    df = pd.read_sql(sql, engine)
    return df


if __name__ == '__main__':
    platform_id = 71
    cycle = "('" + "','".join(['2021-02-08', '2021-02-15', '2021-02-22', '2021-03-01', '2021-03-08', '2021-03-15', '2021-03-22','2021-03-29']) + "')"

    sql_posotive = """WITH tb_base AS (
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
        FROM((SELECT
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
    )) ,

tb_mid_1 as (
    SELECT
    	platform_id,
    	room_id,
    	sales_number_add_sum,
    	sales_price_add,
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
        IF(sales_number_add_sum_avg_pre=0,if(sales_number_add_sum_avg=0,0,999),round(((sales_number_add_sum_avg-sales_number_add_sum_avg_pre )*1.0)/sales_number_add_sum_avg_pre, 2)) AS sales_number_add_sum_avg_percent,
        IF( sales_price_add_avg_pre = 0.0, if(sales_price_add_avg=0.0,0,999),round(((sales_price_add_avg - sales_price_add_avg_pre )*1.0)/sales_price_add_avg_pre,2)) AS sales_price_add_avg_percent,
        statistics_date
    FROM(
    	SELECT
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
    -- 	在八个周期中直播天数大于等于6个周期的主播数据
    		( SELECT platform_id, room_id, count(*) AS live_cycles FROM tb_base GROUP BY platform_id, room_id HAVING count( * ) >= 5) t1
    		INNER JOIN (select * from tb_base where rank != 1) t2 ON t1.platform_id = t2.platform_id AND t1.room_id = t2.room_id ) order by statistics_date asc
),
tb_mid_2 as (
    select * from tb_mid_1 where sales_number_add_sum_avg_add >0 -- and sales_price_percent>0.00
),
tb_increase_bigger_n as (
    select  t4.platform_id,
        	t4.room_id,
        	t3.live_cycles,
        	t4.sales_number_add_sum,
        	t4.sales_price_add,
        	t4.sales_number_add_sum_avg,
        	t4.sales_price_add_avg,
        	t4.sales_number_add_sum_avg_pre,
        	t4.sales_price_add_avg_pre,
        	t4.sales_number_add_sum_avg_pre_2,
    	    t4.sales_price_add_avg_pre_2,
        	t4.sales_number_add_sum_avg_add,
        	t4.sales_price_add_avg_add,
        	t4.sales_number_add_sum_avg_add_2,
        	t4.sales_price_add_avg_add_2,
            t4.statistics_date
            -- 直播天数大于等于6个周期的主播中，其中有5个周期天销量和销售额增长的主播
            from (select platform_id,room_id, count(*) AS live_cycles from tb_mid_2 group by platform_id,room_id having count(*) >= 4) t3 inner join tb_mid_2 t4 
        on t3.platform_id = t4.platform_id AND t3.room_id = t4.room_id order by t3.platform_id,t3.room_id,statistics_date asc
    ),
tb_increase_end as (
    select * from tb_mid_1 where room_id in (select room_id from tb_increase_bigger_n )
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
    	AND fansCount >= 500 and fansCount <= 50000 AND totalViewerMax >= 100  ) t3
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
SELECT	* FROM
	( SELECT platform_id, room_id, sales_number_add_sum_avg_add, sales_price_add_avg_add,sales_number_add_sum_avg_add_2,sales_price_add_avg_add_2, statistics_date
	FROM tb_increase_end ORDER BY platform_id, room_id, statistics_date ASC ) t1
	INNER JOIN (
SELECT	* FROM
	( SELECT *, row_number () over ( PARTITION BY platform_id, room_id ORDER BY statistics_date ASC ) AS rank FROM tb_data ) 
WHERE
	rank = 3
	) t2 ON t1.platform_id = t2.platform_id AND t1.room_id = t2.room_id AND t1.statistics_date = t2.statistics_date
""".format(platform_id, cycle, platform_id, cycle, platform_id, platform_id, cycle, platform_id, cycle)
    print(sql_posotive)
    # df = read_presto_data_test(sql_posotive)
    # print(df.head(5))
