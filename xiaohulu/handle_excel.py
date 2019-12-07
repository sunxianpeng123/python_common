# encoding: utf-8

"""
@author: sunxianpeng
@file: handle_excel.py
@time: 2019/10/16 17:19
"""
import pandas as pd
from sqlalchemy.engine import create_engine
class Main():
    def __init__(self):
        pass
    def read_excel(self,path):
        data = pd.read_excel(path)
        return data
    def read_mysql_data(self,sql):
        engine = create_engine("mysql+pymysql://xiaohulu:6DgajISh!@113.107.166.14:53306/short_video_platform")
        df = pd.read_sql_query(sql,engine)
        return df
    def df_merge(self,left, right, merge_keys,join_type):
        res_df = pd.merge(left, right, on=merge_keys, how=join_type)
        return  res_df
    def save_df_to_excel(self,df, path):
        df.to_excel(path)

if __name__ == '__main__':
    m = Main()
    platform_id = "201"
    how = "left"

    name = "month_8_新增视频数"
    path = u"F:\PythonProjects\python_study/xiaohulu\data/"+name+".xlsx"
    write_path = u"F:\PythonProjects\python_study/xiaohulu\data/"+name+"new.xlsx"
    video_tag_name_sql_201 = "select tag_id,tag_name from  anchor_tag_config where platform_id =" + platform_id
    df = m.read_excel(path)
    tag_name_df = m.read_mysql_data(video_tag_name_sql_201)
    res_with_tag_name_df = m.df_merge(df, tag_name_df, ["tag_id"], how)

    m.save_df_to_excel(res_with_tag_name_df,write_path)
