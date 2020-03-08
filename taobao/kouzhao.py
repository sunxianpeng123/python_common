# -*- coding: utf-8 -*-
# @Time : 2020/3/7 23:02
# @Author : sxp
# @Email : 
# @File : kouzhao.py
# @Project : python_common

# 淘宝秒杀脚本，扫码登录版
from selenium import webdriver
import datetime
import time

def affirm_brower(name):
    browser = None
    if name == "chrome":
        ua = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.132 Safari/537.36'
        options = webdriver.ChromeOptions()
        options.add_argument('--ignore-certificate-errors')
        # options.add_argument('--user-data-dir=C:/Users\sxp\AppData\Local\Google\Chrome/User Data\Default')  # 设置成用户自己的数据目录
        options.add_argument('user-agent=' + ua)
        browser = webdriver.Chrome(options=options)
        browser.maximize_window()
    elif name == "firefox":
        ua = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:73.0) Gecko/20100101 Firefox/73.0'
        profile = webdriver.FirefoxProfile('C:/Users\sxp\AppData\Local\Mozilla\Firefox\Profiles\default')
        profile.accept_untrusted_certs = True
        profile.set_preference('general.useragent.override', ua)

        browser = webdriver.Firefox(firefox_profile=profile)
        browser.maximize_window()
    return browser


def login(browser):
    # 打开淘宝首页，通过扫码登录
    browser.get("https://www.taobao.com")
    time.sleep(3)
    if browser.find_element_by_link_text("亲，请登录"):
        browser.find_element_by_link_text("亲，请登录").click()
        print(f"请尽快扫码登录")
        time.sleep(10)
    now = datetime.datetime.now()
    print('login success:', now.strftime('%Y-%m-%d %H:%M:%S'))

def picking(choose,select_all_button):
    # 打开购物车列表页面
    browser.get("https://cart.taobao.com/cart.htm")
    time.sleep(3)
    # 是否全选购物车
    if choose == 1:
        while True:
            try:
                if browser.find_element_by_id(select_all_button):
                    browser.find_element_by_id(select_all_button).click()
                    break
            except:
                print(f"找不到购买按钮")
    else:
        print(f"请手动勾选需要购买的商品")
        time.sleep(5)

def buy(begin_time):
    while True:
        now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        # 对比时间，时间到的话就点击结算
        if now > begin_time:
            # 点击结算按钮
            while True:
                try:
                    if browser.find_element_by_link_text("结 算"):
                        browser.find_element_by_link_text("结 算").click()
                        print(f"结算成功，准备提交订单")
                        break
                except:
                    pass
            # 点击提交订单按钮
            while True:
                try:
                    if browser.find_element_by_link_text('提交订单'):
                        browser.find_element_by_link_text('提交订单').click()
                        print(f"抢购成功，请尽快付款")
                except:
                    print(f"再次尝试提交订单")
            time.sleep(0.01)

if __name__ == '__main__':
    # url
    # brower_name = "firefox"
    brower_name = "chrome"
    test = "https://www.baidu.com"
    taobao = "https://www.taobao.com"
    select_all_button = "J_SelectAll2"

    # 时间格式："2018-09-06 11:20:00.000000"
    begin_time = "2020-03-08 04:01:00.000000"#input("请输入抢购时间，格式如(2020-03-08 02: 41:00.000000):")
    choose = 1#int(input("到时间自动勾选购物车请输入“1”，否则输入“2"))
    #

    browser = affirm_brower(brower_name)
    # for test
    # browser.get(taobao)
    login(browser)
    picking(choose, select_all_button)
    buy(begin_time)



