import urllib.request
from bs4 import BeautifulSoup
import xlwt
from lxml import etree

#写入excel文件
def save_excel(sheet_name, Contents, Leader, PhoneNum1, PhoneNum2):
    '''保存数据到Excel'''
    # 创建一个workbook 设置编码
    workbook = xlwt.Workbook()
    # 创建一个worksheet
    worksheet = workbook.add_sheet(sheet_name)
    worksheet.write(0, 0, label="公司名称")
    worksheet.write(0, 1, label="联系人")
    worksheet.write(0, 2, label="电话")
    worksheet.write(0, 3, label="手机号码")
    try:
        for i in range(len(Contents)):
            if len(Contents[i]) > 1:
                print(i)
                content_list = Contents[i]
                leader_list = Leader[i]
                phoneNum_list1 = PhoneNum1[i]
                phoneNum_list2 = PhoneNum2[i]
                # 写入excel
                # 参数对应 行, 列, 值
                worksheet.write(i + 1, 0, label=content_list)
                worksheet.write(i + 1, 1, label=leader_list)
                worksheet.write(i + 1, 2, label=phoneNum_list1)
                worksheet.write(i + 1, 3, label=phoneNum_list2)
                # 保存
                workbook.save(sheet_name + '.xls')
                # time.sleep(0.1)
    except:
        print(sheet_name, '保存OK')
        pass


page = urllib.request.urlopen('http://www.likuso.com/city178/p1/')  #请求站点获得一个HTTPResponse对象
contents = page.read().decode('utf-8', 'ignore')
#总的数据条数
total = etree.HTML(contents).xpath('//div[@class="breadFind"]/span/text()')
print(total[0])
items = etree.HTML(contents).xpath('//h2[@class="clearfix" or @class="clearfix .backcolor"]/a/@href')

CompanyName = []
Linker = []
TelNum = []
PhoneNum = []
i = 0
for item in items:
    i = i + 1
    if (i <= 2):
        itemRes = urllib.request.urlopen(item).read().decode('utf-8', 'ignore')
        companyName = etree.HTML(itemRes).xpath('//span[contains(text(),"全称")]/..//text()')[1]
        linker = etree.HTML(itemRes).xpath('//span[contains(text(),"联系人")]/..//text()')[1]
        if (etree.HTML(itemRes).xpath('//span[contains(text(),"电话")]/..//text()') == []):
            telNum = '暂无'
        else:
            telNum = etree.HTML(itemRes).xpath('//span[contains(text(),"电话")]/..//text()')[1]
        phoneNum = etree.HTML(itemRes).xpath('//span[contains(text(),"手机")]/..//text()')[1]
        print(companyName, linker,telNum, phoneNum)
        CompanyName.append(companyName)
        Linker.append(linker)
        TelNum.append(telNum)
        PhoneNum.append(phoneNum)
save_excel("联系", CompanyName, Linker, TelNum, PhoneNum)