# encoding=utf-8
import requests
from lxml import etree
import json

headers = {
    'authority': 'www.block123.com',
    'sec-ch-ua': '\\',
    'sec-ch-ua-mobile': '?0',
    'upgrade-insecure-requests': '1',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.77 Safari/537.36',
    'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
    'sec-fetch-site': 'same-origin',
    'sec-fetch-mode': 'navigate',
    'sec-fetch-user': '?1',
    'sec-fetch-dest': 'document',
    'referer': 'https://www.block123.com/zh-hans/c/016707973922.htm',
    'accept-language': 'zh,en-US;q=0.9,en;q=0.8,zh-CN;q=0.7',
    'cookie': '_ga=GA1.2.442623307.1622617744; _gid=GA1.2.229102503.1623933306; Hm_lvt_4f7c3b36ed0491f90852e05efd64e62f=1622617744,1622621492,1623762242,1623933306; messages=\\1992be1b89924165398e30257015c9c397531294$[[\\\\\\__json_message\\\\\\\\\\0540\\\\05420\\\\054\\\\\\Confirmation e-mail sent to 1231241421@qq.com.\\\\\\]\\\\054[\\\\\\__json_message\\\\\\\\\\0540\\\\05425\\\\054\\\\\\\\\\\\\\u4ee5 1231241421@qq.com..\\\\\\\\u8eab\\\\\\\\u4efd\\\\\\\\u6210\\\\\\\\u529f\\\\\\\\u767b\\\\\\\\u5f55\\\\\\]]\\; csrftoken=he7WlaO5LqsrAAk2ypfBqkrUEEk3s89vdkihajhtf8GVVI7kqnp2RLDPpYvn21NY; sessionid=rdm3v9oya9tj6qp3c7hu4y5hotlglt95; Hm_lpvt_4f7c3b36ed0491f90852e05efd64e62f=1623933663',
}


def getprojectlist():
    urllist = [
        'https://www.block123.com/zh-hans/c/184982400121.htm', 'https://www.block123.com/zh-hans/c/400455894778.htm',
        'https://www.block123.com/zh-hans/c/852112675460.htm', 'https://www.block123.com/zh-hans/c/182175709311.htm',
        'https://www.block123.com/zh-hans/c/213256804209.htm', 'https://www.block123.com/zh-hans/c/016707973922.htm',
    ]
    for url in urllist:
        index = 1
        baseurl = url + "?page=%s"
        while 1:
            url2 = baseurl % index
            print(url2)
            r = requests.get(url2, headers=headers)
            tree = etree.HTML(r.text)
            print(r.status_code, r.url)
            for info in tree.xpath(".//div[@class='navs-list-content']//li/a/@href"):
                print("https://www.block123.com" + info)
                with open("projectlist.txt", "a+", encoding='utf-8') as f:
                    f.write("https://www.block123.com" + info + "\n")
            if tree.xpath(".//li[@class='last disabled']"):
                break
            else:
                index += 1


# getprojectlist()

def getdetailinfo(url):
    r = requests.get(url, headers=headers)
    tree = etree.HTML(r.text)
    info = {}
    info["type"] = tree.xpath("//ol[@class='breadcrumb']/li[2]/a/text()")[0]
    info["name"] = tree.xpath(".//*[@class='nav-name']/text()")[0]
    info["brief"] = "".join(tree.xpath(".//*[@class='bio-wrapper']/text()"))
    info["logo"] = tree.xpath(".//div[@class='left-item']/img/@src")[-1]
    info["tag"] = "".join(tree.xpath(".//div[@class='tags-wrapper']/a/text()"))
    info["desc"] = "\n".join(tree.xpath(".//div[@class='desc-content item-content']/p/text()"))
    for item in tree.xpath(".//div[@class='nav-detail-portfolio-wrapper item-box']"):
        name = item.xpath(".//div[@class='item-title']/text()")[0].split("\xa0")[0]
        clist = []
        for ii in tree.xpath(".//div[@class='portfolio-content item-content']/a/@href"):
            clist.append("https://www.block123.com" + ii)
        info[name] = []
        info[name] = clist
    info["website"] = "".join(tree.xpath(".//div[@class='web-site']/a/text()")).strip()
    info["twitter"] = ""
    info["facebook"] = ""
    info["facebook"] = "linkedin"
    for i in tree.xpath(".//div[@class='social-list']/a/@href"):
        if "twitter" in i:
            info["twitter"] = i.replace("?ref=block123", "")
        if "facebook" in i:
            info["facebook"] = i.replace("?ref=block123", "")
        if "linkedin" in i:
            info["linkedin"] = i.replace("?ref=block123", "")
    with open("detailinfo.txt", "a+", encoding='utf-8') as ff:
        ff.write(json.dumps(info, ensure_ascii=False) + "\n")


# with open("projectlist.txt","r", encoding='utf-8') as f:
#     for line in f.readlines():
#         line = line.strip()
#         getdetailinfo(line)

def getmemberinfo(url):
    r = requests.get(url, headers=headers)
    tree = etree.HTML(r.text)
    info = {}
    info["type"] = tree.xpath("//ol[@class='breadcrumb']/li[2]/a/text()")[0]
    info["name"] = tree.xpath(".//*[@class='nav-name']/text()")[0]
    info["brief"] = "".join(tree.xpath(".//*[@class='bio-wrapper']/text()"))
    info["logo"] = tree.xpath(".//div[@class='left-item']/img/@src")[-1]
    info["tag"] = "".join(tree.xpath(".//div[@class='tags-wrapper']/a/text()"))
    info["desc"] = "\n".join(tree.xpath(".//div[@class='desc-content item-content']/p/text()"))
    for item in tree.xpath(".//div[@class='nav-detail-portfolio-wrapper item-box']"):
        name = item.xpath(".//div[@class='item-title']/text()")[0].split("\xa0")[0]
        clist = []
        for ii in tree.xpath(".//div[@class='portfolio-content item-content']/a/@href"):
            clist.append("https://www.block123.com" + ii)
        info[name] = []
        info[name] = clist
    info["website"] = "".join(tree.xpath(".//div[@class='web-site']/a/text()")).strip()
    info["twitter"] = ""
    info["facebook"] = ""
    info["linkedin"] = ""
    for i in tree.xpath(".//div[@class='social-list']/a/@href"):
        if "twitter" in i:
            info["twitter"] = i.replace("?ref=block123", "")
        if "facebook" in i:
            info["facebook"] = i.replace("?ref=block123", "")
        if "linkedin" in i:
            info["linkedin"] = i.replace("?ref=block123", "")
    with open("memberlinfo.txt", "a+", encoding='utf-8') as ff:
        ff.write(json.dumps(info, ensure_ascii=False) + "\n")


# getmemberinfo("https://www.block123.com/zh-hans/nav/997629487911.htm")
dul = {}
with open("detailinfo.txt", "r", encoding='utf-8') as f:
    for line in f.readlines():
        line = line.strip()
        data = json.loads(line)
        for url in data["团队成员"]:
            with open("memberlist.txt", "a+", encoding='utf-8') as f:
                if url not in dul:
                    dul[url] = 1
                    f.write(url + "\n")
