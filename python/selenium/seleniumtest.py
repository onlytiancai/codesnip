from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import os
import time
import csv

#引入chromedriver.exe
chromedriver = "C:/Program Files (x86)/Google/Chrome/Application/chromedriver.exe"
os.environ["webdriver.chrome.driver"] = chromedriver

# 抓取计数及总量
count = 0
total = 10

urls = set()

def get_goods(writer, browser):
    global count, total
    goods = browser.find_elements_by_class_name('gl-item')
    
    for good in goods:
        
        url = good.find_element_by_tag_name('a').get_attribute('href')

        # 跳过不符合要求的商品
        if url.find('item.jd.com') == -1:
            continue

        # 跳过已经抓取过的商品
        if url in urls:
            continue

        count = count + 1
        urls.add(url)
        
        name = good.find_element_by_css_selector('.p-name em').text
        price = good.find_element_by_css_selector('.p-price i').text
        commit = good.find_element_by_css_selector('.p-commit a').text
                
        if count > total:
            print('抓取完成')
            break

        writer.writerow([count, name, url, price, commit])
        
        msg = '''
        序号：%s
        商品：%s
        链接：%s
        价钱：%s
        评论: %s        
        ''' % (count, name, url, price, commit)
        
        print(msg, end='\n\n')

        # 翻页抓取下一页
        button = browser.find_element_by_partial_link_text('下一页')
        button.click()
        time.sleep(1)
        get_goods(writer, browser)
        

def spider(url, keyword):
    browser = webdriver.Chrome(chromedriver)
    browser.get(url)
    browser.implicitly_wait(3)
    
    try: 
        input = browser.find_element_by_id('key')
        input.send_keys(keyword)
        input.send_keys(Keys.ENTER)

        with open('output.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['序号', '商品', '链接', '价钱', '评论'])
            get_goods(writer, browser)
        
    finally:
        browser.close()

if __name__ == '__main__':
    spider('https://www.jd.com', keyword='python')
