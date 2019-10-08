from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import pickle
import requests
from requests import Timeout
from lxml import etree
from lxml import etree
from lxml import html
import os
import time
import sys
import csv
import pandas as pd
from cathay.config import ApplicationConfig

class Reuters_Crawler():
    def __init__(self):
        chrome_options = webdriver.ChromeOptions()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--disable-dev-shm-usage')
        
        self._browser=webdriver.Chrome(chrome_options=chrome_options)
        self._downloader = Downloader()
        self._header = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36',
        }
        
    def crawler(self, url, url_output, data_output):
        self._browser.get(url)
        self._rolling()
        
        url = []
        for i in self._browser.find_elements_by_xpath("//div[@class='search-result-indiv']/div/h3/a"):
            url.append(i.get_attribute('href'))
        self._browser.quit()
        
        with open(f"{ApplicationConfig.get_reuters_path()}"+url_output, 'wb') as f:
            pickle.dump(url, f)
            
        result = []
        for i in url:
            print(i)
            data = self._crawler(i)
            if data['title'] != []:
                result.append(data)
            print(data['title'])
            
        data = pd.DataFrame({'date':[x['date'] for x in result], 'title':[x['title'] for x in result], 'text':[x['text'] for x in result]})    
        with open(f"{ApplicationConfig.get_reuters_path()}"+data_output, 'wb') as f:
            pickle.dump(data,f)
        
    
    def _rolling(self):
        while True:
            try:
                self._browser.find_element_by_xpath("//div[@class='search-result-more-txt search-result-no-more']")
                break
            except:
                try:
                    self._browser.find_element_by_xpath("//div[@class='search-result-more-txt']").click()
                except:
                    self._rolling()
                    
   
    def _crawler(self, url):
        content = self._downloader.download(url, retry_count=2, headers=self._header)
        content = str(content)
        html = etree.HTML(content)
        date = html.xpath('//div[@class="ArticleHeader_date"]/text()')
        title =  html.xpath('//h1[@class="ArticleHeader_headline"]/text()')
        text = html.xpath('//div[@class="StandardArticleBody_body"]/p/text()')
        text = ''.join(text)
        data = {'date':date[0], 'title':title[0], 'text':text}
        return data

    
class Downloader(object):
    def __init__(self):
        self.content_cache = {}
        self.request_session = requests.session()
        self.request_session.proxies

    def download(self, url, retry_count, headers, proxies=None, data=None):
        '''
        :param url: 准备下载的 URL 链接
        :param retry_count: 如果 url 下载失败重试次数
        :param headers: http header={'X':'x', 'X':'x'}
        :param proxies: 代理设置 proxies={"https": "http://12.112.122.12:3212"}
        :param data: 需要 urlencode(post_data) 的 POST 数据
        :return: 网页内容或者 None
        '''

#         proxies={"https": "http://5.79.113.168"}
        
        # 如果缓存里面有，直接返回
        if url in self.content_cache:
            # 　print("缓存里面有，直接返回")
            return self.content_cache.get(url)

        if headers:
            self.request_session.headers.update(headers)
        try:
            #print("Downloader downloading the page...")
            if data:
                content = self.request_session.post(url, data, proxies=proxies).content
            else:
                content = self.request_session.get(url, proxies=proxies).content
            content = content.decode('utf8', 'ignore')
            content = str(content)
            # print("url加入缓存-> url=" + url)
            self.content_cache[url] = content
        except (ConnectionError, Timeout) as e:
            print('Downloader download ConnectionError or Timeout:' + str(e))
            content = None
            if retry_count > 0:
                self.download(url, retry_count - 1, headers, proxies, data)
        except Exception as e:
            print('Downloader download Exception:' + str(e))
            content = None
     
        return content
