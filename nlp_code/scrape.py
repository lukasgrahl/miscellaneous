from itertools import chain
import requests
from bs4 import BeautifulSoup

root = "https://en.wikipedia.org"
url = root + "/wiki/Main_Page"
req = requests.get(url)
page = BeautifulSoup(req.content, 'html.parser')
dyks = page.find_all('div', {"id": "mp-dyk"})
dyks = [
            [
                [
                    root + a["href"] for a in li.find_all("a")                      # all links in dyk
                ] for li in dyk.find_all("li")                                      # all dyk items
             ] for dyk in dyks                                                      # dyk box
]
print(dyks)

# url = "https://www.thomas-renault.com/side/ex1.html"
# req = requests.get(url)
#
# soup = BeautifulSoup(req.content, 'html.parser')
# # find the titel, website title usually just one
# # print(soup.find('title'))
#
# # find all links
# # for element in soup.find_all("a"):
# body = soup.body
# print(body.find("p").text)
#
# output = []
# output.append([i.text for i in body.find("p")])
# output.append([i.text for i in body.find("h3")])
# output = [*chain(*output)]
#
# print(output)
# print(body.
