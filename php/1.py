import sys
import html
import requests
print(sys.argv)
response = requests.get(sys.argv[1])
print(response.status_code, html.unescape(response.text))
