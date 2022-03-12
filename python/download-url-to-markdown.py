import sys
from pathlib import Path
import requests
from html2text import HTML2Text
from bs4 import BeautifulSoup


if len(sys.argv) != 3:
    print('Usage: %s <url> <file>' % sys.argv[0])
    sys.exit()

cmd, url, save_file = sys.argv

save_file = Path(save_file)
if save_file.exists():
    print('file exist: %s' % save_file)
    sys.exit()

headers = { 'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:22.0) Gecko/20100101 Firefox/22.0'}
r = requests.get(url, headers=headers)
if r.status_code != 200:
    print('got %s response' % r.status_code)
    sys.exit()

html = r.text

img_dir = save_file.parent.joinpath(save_file.stem + '_images')
if not img_dir.exists():
    img_dir.mkdir()

soup = BeautifulSoup(html, "html.parser")
images = soup.findAll('img')
image_map = {}
for i, img in enumerate(images):
    img_src = img.get("src")
    if img_src.startswith('http://') or img_src.startswith('https://'):
        content_type_map = { 'image/jpeg': '.jpg', 'image/png': '.png', 'image/gif': '.gif', }
        r = requests.get(img_src, stream=True, headers=headers)
        img_suffix = content_type_map.get(r.headers['content-type'], '.jpg')
        img_file = img_dir.joinpath(str(i) + img_suffix)
        image_map[img_src] = img_file
        print('downlod image %s: %s got %s' % (i, img_src, r.status_code))
        if r.status_code == 200:
            with open(img_file, 'wb') as f:
                for chunk in r.iter_content(1024):
                    f.write(chunk)

h = h = HTML2Text() 
md = h.handle(html)
for k, v in image_map.items():
    md = md.replace(k, save_file.stem + '_images/' + v.name)

with open(save_file, 'w') as f:
    f.write(url + '\n')
    f.write(md)

print('download %s ok, %s characters have been written..' % (save_file, len(md)))
