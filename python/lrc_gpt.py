import glob
import os
import json
import requests
import config


url = config.gpturl
headers = {'Content-Type':'application/json', 'Authorization': config.gptkey}


def traverse_dir(path):
    files = glob.glob(os.path.join(path, "*"))
    total = 0
    for file in files:
        if os.path.isdir(file):
            traverse_dir(file)
        else:
            if file.endswith('.lrc'):
                if file.endswith('cn.lrc'):
                    continue
                dir = os.path.dirname(file)
                file_name = os.path.basename(file)
                stem, suffix = os.path.splitext(file_name)
                out_file = os.path.join(dir, stem+'_cn'+suffix )
                # print('process', total, file)
                if not os.path.exists(out_file):
                    content = open(file, encoding='utf-8').read()                    
                    content = '请把如下```包含起来的内容翻译成中文，格式不要改变，如下只需要翻译的内容。\n```%s```' % content
                    data = json.dumps({'model':'gpt-3.5-turbo', "messages": [{"role": "user", "content": content}]})
                    r = requests.post(url, data, headers=headers, timeout=(10,120))
                    r = r.json()
                    try:
                        r_text  = r['choices'][0]['message']['content'].strip()
                        open(out_file,'w', encoding='utf8').write(r_text)
                        print(out_file, 'write sucess', len(content), len(r_text))
                    except:
                        print(r)
                        raise
                total += 1

dir_path = "D:\\BaiduNetdiskDownload"
print('待遍历的目录为：', dir_path)
print('遍历结果为：')
# traverse_dir(dir_path)

dir_path = "D:\\BaiduNetdiskDownload\\NCE1-英音-(MP3+LRC)"
files = glob.glob(os.path.join(dir_path, "*"))
names = set()
for file in files:    
    print(file)
    dir = os.path.dirname(file)
    file_name = os.path.basename(file)
    stem, suffix = os.path.splitext(file_name)
    if suffix not in ['.lrc','.mp3']:
        continue
    if stem.endswith('_cn'):
        stem = stem[:-3]
    names.add(stem)
names = sorted(names)
print(names)
print(len(names))

result = []
for name in names:
    mp3 = name + '.mp3'
    lrc_en = name + '.lrc'
    lrc_cn = name + '_cn.lrc'
    for f in [mp3, lrc_cn, lrc_en]:
        if not os.path.exists(os.path.join(dir_path,f)):
            print(f)
            raise Exception("file not exists")
    result.append({
        'mp3': mp3,
        'lrc_en': lrc_en,
        'lrc_cn': lrc_cn,
    })

open('out_json.json','w').write(json.dumps(result, indent=4))