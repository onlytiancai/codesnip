import json
import sys
import requests
import config

data = json.dumps({'model':'gpt-3.5-turbo', 
                   "messages": [
                       {"role": "user", "content": sys.argv[1]}
                   ],
                   'stream': True,
                 })
headers = {'Content-Type':'application/json', 
           'Accept': 'text/event-stream',
           'Authorization': 'Bearer '+ config.OPENAI_API_KEY,
           }
r = requests.post(config.openai_api_url, data, stream=True, headers=headers)
for line in r.iter_lines():
    if line:
        line = line.decode('utf-8')[len('data:'):].strip()
        print(line)
        if line != '[DONE]':
            data = json.loads(line)
            if data['choices'][0]['finish_reason'] != 'stop':
                txt = data['choices'][0]['delta']['content']
                print(txt, end='')
        else:
            print('')


