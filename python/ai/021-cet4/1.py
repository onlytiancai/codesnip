filename = "CET4luan_1.json";

import json;

file = open( filename,'r',encoding='utf-8' );
for line in file.readlines():
    words = line.strip();
    word_json = json.loads( words );
    print( word_json["headWord"] );
    
file.close()