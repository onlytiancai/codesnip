# -*- coding: utf-8 -*-
import db

for line in open('./log.log'):
    if line.find('edit houses') != -1:
        pos1 = line.find('{')
        pos2 = line.find('}')
        data = line[pos1: pos2+1]
        print data
        data = eval(data)
        try:
            db.modify_house(data['ip'], data['name'], data['text'], data['lastmodified'], '')
        except:
            print 'error:', data
        print data
