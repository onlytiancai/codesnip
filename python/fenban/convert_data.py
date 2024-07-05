import csv
reader = csv.reader(open('fakedata.csv', encoding='utf-8'))
writer = csv.writer(open('fakedata_v2.csv', 'w', encoding='utf-8'))
for row in reader:
    writer.writerow([row[0],row[1],row[3],row[4],row[5],row[2]])
