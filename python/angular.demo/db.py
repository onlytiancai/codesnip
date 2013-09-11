import sqlite3

cx = sqlite3.connect("./houses.db", check_same_thread=False)


def add_house(ip, name, text, lastmodified, token):
    cu = cx.cursor()
    cu.execute("insert into houses(ip, name, text, lastmodified, token) values(?, ?, ?, ?, ?)",
               (ip, name, text, lastmodified, token))
    cu.execute("insert into history(ip, name, text, lastmodified) values(?, ?, ?, ?)",
               (ip, name, text, lastmodified))
    cx.commit()
    cu.close()

def get_all_houses():
    cu = cx.cursor()
    rows = cu.execute('SELECT ip, name, text, lastmodified FROM houses ORDER BY lastmodified desc')
    return [dict(ip=row[0], name=row[1], text=row[2], lastmodified=row[3])
            for row in rows]
    cu.close()


def modify_house(ip, name, text, lastmodified, token):
    cu = cx.cursor()
    cu.execute("update houses set ip=?, text=?, lastmodified=? where name=?",
               (ip, text, lastmodified, name))
    cu.execute("insert into history(ip, name, text, lastmodified) values(?, ?, ?, ?)",
               (ip, name, text, lastmodified))
    cx.commit()
    cu.close()


def get_history(name):
    cu = cx.cursor()
    rows = cu.execute('SELECT ip, name, text, lastmodified FROM history where name = ? ORDER BY lastmodified desc',
                      (name, ))
    return [dict(ip=row[0], name=row[1], text=row[2], lastmodified=row[3])
            for row in rows]
    cu.close()

if __name__ == '__main__':
    add_house(ip='1.1.1.1', name='wawa', text='hello world', lastmodified='2013-08-07', token='111')
    modify_house(ip='1.1.1.1', name='wawa', text='goodbyd world', lastmodified='2013-08-09', token='111')
    print 'all houses', get_all_houses()
    print 'wawa history', get_history('wawa')
