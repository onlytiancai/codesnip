import sqlite3
from flask import Flask
from flask import render_template
from flask import g

app = Flask(__name__)

DATABASE = './hn.db'

def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
    db.row_factory = sqlite3.Row
    return db

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

def query_db(query, args=(), one=False):
    cur = get_db().execute(query, args)
    rv = cur.fetchall()
    cur.close()
    return (rv[0] if rv else None) if one else rv

def page_info(total, page, page_size=10):
    pages = total // page_size
    if total % page_size > 0:
        pages = pages + 1
    return dict(pages=pages, page=page, has_prev=page>1, has_next=page<pages,
                page_size=page_size, offset=(page-1)*page_size) 


@app.route('/<int:page>')
@app.route('/')
def index(page=1):
    total = query_db('select count(*) from feeds', (), True)
    pi = page_info(total[0], page, 10)
    rows = query_db('select * from feeds order by id desc limit ? offset ?', (pi['page_size'], pi['offset']))
    return render_template('index.html', rows=rows, page_info=pi)
