from os import environ, path
from dotenv import load_dotenv

basedir = path.abspath(path.dirname(__file__))
load_dotenv(path.join(basedir, '.env'))

conn = dict(host='gateway01.eu-central-1.prod.aws.tidbcloud.com',
                 user=environ.get('DB_USER'),
                 password=environ.get('DB_PASS'),
                 port=4000,
                 charset='utf8mb4',)
