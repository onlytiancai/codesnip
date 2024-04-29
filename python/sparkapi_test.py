import os
import config
os.environ['SPARK_APP_ID'] = config.SPARK_APP_ID
os.environ['SPARK_API_SECRET'] = config.SPARK_API_SECRET
os.environ['SPARK_API_KEY'] = config.SPARK_API_KEY
os.environ['SPARK_API_MODEL'] = "v2.0"
os.environ['SPARK_CHAT_MAX_TOKENS'] = "4096"
os.environ['SPARK_CHAT_TEMPERATURE'] = "0.5"
os.environ['SPARK_CHAT_TOP_K'] = "4"


from sparkapi.core.api import SparkAPI
from sparkapi.core.config import SparkConfig
cfg = SparkConfig().model_dump()
api = SparkAPI(**cfg)


print(''.join(api.get_completion('讲个关于程序员的段子')))
