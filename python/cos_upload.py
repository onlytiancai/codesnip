from qcloud_cos import CosConfig, CosS3Client
import config

client = CosS3Client(CosConfig(
    Region='ap-nanjing', 
    SecretId=config.qcloud_cos_secret_id, 
    SecretKey=config.qcloud_cos_secret_key, 
    Scheme='https'
))
response = client.list_buckets()
for bucket in response['Buckets']['Bucket']:
    print(bucket['Name'])
