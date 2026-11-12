import os
import resend
from resend.exceptions import ResendError
from dotenv import load_dotenv

load_dotenv()

resend.api_key = os.environ["RESEND_API_KEY"]

params: resend.Emails.SendParams = {
    "from": "Acme <hao@ihuhao.com>",
    "to": ["wawasoft@qq.com"],
    "subject": "hello world",
    "html": "<strong>it works!</strong>",
}

try:
    email = resend.Emails.send(params)
    print(email)
except ResendError as error:
    print(error)
