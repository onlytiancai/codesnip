import json
import sys
from urllib.parse import quote
import requests
from termcolor import colored

API_URL = "http://fanyi.youdao.com/openapi.do?keyfrom=JIANGDi&key=891853312&type=data&doctype=json&version=1.1&q="


class Fanyi:
    def __init__(self, input_str):
        self.input_str = input_str
        self.query_for_hash()

    def query_for_hash(self):
        query_url = API_URL + quote(self.input_str.replace(" ", "+"))
        result_json = requests.get(query_url).text
        self.result_hash = json.loads(result_json)

    def translations(self):
        translations = self.result_hash.get("translation")
        lines = ["  " + colored(translation, "green") for translation in translations]
        lines.append("")
        return lines

    def word_and_phonetic(self):
        line = " " + self.input_str
        phonetic = self.result_hash.get("basic", {}).get("phonetic")
        if phonetic:
            line += f" [ {colored(phonetic, 'magenta')} ]"
        return [line, ""]

    def dict_explains(self):
        dict_explains = self.result_hash.get("basic", {}).get("explains")
        if dict_explains:
            lines = [" - " + colored(explain, "green") for explain in dict_explains]
            lines.append("")
            return lines
        return []

    def web_results(self):
        web_results = self.result_hash.get("web")
        if not web_results:
            return []
        lines = []
        for i, web_result in enumerate(web_results):
            web_result_key = web_result["key"].replace(self.input_str, colored(self.input_str, "yellow"))
            web_result_value = ", ".join(web_result["value"]).replace(" ", "").replace(",", ", ")
            lines.append(f" {i+1}. {web_result_key}")
            lines.append(f"    {colored(web_result_value, 'cyan')}")
        lines.append("")
        return lines

    def yd_result(self):
        return self.dict_explains() or self.translations()

    def result(self):
        output = []
        output.extend(self.word_and_phonetic())
        output.extend(self.yd_result())
        output.extend(self.web_results())
        return output


if len(sys.argv) == 1:
    try:
        while True:
            input_str = input("> ").strip()
            if not input_str:
                continue
            results = Fanyi(input_str).result()
            print("\n".join(results))
    except KeyboardInterrupt:
        print("bye~")
elif "-h" in sys.argv or "--help" in sys.argv:
    print(
        """
fy: Translate tools in the command line
  $ fy word
  $ fy world peace
  $ fy chinglish
  $ fy
  > enter the loop mode, ctrl+c to exit"""
    )
else:
    input_str = " ".join(sys.argv[1:])
    results = Fanyi(input_str).result()
    print("\n".join(results))
