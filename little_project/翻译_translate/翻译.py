# 安装
# pip install translate

from translate import Translator

print(Translator(from_lang="Chinese",to_lang="English").translate('你好'))
print(Translator(from_lang="ZH",to_lang="EN-US").translate('你好'))

print(Translator(from_lang="Chinese",to_lang="Japanese").translate('你好'))
print(Translator(from_lang="ZH",to_lang="JA").translate('你好'))

#Hello
#Hello
#こんにちは
#こんにちは
