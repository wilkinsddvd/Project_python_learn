import openai
openai.api_key=OPENAI_API_KEY
openai.proxy=OPENAI_PROXY

completion = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "user", "content": "一亩地租金1000元，那么3平方米地的租金应该是多少呢？"}
  ]
)