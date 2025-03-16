# coding=utf-8

from chatterbot import ChatBot

# 第一步，创建chatterbot

chatbot = ChatBot(

    'Ron Obvious',

    trainer='chatterbot.trainers.ChatterBotCorpusTrainer'

)

# 第二步，训练语料

chatbot.train("chatterbot.corpus.english")

# 第三步，输入对话得到答案

while True:
    q = raw_input()

    print(chatbot.get_response(q))