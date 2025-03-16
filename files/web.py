# flask

# demo
# 网站->服务器->域名


from flask import Flask

app = Flask(__name__)

@app.route("/")
def Hello_world():
    return "这是我的第一个python网页"

if __name__ == "__main__":
        app.run(debug = True)