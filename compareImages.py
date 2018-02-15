from flask import Flask
from flask import request
from flask_cors import CORS, cross_origin

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route("/api/compare", methods=['POST'])
@cross_origin()
def hello_world():
    # array of 2 images in base 64 format
    imgArray = request.data
    result = 100
    return str(result)


def compareImages():
    data = hello_world()
    return str(data)

if __name__ == '__main__':
    app.run()
