
from flask_script import Manager


manager = Manager(app)

@app.route('/')
def hello_world():
    return 'Hello World!'


if __name__ == '__main__':
    app.run()
