from flask_script import Manager

from App import create_app

app = create_app()
manager = Manager(app=app)


if __name__ == '__main__':
    # python manager.py runserver
    # flask,
    # flask-blueprint,控制路由
    # flask-sqlAlchemy，orm操作数据层
    manager.run()
