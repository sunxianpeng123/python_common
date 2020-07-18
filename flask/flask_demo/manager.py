from flask_migrate import MigrateCommand
from flask_script import Manager

from App import create_app
import os

env = os.environ.get('FLASK_ENV','develop')
app = create_app(env)
manager = Manager(app=app)
manager.add_command('db', MigrateCommand)

if __name__ == '__main__':
    # python manager.py runserver
    # flask,
    # pip install flask-blueprint,控制路由,Flask 可以通过Blueprint来组织URL以及处理请求。Flask使用Blueprint让应用实现模块化
    #           在使用flask进行一个项目编写的时候，可能会有许多个模块，如一个普通的互联网sass云办公应用，会有用户管理、部门管理、账号管理等模块，
    #           如果把所有的这些模块都放在一个views.py文件之中，那么最后views.py文件必然臃肿不堪，并且极难维护，因此flask中便有了blueprint的概念，
    #           可以分别定义模块的视图、模板、视图等等，我们可以使用blueprint进行不同模块的编写，不同模块之间有着不同的静态文件、模板文件、view文件，十分方便代码的维护和管理，
    # pip install flask-sqlalchemy，orm操作数据层,
    #           flask中一般使用flask-sqlalchemy来操作数据库，使用起来比较简单，易于操作。
    # pip install flask_script,
    #           flask_script的作用是可以通过命令行的形式来操作flask例如通过一个命令跑一个开发版本的服务器，设置数据库，定时任务等.安装flask_script比较简单
    # pip install flask-migrate
    #           在开发程序的过程中，你会发现有时需要修改数据库模型，而且修改之后还需要更新数据库。
    #           仅当数据库表不存在时，Flask-SQLAlchemy 才会根据模型进行创建。因此，更新表的唯一方式就是先删除旧表，
    #           不过这样做会丢失数据库中的所有数据。更加好的表示方法就是利用数据库迁移框架
    #           python manager.py
    #           python manager.py db init
    #           python manager.py db migrate
    #           python manager.py db upgrade
    manager.run()
