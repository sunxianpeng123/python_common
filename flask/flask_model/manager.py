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
    #1、 pip install flask-blueprint,控制路由,Flask 可以通过Blueprint来组织URL以及处理请求。Flask使用Blueprint让应用实现模块化
    #           在使用flask进行一个项目编写的时候，可能会有许多个模块，如一个普通的互联网sass云办公应用，会有用户管理、部门管理、账号管理等模块，
    #           如果把所有的这些模块都放在一个views.py文件之中，那么最后views.py文件必然臃肿不堪，并且极难维护，因此flask中便有了blueprint的概念，
    #           可以分别定义模块的视图、模板、视图等等，我们可以使用blueprint进行不同模块的编写，不同模块之间有着不同的静态文件、模板文件、view文件，十分方便代码的维护和管理，
    # 2、pip install flask-sqlalchemy，orm操作数据层,
    #           flask中一般使用flask-sqlalchemy来操作数据库，使用起来比较简单，易于操作。
    # 3、pip install flask_script,
    #           flask_script的作用是可以通过命令行的形式来操作flask例如通过一个命令跑一个开发版本的服务器，设置数据库，定时任务等.安装flask_script比较简单
    # 4、pip install flask-migrate
    #           Flask-Migrate 插件提供了和 Django 自带的 migrate 类似的功能。
    #           python manager.py
    #           python manager.py db init
    #           python manager.py db migrate
    #           python manager.py db upgrade
    # 5、odoo
    # 比Django还重的web框架，可以快速生成网站
    # 6、pip install flask-session
    #           允许设置session到指定存储的空间中,
    # 7、pip install flask-bootstrap(还有一个类似的插件 bootstrap-flask)
    #                   Bootstrap-Flask是一个简化在Flask项目中集成前端开源框架Bootstrap过程的Flask扩展。使用Bootstrap可以快速的创建简洁、美观又功能全面的页面，
    #           而Bootstrap-Flask让这一过程更加简单和高效。尤其重要的是，Bootstrap-Flask支持最新版本的Bootstrap 4版本。
    #                   bootstrap-flask和Flask-Bootstrap的区别
    #                   简单来说，Bootstrap-Flask的出现是为了替代不够灵活且缺乏维护的Flask-Bootstrap。它的主要设计参考了Flask-Bootstrap,
    #           其中渲染表单和分页部件的宏基于Flask-Bootstrap中的相关代码修改实现。和Flask-Bootstrap相比，前者有下面这些优点：
    #                   去掉了内置的基模板，换来更大的灵活性，
    #                   提供了资源引用代码生成函数
    #                   支持Bootstrap 4
    #                   标准化的Jinja2语法提供了更多方便的宏，比如简单的分页导航部件、导航链接等
    #                   宏的功能更加丰富，比如分页导航支持传入URL片段
    #                   统一的宏命名，即“render_*”，更符合直觉
    # pip install flask-debugtoolbar

    manager.run()

