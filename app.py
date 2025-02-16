from flask import Flask, render_template, request, redirect, g
from flask_cors import CORS
import pymysql
import threading
from datetime import datetime
import uuid

app = Flask(__name__)
CORS(app)

# 嵌入式平台下不需要考虑高并发, 更多的是安全性
data_lock = threading.Lock()

arg_kwargs = {
    'host': "localhost",
    'port': 3308,
    'user': 'nanoka',
    'password': "12345678n",
    'database': "manage",
    'charset': 'utf8mb4'
}

db = pymysql.connections.Connection(**arg_kwargs)
cursor = db.cursor()

@app.route('/')
def index():
    return redirect('/login')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        if username == 'abcd' and password == '1234':
            return redirect('/login_success')
        else:
            return '登录失败，请检查用户名和密码是否正确！'        
    else:
        return render_template('login.html')
      
@app.route('/login_success')
def login_success():
    return render_template('login_success.html')
  
@app.before_request
def before_request():
    # 获取线程锁
    data_lock.acquire()
    
    request_id = str(uuid.uuid4())
    g.request_id = request_id
    
    client_ip = request.remote_addr
    client_port = request.host.split(":")[-1]
    request_path = request.path
    request_method = request.method
    request_body = request.get_data(as_text=True)
    
    try:
        if request_body is not None:
            sql = "INSERT INTO web (ip, port, path, method, body, time, request_id) VALUES (%s, %s, %s, %s, %s, %s, %s)"
            cursor.execute(sql, (client_ip, client_port, request_path, request_method, None, datetime.now(), request_id))
        else:
            sql = "INSERT INTO web (ip, port, path, method, time, request_id) VALUES (%s, %s, %s, %s, %s, %s)"
            cursor.execute(sql, (client_ip, client_port, request_path, request_method, datetime.now(), request_id))
        db.commit()
        
    except Exception as e:
        app.logger.error(f"Error in before_request: {e}")
        
@app.after_request
def after_request(response):
    request_id = g.get('request_id', 'N/A')
    status_code = response.status_code
    
    try:
        sql = "UPDATE web SET status = %s WHERE request_id = %s"
        cursor.execute(sql, (status_code, request_id))
        db.commit()
        
    except Exception as e:
        app.logger.error(f"Error in after_request: {e}")
    
    data_lock.release()
    return response
        
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=82, debug=False)