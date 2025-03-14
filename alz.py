import joblib
import os
from sqlalchemy import create_engine
import pandas as pd
import numpy as np

model_path = os.path.join('static', 'random_forest_model.pkl')
clf = joblib.load(model_path)

db_url = 'mysql+pymysql://nanoka:12345678n@localhost:3308/manage'
engine = create_engine(db_url)

query = '''
SELECT
	COUNT(*) AS num_length,
	COALESCE (
		SUM( CASE WHEN STATUS NOT IN ( 200, 302 ) THEN 1 WHEN STATUS != 302 THEN 1 ELSE 0 END ) / NULLIF( COUNT(*), 0 ),
		0 
	) AS rate_failure,
	COALESCE (
		SUM( CASE WHEN path = '/login' AND method = 'POST' AND STATUS != 200 THEN 1 ELSE 0 END ) / NULLIF( COUNT( CASE WHEN path = '/login' AND method = 'POST' THEN 1 ELSE NULL END ), 0 ),
		0 
	) AS rate_failure_login_post,
	COUNT( DISTINCT PORT ) AS num_port,
	COALESCE ( SUM( IFNULL( LENGTH( body ), 0 )), 0 ) AS total_body_length,
	COALESCE ( SUM( IFNULL( LENGTH( REGEXP_REPLACE ( body, '[a-zA-Z0-9 ]', '' )), 0 )), 0 ) AS special_characters_length 
FROM
	web 
WHERE
	time >= NOW() - INTERVAL 10 MINUTE;
'''
data = pd.read_sql(query, engine)
print(data.T)

data_input = np.array([data[col][0] for col in data.columns])

output = clf.predict(data_input.reshape(1, -1))
label = {
  0: '正常访问',
  1: '拒绝服务攻击',
  2: '扫描攻击',
  3: '注入攻击',
  4: '密码爆破或密码播撒'
}

print('\n现在的访问状态是: "{}"'.format(label[output[0]]))

