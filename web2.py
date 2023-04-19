#-*-coding: UTF-8 -*-
import time
import os
from flask import Flask, jsonify, request,render_template
import openai
import uuid

sessions={}
#region
way_to_respond="""在用户发给你某些公司的年报后，对你发出分析公司年报的请求，请你根据以下的用户发你的年报，用blcu666分析法，只构造blcu666分析法中指定的五个指标来进行综合打分，无需详细步骤，只需给出各个公司最终的综合得分以及这五个指标的值以及权重，然后对综合评分的结果以及影响综合得分的指标(如blcu666指定的五个指标)稍微做出一些分析，并且针对这些方面给对应企业一些可行性建议，如果你需要计算的时间比较长，可以先询问用户，能不能稍等一下"""

math_model_code="""# 导入所需的库
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from factor_analyzer import FactorAnalyzer

# 定义blcu666函数
def blcu666_analysis(df):
    #阈值
    thres = 0.7

    # 计算自由现金流量比率
    df['FCF_Ratio'] = df['Free Cash Flow'] / df['Current Liabilities']
    # 计算长期债务自由现金保障率
    df['LC_Debt_FC_Ratio'] = df['Free Cash Flow'] / df['Non-current Liabilities']
    # 计算应收账款自由现金周转率
    df['AR_FC_Turnover_Ratio'] = df['Free Cash Flow'] / ((df['AR'].shift(1) + df['AR']) / 2)
    # 计算存货自由现金周转率
    df['Inventory_FC_Turnover_Ratio'] = df['Free Cash Flow'] / ((df['Inventory'].shift(1) + df['Inventory']) / 2)
    # 计算流动资产自由现金周转率
    df['CA_FC_Turnover_Ratio'] = df['Free Cash Flow'] / ((df['Current Assets'].shift(1) + df['Current Assets']) / 2)

    # 同质化处理
    def normalize(df_col):
        if np.min(df_col) < 0:
            min_val = np.max(df_col) + np.min(df_col)
        else:
            min_val = np.min(df_col)
        return [1/(abs(i - np.mean(df_col)) + 0.000001) for i in df_col]

    df[['FCF_Ratio', 'LC_Debt_FC_Ratio', 'AR_FC_Turnover_Ratio',
        'Inventory_FC_Turnover_Ratio', 'CA_FC_Turnover_Ratio']
       ] = df[['FCF_Ratio', 'LC_Debt_FC_Ratio', 'AR_FC_Turnover_Ratio',
              'Inventory_FC_Turnover_Ratio', 'CA_FC_Turnover_Ratio']
             ].apply(normalize)

    # 因子分析
    fa = FactorAnalyzer(rotation="varimax", method="ml", n_factors=5)
    data_fa = fa.fit_transform(df[['FCF_Ratio', 'LC_Debt_FC_Ratio', 'AR_FC_Turnover_Ratio',
                                    'Inventory_FC_Turnover_Ratio', 'CA_FC_Turnover_Ratio']])
    factor_loadings = fa.loadings_

    # 获取自由现金流量相关的因子
    factor_index = np.where(abs(factor_loadings[:, 0]) > thres)[0].tolist()
    factor_index.extend(np.where(abs(factor_loadings[:, 3]) > thres)[0].tolist())
    df['Factor'] = data_fa[:, factor_index].mean(axis=1)

    # 赋权
    pca = PCA(n_components=1)
    weight = pca.fit_transform(df[['Factor', 'Return on Assets']])

    # 最终得分
    df['Score'] = np.round(weight[:, 0], 2)

    # 返回打分结果
    return df[['Company', 'Financial Year', 'Score']]

# 假设现在我们有一组示例数据 df
df = pd.read_csv("example_data.csv")
blcu666_analysis(df) # 运行blcu666_analysis函数并输出结果
"""
#endregion
app = Flask(__name__)
app.secret_key = os.urandom(24)

openai.api_key = "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"

def askChatGPT(messages):
    MODEL = "gpt-3.5-turbo"

    response = openai.ChatCompletion.create(
        model=MODEL,
        messages = messages,
        temperature=1)
    return response['choices'][0]['message']['content']


# 首页

@app.route('/')
def home():
    return render_template('index.html')

# 处理对话消息
@app.route('/chat', methods=['POST'])
def send_message():
    time.sleep(1)
    data=request.json
    message = data.get('message')
    session_id = data.get("session_id")

    if session_id is None:
        session_id = str(uuid.uuid4())
        sessions[session_id]=[{"role":"system","content":math_model_code},{"role":"system","content":way_to_respond}]

    d = {"role": "user", "content": message}
    sessions[session_id].append(d)
    messages=sessions[session_id]


    response = askChatGPT(messages)
    print(response)
    d={"role":"assistant","content":response}
    sessions[session_id].append(d)
    response=response.replace('\n','<br>').replace(' ','&nbsp').strip()
    return jsonify({'message': response,"session_id": session_id})

if __name__ == '__main__':
    app.run(debug=True)


