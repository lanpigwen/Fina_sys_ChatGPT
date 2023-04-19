# 导入所需的库
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from factor_analyzer import FactorAnalyzer

# 定义Fina888函数
def Fina888_analysis(df):
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
Fina888_analysis(df) # 运行Fina888_analysis函数并输出结果