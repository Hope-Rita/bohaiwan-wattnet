import xlsxwriter as xw
import numpy as np

from utils.calculate_metric import calculate_metrics


def xw_toExcel(data, fileName):  # xlsxwriter库储存数据到excel
    workbook = xw.Workbook(fileName)  # 创建工作簿
    worksheet1 = workbook.add_worksheet("sheet1")  # 创建子表
    worksheet1.activate()  # 激活表
    title = ['time', 'RMSE', 'MAE', 'MAPE', 'PCC']  # 设置表头
    worksheet1.write_row('A1', title)  # 从A1单元格开始写入表头
    RMSE = data['RMSE']
    MAE = data['MAE']
    MAPE = data['MAPE']
    PCC = data['PCC']
    i = 2  # 从第二行开始写入数据
    time = len(RMSE)
    for t in range(time):
        insertData = [t+1,RMSE[t], MAE[t], MAPE[t], PCC[t]]
        row = 'A' + str(i)
        worksheet1.write_row(row, insertData)
        i += 1
    workbook.close()  # 关闭表

# truth = np.load(r"D:\A_实验室\土木2组\谭博项目\LSTM结果\24-12\truth.npy")
# pred = np.load(r"D:\A_实验室\土木2组\谭博项目\LSTM结果\24-12\pred.npy")
# data = calculate_metrics(pred, truth, mode='test')
#
# xw_toExcel(data,'lstm-result.xlsx')