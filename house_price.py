import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# 讀取資料
url = 'https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/sklearn/datasets/data/boston_house_prices.csv'
df = pd.read_csv(url, skiprows=1)  # 跳過第一行註解

# 確認房價欄位名稱
print(df.columns)

# 計算房價統計值
df['MEDV'] = pd.to_numeric(df['MEDV'], errors='coerce')  # 確保資料是數值型
highest_price = df['MEDV'].max()
lowest_price = df['MEDV'].min()
average_price = df['MEDV'].mean()
median_price = df['MEDV'].median()

# 輸出結果
print(f"最高房價: {highest_price}")
print(f"最低房價: {lowest_price}")
print(f"平均房價: {average_price}")
print(f"中位數房價: {median_price}")

# 繪製房價分佈圖
# 讀取資料
url = 'https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/sklearn/datasets/data/boston_house_prices.csv'
df = pd.read_csv(url, skiprows=1)  # 跳過第一行註解

# 確保房價資料是數值型
df['MEDV'] = pd.to_numeric(df['MEDV'], errors='coerce')

# 繪製直方圖
url = 'https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/sklearn/datasets/data/boston_house_prices.csv'
df = pd.read_csv(url, skiprows=1)  # 跳過第一行註解

# 確保房價資料是數值型
df['MEDV'] = pd.to_numeric(df['MEDV'], errors='coerce')

# 設定區間與繪圖數據
bin_edges = range(0, int(df['MEDV'].max()) + 10, 10)  # 區間 [0, 10), [10, 20), ...
hist, edges = np.histogram(df['MEDV'], bins=bin_edges)

# 設定條形圖寬度和位置
bar_width = 8  # 長條寬度，留間距
bar_positions = edges[:-1] + 1  # 每個長條的中心位置稍微偏移，避免重疊

# 繪製條形圖
plt.figure(figsize=(10, 6))
plt.bar(bar_positions, hist, width=bar_width, color='skyblue', edgecolor='black', align='center')

# 設定 X 軸區間標籤
x_labels = [f"{edges[i]}-{edges[i+1]}" for i in range(len(edges)-1)]
plt.xticks(ticks=bar_positions, labels=x_labels, rotation=45, fontsize=10)

# 添加標題和標籤
plt.title('Distribution of Houce Price', fontsize=16)
plt.xlabel('House Price Range(thousand dollars)', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 顯示圖表
plt.tight_layout()
plt.show()


# 讀取資料
url = 'https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/sklearn/datasets/data/boston_house_prices.csv'
df = pd.read_csv(url, skiprows=1)  # 跳過第一行註解

# 確保 RM 和 MEDV 資料是數值型
df['RM'] = pd.to_numeric(df['RM'], errors='coerce')
df['MEDV'] = pd.to_numeric(df['MEDV'], errors='coerce')

# 將 RM 四捨五入到個位數
df['RM_rounded'] = df['RM'].round()

# 使用 groupby 計算每個 RM 值的平均房價
grouped = df.groupby('RM_rounded')['MEDV'].mean().reset_index()
grouped.columns = ['RM', 'Average_Price']

# 繪製直方圖
plt.figure(figsize=(10, 6))
bar_width = 0.8  # 設定長條寬度
plt.bar(grouped['RM'], grouped['Average_Price'], width=bar_width, color='skyblue', edgecolor='black')

# 設定 X 軸標籤
x_labels = [f"{int(rm)}" for rm in grouped['RM']]
plt.xticks(ticks=grouped['RM'], labels=x_labels, fontsize=10)

# 添加標題和標籤
plt.title('Distribution of Boston Housing Prices Group by RM', fontsize=16)
plt.xlabel('RM', fontsize=14)
plt.ylabel('MEDV', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 顯示圖表
plt.tight_layout()
plt.show()

# 線性回歸模型預測房價
x = df.drop('MEDV', axis=1)  # 特徵
y = df['MEDV']  # 標籤
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 建立線性回歸模型
model = LinearRegression()
model.fit(x_train, y_train)

# 評估表現
y_pred = model.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
asme = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)

print("均方誤差: ", mse)
print("平均均方誤差: ", asme)
print("R2 分數: ", r2)