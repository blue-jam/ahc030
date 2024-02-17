import numpy as np
import matplotlib.pyplot as plt

def generate_normal_random(k, v, e):
    # 平均と分散から標準偏差を計算する
    std_deviation = np.sqrt(k * e * (1 - e))

    # 平均と標準偏差を使って正規分布に従う乱数を生成する
    x = np.random.normal((k - v) * e + (1 - e) * v, std_deviation)

    # 最も近い整数に丸める
    rounded_x = np.round(x)

    return int(rounded_x) if rounded_x > 0 else 0


def plot_list(k, t, l):
    # x軸の値を0からlen(l)-1までの整数とします
    x = range(len(l))
    # リストlの要素をy軸にプロットします
    plt.plot(x, l)
    # グラフのタイトルと軸ラベルを設定します
    plt.title(f'List Plot: k={k}, x={t}')
    plt.xlabel('Index')
    plt.ylabel('Value')
    # グラフを表示します
    plt.show()


# two dimensional list to store the result of generate_normal_random_list
result = [[[[0 for i in range(100)] for j in range(21)] for k in range(21)] for e in range(21)]
for e in range(1, 21):
    for k in range(2, 21):
        for v in range(0, 21):
            for t in range(1000):
                x = generate_normal_random(k, v, e / 100)
                result[e][k][v][x] += 1

for e in range(1, 21):
    for k in range(9, 10):
        for x in range(0, 1):
            sum = 0
            for v in range(0, 21):
                sum += result[e][k][v][x]
            if sum <= 0:
                continue
            for v in range(0, 21):
                result[e][k][v][x] = result[e][k][v][x] / sum
            print(f"e={e / 100}, k={k}, x={x}, result={result[e][k][0][x]}")
            # l = []
            # for v in range(0, 21):
            #     l.append(result[k][v][x])
            # plot_list(k, x, l)