import re
import os

import matplotlib
matplotlib.use('Agg')  # 适用于无头服务器
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # 定义匹配模式
    pattern = re.compile(r'Pseudo Target MSELoss:\s*([0-9.]+)')
    pattern1 = re.compile(r'loss_threshold:\s*([0-9.]+)')
    # 设定日志文件路径
    parent_file = 'log/our/cifar10'
    log_files = []
    # 遍历log_file这个文件夹
    for root, dirs, files in os.walk(parent_file):
        for file in files:
            sub_file = os.path.join(root, file)
            log_files.append(sub_file)
    
    plot_data = []
    mse_losses = []
    loss_thresholdss = []
    ll = [3, 5, 7, 9, 11] # 1.5 1.65 1.7 1.8 1.9
    # 打开并读取日志文件
    
    for indice in ll:

        with open(log_files[indice], 'r') as file:
            for line in file:
                match = pattern.search(line)
                if match:
                    mse_loss = match.group(1)
                    mse_losses.append(float(mse_loss))
            plot_data.append(mse_losses[-50:])
    print(plot_data)

    for idx, mse_losses in enumerate(plot_data):
        plt.plot(mse_losses, label=f'indices={ll[idx]}')
    # 添加标题和标签
    plt.title('MSE loss pic')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')

    # 显示图例
    plt.legend()
    #
    plt.savefig('output_plot.png')
    plt.show()