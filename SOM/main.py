from sys import argv
from io_helper import read_tsp, normalize
import numpy as np
from neuron import generate_network, get_neighborhood,get_route
from distance import select_closest, euclidean_distance,route_distance
from plot import plot_network, plot_route, create_gif
import imageio
import os
def main():
    
    # 选择输入数据文件
    if len(argv) != 2:
        print("输入: python SOM/main.py <filename>.tsp")
        return -1
    # 读取tsp文件
    
    problem = read_tsp(argv[1])

    # problem = read_tsp('assets/qa194.tsp')
    route = som(problem, 100000)
    problem = problem.reindex(route)
    distance = route_distance(problem)
    print("最短路线为：{}".format(distance))

def som(problem, iterations, learning_rate = 0.8):
    '''
    SOM解决TSP问题
    '''
    # 将城市数据归一化处理
    cities = problem.copy()
    cities[['x', 'y']] = normalize(cities[['x', 'y']])

    # 神经元数量设定为城市的8倍
    n = cities.shape[0] * 8
    
    # 建立神经网络
    network = generate_network(n)
    
    print('创建{} 个神经元. 开始进行迭代:'.format(n))

    for i in range(iterations):
        if not i % 100 :
            print('\t> 迭代过程 {}/{}'.format(i, iterations), end = '\r')
        # 随机选择一个城市
        city = cities.sample(1)[['x', 'y']].values
        
        #优胜神经元（距离该城市最近的神经元）
        winner_idx = select_closest(network, city)
        
        # 以该神经元为中心建立高斯分布
        gaussian = get_neighborhood(winner_idx, n // 10, network.shape[0])

        # 更新神经元的权值，使神经元向被选中城市移动
        network += gaussian[:, np.newaxis] * learning_rate * (city - network)

        # 学习率衰减，方差衰减
        learning_rate = learning_rate * 0.99997
        n = n * 0.9997

        # 每迭代1000次时作图
        if not i % 1000:
            plot_network(cities, network, name = 'som_diagrams/{:05d}.png'.format(i))
            pass
        # 判断方差和学习率是否达到阈值
        if n < 1:
            print('方差已经达到阈值，完成执行次数{}'.format(i))
            break
        if learning_rate < 0.001:
            print('学习率已经达到阈值，完成执行次数{}'.format(i))
            break
    else:
        print('完成迭代：{}次'.format(iterations))


    plot_network(cities, network, name = 'som_diagrams/final.png')
    route = get_route(cities, network)
    cities = cities.reindex(route)
    plot_route(cities, route, 'som_diagrams/route.png')

    # 将多个png文件合成gif文件
    path = os.chdir('.\som_diagrams')
    pic_list = os.listdir()    
    create_gif(pic_list,'result.gif', 0.3)
    
    return route
if __name__ == '__main__':
    main()