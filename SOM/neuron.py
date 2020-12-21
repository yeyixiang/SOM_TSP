import numpy as np
from  distance import select_closest
def generate_network(size):
    """
    建立一个size 大小的神经网络
    返回size * 2 维矩阵，其数值在[0, 1]中间
    """
    return np.random.rand(size, 2)

def get_neighborhood(center, radix, domain):
    """
    以优胜神经元为中心，建立高斯分布
    """
    # 高斯分布的方差设置一个下界
    if radix < 1:
        radix = 1
        
    # 计算各个神经元到获胜神经元的距离
    deltas = np.absolute(center - np.arange(domain))
    distances = np.minimum(deltas, domain - deltas)
    
    # 返回该神经元周围的高斯分布
    return np.exp((-(distances * distances)) / (2 * (radix * radix)))

def get_route(cities, network):
    """
    返回tsp路线
    """
    # 找出距离每个城市最近的神经元
    cities['winner'] =cities[['x','y']].apply(lambda c:select_closest(network, c) ,axis = 1, raw =True)
    
    # 返回神经元按一定顺序排序的数列
    return cities.sort_values('winner').index