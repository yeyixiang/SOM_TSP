import numpy as np

def select_closest(candidates, origin):
    """
    返回最近距离的候选者
    """
    
    return euclidean_distance(candidates, origin).argmin()

def euclidean_distance(a, b):
    """ 
    返回两个点数组的距离数组
    """
    return np.linalg.norm(a - b ,axis = 1)

def route_distance(cities):
    '''
    按一定顺序返回所有城市的路线的总长度
    '''
    
    points = cities[['x','y']]
    
    distance = euclidean_distance(points, np.roll(points, 1, axis = 0))

    return np.sum(distance)