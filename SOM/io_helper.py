import pandas as pd
import numpy as np

def read_tsp(filename):
    """
    读取read_tsp数据 转化为 pandas DataFrame
    """
    with open(filename) as file:
        node_coord_start  = None
        city_number = None
        lines = file.readlines()
        i = 0
        # 获取tsp数据
        while True:
            line = lines[i]
            if line.startswith('DIMENSION :'):
                #获取城市数量 
                city_number = int(line.split()[-1])
            if line.startswith('NODE_COORD_SECTION'):
                node_coord_start = i
                break
            i = i + 1
        print(' {} 个城市数据被读取.'.format(city_number))
        file.seek(0)

        # 读取各个城市的坐标信息转化为Dataframe
        cities = pd.read_csv(
            file,
            skiprows = node_coord_start +1,
            sep = ' ',
            names = ['city', 'y', 'x'],
            dtype = {'city' : str, 'x' : np.float64, 'y' : np.float64},
            header = None,
            nrows = city_number 
        )
        
        return cities

def  normalize(points):
    """
    对向量进行归一化处理，[0,1]在y上，将原始比例维持在x上来标准化数据
    """
    ratio = (points.x.max() - points.x.min()) / (points.y.max() - points.y.min()),1
    
    ratio = np.array(ratio) / max(ratio)
    
    norm = points.apply(lambda c: (c - c.min()) / (c.max() - c.min()))
    
    return norm.apply(lambda p: ratio * p, axis = 1)