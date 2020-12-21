import matplotlib.pyplot as plt
import matplotlib as mpl
import imageio

def plot_network(cities, neurons ,name = 'som_diagram.png' ,ax = None):
    """
    画出TSP问题的图解
    """
    # 数据量较大，设置'agg.path.chunksize'
    mpl.rcParams['agg.path.chunksize'] = 10000

    if not ax:
        fig = plt.figure(figsize = (5, 5), frameon = True)
        axis = fig.add_axes([0, 0, 1, 1])

        axis.set_aspect('equal', adjustable = 'datalim')
        plt.axis('off')

        axis.scatter(cities['x'], cities['y'], color = 'red' ,s = 4)
        axis.plot(neurons[:, 0], neurons[:,1], 'r.', ls = '-', color = '#0063ba', markersize = 2)

        plt.savefig(name, bbox_inches = 'tight', pad_inches = 0, dpi = 200)
        plt.close()
    else:
        ax.scatter(cities['x'], cities['y'], color = 'red', s = 4)
        ax.plot(neurons[:,0], neurons[:,1], 'r.', ls = '-', color = '#0063ba', markersize = 2)
        return ax

def plot_route(cities, route, name = 'som_diagrams', ax = None):
    """画出tsp路线"""

    mpl.rcParams['agg.path.chunksize'] = 10000

    if not ax :
        fig = plt.figure(figsize = (5, 5), frameon = False )
        axis = fig.add_axes([0, 0, 1, 1])

        axis.set_aspect('equal', adjustable = 'datalim')
        plt.axis('off')

        axis.scatter(cities['x'], cities['y'], color = 'red', s = 4)
        route = cities.reindex(route)
        route.loc[route.shape[0]] = route.iloc[0]
        axis.plot(route['x'], route['y'], color = 'purple',linewidth = 1 )
        plt.savefig(name, bbox_inches = 'tight', pad_inches = 0, dpi = 200)
        plt.close()
    else:
        ax.scatter(cities['x'], cities['y'], color='red', s=4)
        route = cities.reindex(route)
        route.loc[route.shape[0]] = route.iloc[0]
        ax.plot(route['x'], route['y'], color='purple', linewidth=1)
        return ax

def create_gif(source, name, duration):
    """
    将多个png图片合成gif图片
    """
    # 读入缓冲区
    frames = []
    for img in source:
        if img.endswith('.png'):
            frames.append(imageio.imread(img))
    imageio.mimsave(name, frames, 'GIF', duration = duration)
