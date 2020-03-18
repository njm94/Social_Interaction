#%% Graph to visualize experiment combinations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math


# Two dimensional rotation
# returns coordinates in a tuple (x,y)
def rotate(x, y, r):
    rx = (x*math.cos(r)) - (y*math.sin(r))
    ry = (y*math.cos(r)) + (x*math.sin(r))
    return rx, ry

# create a ring of points centered on center (x,y) with a given radius
# using the specified number of points
# center should be a tuple or list of coordinates (x,y)
# returns a list of point coordinates in tuples
# ie. [(x1,y1),(x2,y2
def point_ring(center, num_points, radius):
    arc = (2 * math.pi) / num_points # what is the angle between two of the points
    points = []
    for p in range(num_points):
        (px,py) = rotate(0, radius, arc * p)
        px += center[0]
        py += center[1]
        points.append((px,py))

    points = np.array(points)
    return points

# edges females
edges = [['DK1', 'DK4', 1.0],
         ['DK1', 'DK2', 2.0],
         ['DK1', 'DK3', 7.0],
         ['DK2', 'DK4', 5.0],
         ['DK3', 'DK4', 5.0],
         ['DB1', 'DB2', 2.0],
         ['DB1', 'DB3', 2.0],
         ['DB1', 'DB4', 4.0],
         ['DB2', 'DB3', 4.0],
         ['DB2', 'DB4', 3.0],
         ['BV2', 'BV3', 2.0],
         ['DK1', 'DB1', 1.0],
         ['DK2', 'DB1', 2.0],
         ['DK2', 'DB3', 2.0],
         ['DK4', 'DB3', 3.0],
         ['DK4', 'DB4', 3.0],
         ['DK2', 'BV2', 1.0],
         ['DK4', 'BV2', 1.0],
         ['DB3', 'BV2', 1.0],
         ['DB4', 'BV2', 2.0],
         ['BU1', 'BU2', 2.0],
         ['BU1', 'BU3', 2.0],
         ['BU2', 'BU3', 1.0]]


node_names = ['DK1', 'DK2', 'DK3', 'DK4', 'DB1', 'DB2', 'DB3', 'DB4', 'BV2', 'BV3', 'BU1', 'BU2', 'BU3']

# males
edges = [['DK1', 'DK4', 1.0],
         ['DK1', 'DK2', 2.0],
         ['DK1', 'DK3', 7.0],
         ['DK2', 'DK4', 5.0],
         ['DK3', 'DK4', 5.0],
         ['DB1', 'DB2', 2.0],
         ['DB1', 'DB3', 2.0],
         ['DB1', 'DB4', 4.0],
         ['DB2', 'DB3', 4.0],
         ['DB2', 'DB4', 3.0],
         ['BV2', 'BV3', 2.0],
         ['DK1', 'DB1', 1.0],
         ['DK2', 'DB1', 2.0],
         ['DK2', 'DB3', 2.0],
         ['DK4', 'DB3', 3.0],
         ['DK4', 'DB4', 3.0],
         ['DK2', 'BV2', 1.0],
         ['DK4', 'BV2', 1.0],
         ['DB3', 'BV2', 1.0],
         ['DB4', 'BV2', 2.0]]

node_names = ['DK1', 'DK2', 'DK3', 'DK4', 'DB1', 'DB2', 'DB3', 'DB4', 'BV2', 'BV3']

node_pos = point_ring((0, 0), len(node_names), 2)
all_nodes = np.vstack( (node_names, node_pos[:,0], node_pos[:,1], np.zeros((1, node_pos.shape[0]))) )
df = pd.DataFrame(np.transpose(all_nodes), columns = ['Name', 'x', 'y', 'color'])

# assign colors to cages
c_array = ['c','y','k','m']
c_count = 0
for i in np.arange(len(node_names)):
    if i == 0:
        df.color[i] = c_array[c_count]
    else:
        if df.Name[i][0:2] == df.Name[i-1][0:2]:
            df.color[i] = df.color[i-1]
        else:
            c_count += 1
            df.color[i] = c_array[c_count]


plt.figure()

# add edges to plot
for i in np.arange(len(edges)):
    temp1 = df[df.Name == edges[i][0]]
    x1 = pd.to_numeric(temp1.x.values, errors='coerce')
    x1 = x1[0]
    y1 = pd.to_numeric(temp1.y.values, errors='coerce')
    y1 = y1[0]

    temp2 = df[df.Name == edges[i][1]]
    x2 = pd.to_numeric(temp2.x.values, errors='coerce')
    x2 = x2[0]
    y2 = pd.to_numeric(temp2.y.values, errors='coerce')
    y2 = y2[0]

    plt.plot((x1, x2), (y1, y2), 'k', linewidth = edges[i][2], zorder=1)


plt.scatter(pd.to_numeric(df.x, errors='coerce').fillna(0, downcast='infer'),
            pd.to_numeric(df.y, errors='coerce').fillna(0, downcast='infer'),
            s=500, zorder=2, color=df.color)
plt.axis('off')

# position males
plt.text(-2.5, 1.4, 'Cage 1', fontsize=20)
plt.text(0.75, -2.4, 'Cage 2', fontsize=20)
plt.text(1.75, 1.4, 'Cage 3', fontsize=20)


# position females
# plt.text(-2.2, 1.75, 'Cage 1', fontsize=20)
# plt.text(-1.8, -2.3, 'Cage 2', fontsize=20)
# plt.text(1.8, -1.5, 'Cage 3', fontsize=20)
# plt.text(1.9, 1.25, 'Cage 4', fontsize=20)
plt.savefig('B:/Social_Outputs/group_data/experiment_graph_males.pdf', format='pdf')
plt.show()

