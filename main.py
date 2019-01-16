from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import math


### reading content of files ###
ir5_content = pd.read_csv('ir5.xyz', sep='\s+', skiprows=2, header=None)
fau_content = pd.read_csv('FAU.car', sep='\s+', skiprows=5, usecols=np.arange(1,4), header=None, skipfooter=2)

### initiate figure in pyplot ###
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')


### divide cords into corresponding lists for X, Y, Z ###
fau_x = fau_content[1][:]
fau_y = fau_content[2][:]
fau_z = fau_content[3][:]
### find the center of fau ###
fau_avg = np.array([np.mean(fau_x), np.mean(fau_y), np.mean(fau_z)])

### -//- ###
ir5_x = ir5_content[1][:]
ir5_y = ir5_content[2][:]
ir5_z = ir5_content[3][:]
ir5_avg = np.array([np.mean(ir5_x), np.mean(ir5_y), np.mean(ir5_z)])

### vector corresponding to moving ir5 to center of fau ###
move = np.abs(fau_avg) + np.abs(ir5_avg)

### move ir5 to center of fau ###
new_x = ir5_x + move[0]
new_y = ir5_y + move[1]
new_z = ir5_z + move[2]

### create points in 3D ###
ax.scatter(fau_x, fau_y, fau_z, marker='.', s=100)
ax.scatter(new_x, new_y, new_z, marker='.', s=100, color='red')

### find min/max values for every dimension ###
maxx = max(fau_x)
minx = min(fau_x)
maxy = max(fau_y)
miny = min(fau_y)
maxz = max(fau_z)
minz = min(fau_z)



### rotate and move vector by a random rotation matrix ###
def rotation_matrix(axis, theta):

    ### Return the rotation matrix associated with counterclockwise rotation about
    ### the given axis by theta radians.

    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

atoms_initial = np.array([new_x, new_y, new_z]).T




proper = open('proper.txt', 'w')
count = 1 ### counts iterations
storage = []
while count<=20:
    store=[]
    rotation_axis = np.random.uniform(low=0, high=1, size=3)  # three numbers between 0 and 1
    rotation_angle = np.random.uniform(low=0, high=2 * np.pi, size=1)  # random number between 0 and 2pi
    print("Rotation axis:{}, rotation angle:{} radians".format(rotation_axis, rotation_angle))

    ###  create rotation matrix
    rotmat = rotation_matrix(rotation_axis, rotation_angle)

    ### apply rotation matrix to points
    atoms_rotated = np.dot(atoms_initial, rotmat)

    ### check whether atoms are in desired position
    for x, y, z in atoms_rotated:
        if x < (maxx-1.5) and x > (minx+1.5) and y < (maxy-1.5) and y > (miny+1.5) and z < (maxz-1.5) and z > (minz+1.5):
            for a, b, c in zip(fau_x, fau_y, fau_z):
                dist = abs(math.sqrt((a-x)**2 + (b-y)**2 + (c-z)**2))
                if dist>=1.5 and dist<=3.5:
                    store.append((x,y,z))
                    break



    if len(store)==5 and store not in storage:
        print('Placement found!')
        storage.append(store)
        count+=1

for elem in storage:
    proper.write('{} placement: \n'.format(storage.index(elem)+1))
    for cord in elem:
        proper.write('{}\n'.format(cord))
    proper.write('\n')


proper.close()



