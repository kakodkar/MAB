import numpy as np

a = None
for i in np.random.permutation(range(0,31,1)) :
    if a is None :
        a = str(i)
    else:
        a += ","+str(i)



a = None
for i in (range(31,0,-1)) :
    if a is None :
        a = str(i)
    else:
        a += ","+str(i)


print a