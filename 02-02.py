import numpy as np

""" 
# 1D
t = np.array([0., 1., 2., 3., 4., 5., 6.])

print(t)  #[0. 1. 2. 3. 4. 5. 6.]
print('Rank of t:', t.ndim)  #Rank of t: 1 -> 1차원
print('Shape of t:', t.shape)  #Shape of t: (7,) -> 크기(1,7) = (1x7)
print('t[0] t[1] t[-1] = ', t[0], t[1], t[-1])  #t[0] t[1] t[-1] =  0.0 1.0 6.0
print('t[2:5] t[4:-1] = ', t[2:5], t[4:-1])  #t[2:5] t[4:-1] =  [2. 3. 4.] [4. 5.]
print('t[:2] t[3:] = ', t[:2], t[3:])  #t[:2] t[3:] =  [0. 1.] [3. 4. 5. 6.]
"""
"""
# 2D
t = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [10., 11., 12.]])
print(t)
print('Rank of t: ', t.ndim)  #Rank of t:  2
print('Shape of t: ', t.shape) #Shape of t:  (4, 3)
"""

import torch

"""
# 1D
t = torch.FloatTensor([0., 1., 2., 3., 4., 5., 6.])
print(t)
print(t.dim())  #1
print(t.shape)  #torch.Size([7])
print(t.size())  #torch.Size([7])
print(t[0], t[1], t[-1])  #tensor(0.), tensor(1.) tensor(6.)
print(t[2:5], t[:2])  #tensor([2., 3., 4.]) tensor([0., 1.])
"""
"""
# 2D
t = torch.FloatTensor([[1., 2., 3.], 
                       [4., 5., 6.], 
                       [7., 8., 9.], 
                       [10., 11., 12.]
                      ])
print(t)
print(t.dim())  #2
print(t.shape)  #torch.Size([4, 3])
print(t.size())

print(t[:, 1])  #tensor([ 2.,  5.,  8., 11.])
print(t[:, 1].size())  #torch.Size([4])

print(t[:, :-1])
"""
"""
# Broadcasting
m1 = torch.FloatTensor([[1, 2]]) #dim: (1,2)
m2 = torch.FloatTensor([[3], [4]]) #dim: (2,1)
print(m1 + m2)
# [1, 2] -> [[1,2],
#            [2, 1]]
# [3]
# [4] -> [[3, 3],
#         [4, 4]]
"""

# Mean
t1 = torch.FloatTensor([1, 2])
print(t1.mean())  #tensor(1.5000)

t2 = torch.FloatTensor([[1, 2], [3, 4]])
print(t2.mean())  #tensor(2.5000)