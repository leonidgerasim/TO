import numpy as np
import pandas as pd
from Lab1_TO import *

a = np.array([[-6., 1., -2.],
              [1., -4., 0.],
              [-2., 0., -8.]])

b = np.array([-6., 0., 0.])

x = np.linalg.solve(a, b)

print(x)

h = np.array([[-6., 1., -2.],
              [1., -4., 0.],
              [-2., 0., -8.]])

x0 = np.array([0., 0., 0.])

extr = Extr(h, x0)

print(extr.f(x))
print(np.linalg.det(h))

# g_descent = extr.grad_descent()
# for i in g_descent.index:
#     print(i)
#     print(g_descent.loc[i]['xk'])
#     print(g_descent.loc[i]['f(xk)'])
#     print(g_descent.loc[i]['||grad(f(xk))||'])
#
# s_descent = extr.steep_descent()
# for i in s_descent.index:
#     print(i)
#     print(s_descent.loc[i]['xk'])
#     print(s_descent.loc[i]['f(xk)'])
#     print(s_descent.loc[i]['||grad(f(xk))||'])

# c_descent = extr.coordinate_descent()
# for i in c_descent.index:
#     print(i)
#     print(c_descent.loc[i]['xk'])
#     print(c_descent.loc[i]['f(xk)'])
#     print(c_descent.loc[i]['||grad(f(xk))||'])

# newton = extr.newton()
# for i in newton.index:
#     print(i)
#     print(newton.loc[i]['xk'])
#     print(newton.loc[i]['f(xk)'])
#     print(newton.loc[i]['||grad(f(xk))||'])

fletcher = extr.fletcher()
for i in fletcher.index:
    print(i)
    print(fletcher.loc[i]['xk'])
    print(fletcher.loc[i]['f(xk)'])
    print(fletcher.loc[i]['||grad(f(xk))||'])




