import numpy as np
import pandas as pd



class Extr:

    def __init__(self, h, x0):
        self.h = h
        self.x0 = x0

    def silvester(self):
        d = []
        for i in range(len(self.h)):
            m = [[] for l in range(i + 1)]
            for j in range(i + 1):
                for k in range(i + 1):
                    m[j].append(self.h[j][k])
            d.append(np.linalg.det(np.array(m)))
        return d

    def f(self, vec):
        return -3*vec[0]**2 - 2*vec[1]**2 - 4*vec[2]**2 + vec[0]*vec[1] - 2*vec[0]*vec[2] + 6*vec[0]

    def grad(self, vec):
        vec1 = -6 * vec[0] + vec[1] - 2 * vec[2] + 6
        vec2 = vec[0] - 4 * vec[1]
        vec3 = -2 * vec[0] - 8 * vec[2]
        return np.array([vec1, vec2, vec3])

    def f_n(self, vec):
        return 3*vec[0]**2 + 2*vec[1]**2 + 4*vec[2]**2 - vec[0]*vec[1] + 2*vec[0]*vec[2] - 6*vec[0]

    def grad_f_n(self, vec):
        vec1 = 6*vec[0] - vec[1] + 2*vec[2] - 6
        vec2 = -vec[0] + 4*vec[1]
        vec3 = 2*vec[0] + 8*vec[2]
        return np.array([vec1, vec2, vec3])

    def norma(self, vec):
        s = 0
        for i in vec:
            s += i**2

        return np.sqrt(s)

    def coord_pk(self, k, xk):
        i = k % 3
        if i == 0:
            return -1*np.array([6*xk[0] - xk[1] + 2*xk[2] - 6, 0, 0])
        elif i == 1 % 3:
            return -1*np.array([0, -xk[0] + 4*xk[1], 0])
        elif i == 2 % 3:
            return -1*np.array([0, 0, 2*xk[0] + 8*xk[2]])

    def newton_pk(self, xk):
        return np.linalg.solve(self.h, self.grad_f_n(xk))

    def fletch_pk(self, k, df):
        xk = df.loc[k]['xk']
        if k == 0:
            return -1*self.grad_f_n(xk)
        else:
            return -1*self.grad_f_n(xk) + (self.norma(self.grad_f_n(df.loc[k]['xk']))**2 / (self.norma(self.grad_f_n(df.loc[k-1]['xk'])))**2) * self.fletch_pk(k-1, df)

    def grad_descent(self):
        data = pd.DataFrame({'xk': [self.x0], 'f(xk)': [self.f(self.x0)], '||grad(f(xk))||': [self.norma(self.grad(self.x0))]})
        xk = self.x0
        tk = 1/2
        while self.norma(self.grad_f_n(xk)) > 10**-3:
            pk = -1*self.grad_f_n(xk)
            print(pk)
            if self.f_n(xk) > self.f_n(xk + tk*pk):
                xk = xk + tk*pk
                data.loc[len(data.index)] = [xk, self.f(xk), self.norma(self.grad(xk))]
            else:
                tk = tk/2
        return data

    def steep_descent(self):
        data = pd.DataFrame(
            {'xk': [self.x0], 'f(xk)': [self.f(self.x0)], '||grad(f(xk))||': [self.norma(self.grad(self.x0))]})
        xk = self.x0
        while self.norma(self.grad_f_n(xk)) > 10**-3:
            pk = -1*self.grad_f_n(xk)
            tk = self.min_d_ft(xk, pk)
            if self.f_n(xk) > self.f_n(xk + tk*pk):
                xk = xk + tk*pk
                data.loc[len(data.index)] = [xk, self.f(xk), self.norma(self.grad(xk))]

        return data

    def d_ft(self, x, p, t):
        s1 = -6*x[0]*p[0] - 6*t*p[0]**2 - 4*p[1]*x[1] - 4*t*p[1]**2 - 8*x[2]*p[2] - 8*t*p[2]**2 + x[0]*p[1] + x[1]*p[0]
        s2 = 2*t*p[0]*p[1] - 2*x[0]*p[2] - 2*x[2]*p[0] - 4*t*p[0]*p[2] + 6*p[0]
        return s1 + s2

    def min_d_ft(self, x, p):
        s1 = 6*x[0]*p[0] + 4*p[1]*x[1] + 8*x[2]*p[2] - x[0]*p[1] - x[1]*p[0] + 2*x[0]*p[2] + 2*x[2]*p[0] - 6*p[0]
        s2 = -6*p[0]**2 - 4*p[1]**2 - 8*p[2]**2 + 2*p[0]*p[1] - 4*p[0]*p[2]
        return s1/s2

    def coordinate_descent(self):
        data = pd.DataFrame(
            {'xk': [self.x0], 'f(xk)': [self.f(self.x0)], '||grad(f(xk))||': [self.norma(self.grad(self.x0))]})
        xk = self.x0
        k = 0
        while self.norma(self.grad_f_n(xk)) > 10 ** -3:
            pk = self.coord_pk(k, xk)
            tk = self.min_d_ft(xk, pk)
            print(self.f_n(xk), self.f_n(xk + tk * pk))
            if self.f_n(xk) > self.f_n(xk + tk * pk):
                xk = xk + tk * pk
                data.loc[len(data.index)] = [xk, self.f(xk), self.norma(self.grad(xk))]
            k += 1

        return data

    def newton(self):
        data = pd.DataFrame(
            {'xk': [self.x0], 'f(xk)': [self.f(self.x0)], '||grad(f(xk))||': [self.norma(self.grad(self.x0))]})
        xk = self.x0
        while self.norma(self.grad_f_n(xk)) > 10 ** -3:
            pk = self.newton_pk(xk)
            tk = 1
            if self.f_n(xk) > self.f_n(xk + tk * pk):
                xk = xk + tk * pk
                data.loc[len(data.index)] = [xk, self.f(xk), self.norma(self.grad(xk))]

        return data

    def fletcher(self):
        data = pd.DataFrame(
            {'xk': [self.x0], 'f(xk)': [self.f(self.x0)], '||grad(f(xk))||': [self.norma(self.grad(self.x0))]})
        xk = self.x0
        k = 0
        while self.norma(self.grad_f_n(xk)) > 10 ** -3:
            pk = self.fletch_pk(k, data)
            tk = self.min_d_ft(xk, pk)
            print(pk, tk, xk)
            if self.f_n(xk) > self.f_n(xk + tk * pk):
                xk = xk + tk * pk
                data.loc[len(data.index)] = [xk, self.f(xk), self.norma(self.grad(xk))]
            k += 1

        return data








