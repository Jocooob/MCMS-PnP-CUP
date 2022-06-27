# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 14:04:41 2022

@author: Administrator
"""

import cvxpy as cp
import numpy

def solve_least_squares_problem(A, B):
    #m = A.shape[0]
    n = A.shape[1]
    # A:m*n B:m*1 x:n*1

    # Define and solve the CVXPY problem.
    x = cp.Variable(n)
    
    B = B.squeeze()

    ones = numpy.ones((1,n))
    cost = cp.sum_squares(A @ x - B)
    prob = cp.Problem(cp.Minimize(cost),[0 <= x, x <= 1,ones@x == 1])
    prob.solve()

    xx = x.value.squeeze()
    
    #print("\nThe optimal value is", prob.value)
    return xx

def Alternate_solution(A, B, iter = 10):
    n = A.shape[1]
    wA = numpy.ones((n,1))/n
    wB = numpy.ones((n,1))/n
    for k in range(iter):
        if k%2==0:
            tA = A
            tB = B @ wB
            wA = solve_least_squares_problem(tA,tB)
        else:
            tA = B
            tB = A @ wA
            wB = solve_least_squares_problem(tA,tB)
    
    #print("\nThe optimal value is", wA, wB)
    
    return wA, wB
    

if __name__ == '__main__': 
    m = 3000
    n = 4
    #numpy.random.seed(1)
    A = numpy.random.rand(m, n)
    B = numpy.random.rand(m, n)
    A = B*[0.5,2,0.3,3]
    out = Alternate_solution(A, B, iter = 10)

    print("The optimal x is")
    print(out)
    print(numpy.sum(out))