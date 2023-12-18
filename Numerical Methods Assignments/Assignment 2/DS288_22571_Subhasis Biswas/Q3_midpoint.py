import time
import math
import numpy as np


def f(x):
    return x**2+math.cos(30*math.e*x)

def Df(x):
    return 2*x-30*math.e*math.sin(30*math.e*x)

step=2*math.pi/(2*30*math.e)


def sgn(x):      # The signum function
    if(x>0):
        return 1
    elif(x<0):
        return -1
    else:
        return 0


def newtonraphson(initial_approx):
    x=initial_approx
    if(Df(x)==0):
        return None
    x_new=x-f(x)/Df(x)
    while(abs(f(x_new))>10**(-6) or (abs(x_new-x)/abs(x_new))>10**(-6)):
        x=x_new
        if(Df(x)==0):
            return None
        x_new=x-f(x)/Df(x)
        if(x_new<initial_approx-step/2 or x_new>initial_approx+step/2):
            return None
        if(f(x_new)==0):
            return x_new
    return x_new


def mullermethod(x0,x1,x2,epsilon):
    init_left=x0
    init_right=x2
    init_mid=x1
    epsilon=10**(-6)
    while(abs(x1-x2)/abs(x2)>epsilon or abs(f(x2))>epsilon):
        
        a=((x1-x2)*(f(x0)-f(x2))-(x0-x2)*(f(x1)-f(x2)))/((x0-x2)*(x1-x2)*(x0-x1))    # Coefficients of the quadratic polynomial
        b=((x0-x2)**2*(f(x1)-f(x2))-(x1-x2)**2*(f(x0)-f(x2)))/((x0-x2)*(x1-x2)*(x0-x1)) 
        c=f(x2)
        if(b**2-4*a*c<0): #check if the discriminant is negative, if yes, return None
            return None

        x3=x2-2*c/(b+sgn(b)*math.sqrt(b**2-4*a*c)) # Muller's formula

        x0,x1,x2=x1,x2,x3
        if(x0<init_left or x2>init_right):  # Checking if the approximation is in the interval,                 
            return None     #if not, stop the iteration and return None
    return x3






def secantmethod(init_left,init_right):
    a=init_left
    b=init_right
    if(f(b)==f(a)):
        return None
    c=(a*f(b)-b*f(a))/(f(b)-f(a))
    while(abs(f(c))>10**(-6) or abs(c-b)/abs(c)>10**(-6)):
        a=b
        b=c
        if(f(b)==f(a)):
            return None
        c=(a*f(b)-b*f(a))/(f(b)-f(a))
        if(f(c)==0):
            return c
        if(a<init_left or b>init_right):
            return None
    return c
        


def regulafalsi(init_left,init_right):
    a=init_left
    b=init_right
    if(f(b)==f(a)):
        return None
    c=(a*f(b)-b*f(a))/(f(b)-f(a))
    old=b
    current=c
    while(abs(f(current))>10**(-6) or abs(current-old)/abs(old)>10**(-6)):
        old=c
        if(f(b)==f(a)):
            return None
        c=(a*f(b)-b*f(a))/(f(b)-f(a))
        if(f(c)==0):
            return c
        if(f(a)*f(c)<0):
            b=c
            current=b
        else:
            a=c
            current=a
    return current 
    


def bisection(init_left,init_right):
    a=init_left
    b=init_right
    old=b
    c=(a+b)/2
    current=c
    while(abs(f(current))>10**(-6) or abs(current-old)/abs(current)>10**(-6)):
        old=current
        c=(a+b)/2
        if(f(c)*f(a)<0):
            b=c
            current=b
        else:
            if(f(c)==0):
                return c
            else:
                a=c
                current=a
    return current

fail_count={"Newton Raphson":0,"Muller's Method":0,"Secant Method":0,"Regula Falsi":0}  # Dictionary to keep track of the number of failures in each method

def root_finder(x,y):
    newtonraphson_output=newtonraphson((x+y)/2)  # Initial approximation for Newton Raphson is the midpoint of the interval
                                                                
    if(newtonraphson_output!=None):
        return newtonraphson_output
    fail_count["Newton Raphson"]+=1
    mullermethod_output=mullermethod(x,(x+y)/2,y,10**(-6))   # Initial approximations for Muller's Method are the endpoints and the midpoint
    if(mullermethod_output!=None):
        return mullermethod_output
    fail_count["Muller's Method"]+=1
    secantmethod_output=secantmethod(x,y)
    if(secantmethod_output!=None):
        return secantmethod_output
    fail_count["Secant Method"]+=1
    regulafalsi_output=regulafalsi(x,y)
    if(regulafalsi_output!=None):
        return regulafalsi_output
    fail_count["Regula Falsi"]+=1
    bisection_output=bisection(x,y)
    
    return bisection_output



run_count=0

time_list=list()
K=1000       # Number of times the main loop is run

while(run_count<=K):
    run_count+=1
    start=time.time()
    root_list=list()
    endpoints=[]
    for i in np.arange(0,1,step):
        if(f(i)*f(i+step)<0):
            endpoints+=[i]
        if(f(i)==0):
            root_list+=[i]

    for i in range(len(endpoints)):
        root_list+=[root_finder(endpoints[i],endpoints[i]+step)]

    end=time.time()
    time_list+=[end-start]



print("Time taken by running the main loop",K,"times:",np.mean(time_list)) # Mean time taken by the program to run the loop K times
print("List of roots:",root_list)
print("Number of roots:",len(root_list))
print("Number of failures in methods (Given that the previous has failed):\n",fail_count, "out of total",K*len(root_list),"root finding attempts")

