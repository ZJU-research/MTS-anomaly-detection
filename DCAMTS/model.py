'''
--------------------------------------- description--------------------------------------------------
@author:Mengcheng Fang
@time:2021.3.26
@Use: Import the model.py file and run model.Run_model() directly
@funtion:c=model.Run_model(data,D,Choose_P,training_step,temp,d,detection_key,local_K)
@Parameter Description:
    data: The input data is a two-dimensional data matrix, the column direction is the variable, and the horizontal direction is the data corresponding to each variable. If there are P pieces of data, each piece has N data points, and the data size is [P,N].
    D: It is int type data, for the parameters of interval anomaly detection, it is the size of the detected interval, which needs to be adjusted by yourself.
    Choose_P: is a list, for the parameters of specific target detection, the list is the subscript of the variable that needs specific identification, if it is the general anomaly detection mode, enter an empty list for this parameter [].
    training_step: the number of iterative operations.
    temp: floating point number, similarity calculation parameter.
    d: Floating point number, used to calculate the parameter d/n+(1-d)Sc of the iterative formula.
    detection_key: integer, the type of recognition, if it is 1, point-to-point anomaly detection, if it is 2, interval anomaly detection, if it is 3, local anomaly detection.
    detection_key: The data type is an integer, which is the type of the model operation method. If it is 1, point-to-point anomaly detection is performed, if it is 2, interval anomaly detection is performed, and if it is 3, local anomaly detection is performed.
    local_K: integer, the local width for local anomaly detection.
'''

import numpy as np
from multiprocessing import Manager
from multiprocessing import Process
#Solve a system of linear equations
from scipy import linalg
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #Used to display Chinese labels normally
plt.rcParams['axes.unicode_minus']=False #Used to display the negative sign normally
#temp is the parameter
def RBF_function(x,I,J,P,D,temp):#A function to find the similarity of two data nodes
    res=0
    node_nums=len(x[0])
    for i in range(len(P)):#Calculate the variables specified in P
        ans=0
        for l in range(D):#Calculated length
            if I+l>=node_nums or J+l>=node_nums:
                break
            k=P[i]
            ans+=(x[k][I+l]-x[k][J+l])**2
        res+=(ans/np.var(x[P[i]]))
    return np.exp(-res)

def Train_model(S,c,d,n,training_step):#Training model
    for i in range(training_step):
        c=d/n+(1-d)*np.matmul(S,c)
        c=(c-min(c))/(max(c)-min(c))
    return c

def  Frobenius_function(A,B):
    return sum(sum(A*B))

def Local_detection(N,K,local_K):#District anomaly detection, delete edges
    for i in range(N):
        for j in range(N):
            if abs(j-i)>local_K:
                K[i][j]=0.0#Delete the edge between node i and node j
    return K

def Normalize_matrix(A):#For matrix numerical normalization, it is necessary to ensure symmetry
    return (A-np.min(A))/(np.max(A)-np.min(A))

def Aligned_kernel_matrix(index,data,N,D,P,y,temp,detection_key,local_K,common_data):#Return aligned kernel matrix
    print('process'+str(index+1)+'starts to calculate ......')
    Kx=np.zeros([N,N],dtype=float)
    Ky=np.zeros([N,N],dtype=float)
    for i in range(N):
        for j in range(N):
            Ky[i][j]=RBF_function(data,i,j,[y],D,temp)

    if detection_key==3:#Perform local anomaly detection processing
        Ky=Local_detection(N,Ky,local_K)

    X=[]
    for i in range(P):
        if i==y:
            continue
        X.append(data[i])
    for i in range(N):
        for j in range(N):
            Kx[i][j]=RBF_function(X,i,j,np.arange(0,P-1),D,temp)

    if detection_key==3:#Perform local anomaly detection processing
        Kx=Local_detection(N,Kx,local_K)

    Kx=Normalize_matrix(Kx)#Normalize the matrix to ensure symmetry, otherwise the eigenvalues may be imaginary numbers

    w, v = np.linalg.eig(Kx)
    w=w.real#Take the real part
    v=v.real#Take the real part

    C=[]
    for i in range(N):
        C.append(w[i]+(Frobenius_function(np.matmul(v[:,i].reshape([N,1]),v[:,i].reshape([1,N])),Ky))/2)
    K_temp = np.zeros([N, N], dtype=float)

    for i in range(N):
        K_temp+=(C[i]*np.matmul(v[:,i].reshape([N,1]),v[:,i].reshape([1,N])))
    common_data.append(K_temp)#Store the result of the calculation

def Normalize_function(data):#Normalize a general array
    new_data=np.zeros([data.shape[0],data.shape[1]],dtype=float)
    for i in range(len(data)):
        if max(data[i])-min(data[i])<=10**-8:
            continue
        new_data[i]=(data[i]-min(data[i]))/(max(data[i])-min(data[i]))
    return new_data

def del_zero_matrix(data):#Delete useless variables
    X=[]
    for i in range(len(data)):
        if max(data[i])-min(data[i])<=10**-8:
            continue
        X.append(data[i])
    return np.array(X)



def Run_model(data,D,Choose_P,training_step,temp,d,detection_key,local_K):
    if detection_key==1:
        D=1
        print('Start point-to-point anomaly detection method... ')
        if len(Choose_P)==0:
            print('Detection target: all variables!')
        else:
            print('Detection target: specific variables!')
    if detection_key==2:
        print('Start interval anomaly detection method... ')
        if len(Choose_P)==0:
            print('Detection target: all variables!')
        else:
            print('Detection target: specific variables!')
    if detection_key==3:
        D=1
        print('Start local anomaly detection method... ')
        if len(Choose_P)==0:
            print('Detection target: all variables!')
        else:
            print('Detection target: specific variables!')
    data = Normalize_function(data)  # Data normalization

    data = del_zero_matrix(data)  # Delete useless variables
    N=len(data[0])#Data length, which is the number of nodes
    K_temp = np.identity(N)  # Initialize a unit vector
    P=len(data)#Number of variables

    if len(Choose_P)==0:#If it is empty, it means that there is no choice, and all are selected by default
        Choose_P=np.arange(0,P)#Select the variable information to be focused on

    print('The system detects '+str(len(Choose_P))+'target variables and starts '+str(len(Choose_P))+'processes ...')

    jobs = []#Storage process
    common_data = Manager().list()  # Here is a shared variable that declares a list
    for i in range(len(Choose_P)):#Start the corresponding process
        p=Process(target=Aligned_kernel_matrix,args=(i,data,N,D,P,Choose_P[i],temp,detection_key,local_K,common_data))#Share the common_data variable
        jobs.append(p)
        p.start()#Start process

    for proc in jobs:
        proc.join()#Use blocking to wait for all processes to end before proceeding

    K_temp = np.identity(N)  # Initialize a unit vector
    for i in range(len(Choose_P)):
        K_temp=np.matmul(K_temp,common_data[i])#Multiply all matrices

    S = Normalize_matrix(K_temp)  # Normalized

    print('The calculation is over... ')

    c = np.array([1] * N)
    c = Train_model(S, c,d, N, training_step)

    return c