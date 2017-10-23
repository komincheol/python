import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from scipy import spatial
import sklearn.manifold as manifold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def eig_Cxy(px_y,x,y):
    C = np.zeros([11,11])
    for i in range(len(px_y)):
        C = C+px_y[i]*np.outer(x[i]-y,x[i]-y)
    eigenvalues, eigenvector = eigh(C, eigvals_only=False, turbo=True)
    return eigenvalues[-1], eigenvector[-1]

def update_py_x(x, y, py, k, T):
    py_x=np.zeros([y.shape[0],x.shape[0]])
    for i in range(k):
        for j in range(x.shape[0]):
            py_x[i,j]=py[i]*np.exp(-(np.linalg.norm(x[j]-y[i])**2 / T)) 
    normalization = py_x.sum(0)
    py_x= py_x / normalization
    return py_x

def update_y(py_x, py, x,k):
    y=np.zeros([py.shape[0],11])
    n = x.shape[0]
    for i in range(k):
        for j in range(n):
            y[i]=y[i]+x[j]*py_x[i,j]/float(n)
        y[i]=y[i]/py[i]
    return y
    
def split(x, y, py_x, py, k, T, k_max):
    n=float(x.shape[0])
    for i in range(k):
        px_y = py_x[i,:] / py[i] / n
        T_crit, eigenvector = eig_Cxy(px_y,x,y[i])
        
#        if k==1:
#            tol=30
#        else:
#            tol=10
        print T, 2*T_crit
        if (T<2*T_crit) & (k<k_max):  #abs(T-2*T_crit)<10
            y[k] = y[i] + (np.random.rand(len(y[i]))-0.5)*2.0
#            y[i] = y[i] - eigenvector*0.1 
            py[i]=py[i]/2.0
            py[k]=py[i]
            k+=1
#            print i,k
#        print T_crit
    return y, py,k

def compute_d(py_x,x,y,k):
    c=0
    for i in range(x.shape[0]):
        for j in range(k):
            c=c+py_x[j,i]*np.linalg.norm(y[j]-x[i])
    
    return c

def new_split(x,y,py_x,py,k,T,k_max):
#    close = 0.5
    for i in range(k):
        if k + 1 <= k_max:
            py_old = np.copy(py)
            y_old = np.copy(y)
            py_x_old = np.copy(py_x)
            py[i] /= 2.0
            py[k] = py[i]
            eps = (np.random.rand(len(y[i]))-0.5)
            y[k] = y[i] + eps
            k+=1
            
            py_x=update_py_x(x, y, py, k, T)
            py=py_x.mean(1)
            y = update_y(py_x, py, data,k)
            dist = np.linalg.norm(y[i] - y[k-1])
            close = 0.7*np.linalg.norm(eps)
            if  dist < close : 
                py=py_old
                y=y_old
                py_x=py_x_old
                k=k-1
            else:	
                py_old[i] /= 2.0
                py_old[k-1]=py_old[i]
                y_old[k-1]=y_old[i] + eps
                py=np.copy(py_old)
                y=np.copy(y_old)
                py_x=py_x_old
                
	## Remove all Y which merge	
	#plot_clusters(X,Y)
	return y,py,k



#%%

f = open("winequality-red.csv", 'r')
red = f.readlines()
f.close()

f = open("winequality-white.csv", 'r')
white = f.readlines()
f.close()

red_feature = np.zeros([len(red)-1, 11])    # one for red
white_feature = np.zeros([len(white)-1, 11])    # zero for white
red_quality = np.ones([len(red)-1, 2])
white_quality = np.zeros([len(white)-1, 2])

#red whine
i=0
for line in red[1:]:
    linebits = line.split(';')
    red_feature[i] = np.asfarray(linebits[:-1])
    red_quality[i][1] = np.asfarray(linebits[-1])
    i+=1
#white wine
i=0
for line in white[1:]:
    linebits = line.split(';')
    white_feature[i] = np.asfarray(linebits[:-1])
    white_quality[i][1] = np.asfarray(linebits[-1])
    i+=1
    
n=500
data = np.append(red_feature[0:n/2],white_feature[0:n/2],0)
cluster = np.append(red_quality[0:n/2],white_quality[0:n/2],0)

#std_scaler = StandardScaler()
#data = std_scaler.fit_transform(data)


#%%

k_max = 5
T_min = 50

k=1
y=np.zeros([2*k_max,11])
py=np.zeros(k_max)
y[0]=np.mean(data,0)
py[0]=1
lambda_max, eigenvector = eig_Cxy(np.ones(n)/n,data,y[0])
T=3*lambda_max

T_crit = np.zeros(k_max)

alpha=0.98
py_x=np.zeros([k_max,n])
tol = 0.1

phase=np.zeros([2,1])
phase[0,0]=T
phase[1,0]=1
a=np.zeros([2,1])
D=np.array([])

y_old=np.zeros_like(y)


bifurcation=np.zeros([k_max,500])
count=1

#pca = PCA(n_components=1, copy=True)

while (T>20):
    
    
    
    current_d=compute_d(py_x,data,y,k)
    D=np.append(D,current_d)
    py_x=update_py_x(data, y, py, k, T)
    py=py_x.mean(1)
    y_old = np.copy(y)
    y = update_y(py_x, py, data,k)
    while (np.linalg.norm(y-y_old)>tol):
        py_x=update_py_x(data, y, py, k, T)
        py=py_x.mean(1)
        y_old = np.copy(y)
        y = update_y(py_x, py, data,k)
    T=T*alpha
    if k < k_max:
        y,py,k = split(data, y, py_x, py, k, T, k_max)
#        y,py,k = new_split(data, y, py_x, py, k, T, k_max)
    
    a=np.zeros([2,1])
    a[0,0]=T
    a[1,0]=k
    phase=np.append(phase,a,1)
    
    # bifurcation
    bifurcation[:,count]=np.linalg.norm(y[0:k_max],axis=1)
    count+=1
    print T,k
    
#    print current_d
    
  #%%
for i in range(py_x.shape[1]):
    a=py_x[:,i].max()
    for j in range(py_x.shape[0]):
        if py_x[j,i]<a:
            py_x[j,i]=0
        else: 
            py_x[j,i]=1
            
#%% bifurcation

#for i in range(phase.shape[1]):
#    for j in range(int(phase[1,i])):
#        plt.text(bifurcation[j,i], phase[0,i], str('yes'),
#                fontdict={ 'size': 9})
        
#x0=-bifurcation[0,0:phase.shape[1]]/2
#x1=+bifurcation[0,0:phase.shape[1]]/2
#plt.plot(x0,phase[0,:])
#plt.plot(x1,phase[0,:])
#x2=bifurcation[1,0:phase.shape[1]]+x1
#plt.plot(x2,phase[0,:])
#x3=bifurcation[2,0:phase.shape[1]]+x2
#plt.plot(x3,phase[0,:])
#x4=-bifurcation[3,0:phase.shape[1]]+x1
#plt.plot(x4,phase[0,:])
#plt.plot(bifurcation[0,1:phase.shape[1]],phase[0,1:])
#plt.plot(bifurcation[1,34:phase.shape[1]],phase[0,34:])
#plt.plot(bifurcation[2,90:phase.shape[1]],phase[0,90:])
#plt.plot(bifurcation[3,125:phase.shape[1]],phase[0,125:])
#plt.plot(bifurcation[4,167:phase.shape[1]],phase[0,167:])
#plt.plot(bifurcation[5,0:phase.shape[1]],phase[0,:],'k.')
#plt.ylabel('T')
#plt.xlabel('Norm of y')
#plt.title("Bifurcation diagram")
#plt.savefig('bifurcation_peturb.eps')
            
        
#%%


plt.plot(np.log(phase[0,:-1].max()/phase[0,:-1]),np.log(D[:]/D[1:].min()),
         np.log(phase[0,:-1].max()/phase[0,:-1]),1.2-0.3*(phase[1,:-1]-1),
#         [0.4,0.4],[0,1.295],'k--',
#         [1.2,1.2],[0,0.87],'k--',
#         [1.45,1.45],[0,0.75],'k--'
         )
#plt.text(-0.08,0.05,'1')
#plt.text(0.75,0.05,'2')
#plt.text(1.25,0.01,'4')
#plt.text(1.25,0.09,'3')
#plt.text(2.3,0.05,'5')
##
#plt.ylabel('log(D/min(D))')
#plt.xlabel('log(beta/min(beta))')
#plt.title("Phase diagram for two clusters")
#plt.savefig('phase5.eps')

#%%  





#%%

visual = np.append(data,y[0:k_max],0)


slt_wine = manifold.LocallyLinearEmbedding(n_neighbors=15, n_components=2, neighbors_algorithm='kd_tree')
wine_2d = slt_wine.fit_transform(visual)



y_min, y_max = np.min(wine_2d, 0), np.max(wine_2d, 0)
wine_2d = (wine_2d - y_min) / (y_max - y_min)/1.05

#%%
plt.figure()
ax = plt.subplot(111)
for i in range(wine_2d.shape[0]-k_max):
    ax.text(wine_2d[i, 0], wine_2d[i, 1], str(int(cluster[i,1])),
             color=plt.cm.Set1(0.1+(1-cluster[i,0])*0.2),
             fontdict={ 'size': 9})
    
for i in range(k_max):
    ax.text(wine_2d[n+i, 0], wine_2d[n+i, 1], str('x'),
             
             fontdict={'weight': 'bold', 'size': 22})

#ax.text(wine_2d[53, 0], wine_2d[53, 1], str('.'),fontdict={'weight': 'bold', 'size': 22})
#ax.text(wine_2d[207, 0], wine_2d[207, 1], str('.'),fontdict={'weight': 'bold', 'size': 22})
#ax.text(wine_2d[215, 0], wine_2d[215, 1], str('.'),fontdict={'weight': 'bold', 'size': 22})
#ax.text(wine_2d[283, 0], wine_2d[283, 1], str('.'),fontdict={'weight': 'bold', 'size': 22})
#ax.text(wine_2d[315, 0], wine_2d[315, 1], str('.'),fontdict={'weight': 'bold', 'size': 22})
#ax.text(wine_2d[53, 0], wine_2d[53, 1], str('.'),fontdict={'weight': 'bold', 'size': 22})

ax.axes.xaxis.set_ticklabels([])
ax.axes.yaxis.set_ticklabels([])
plt.title("Decision boundary")
#plt.savefig('raw_data2.eps')

#ax.axis([0, 1, 0, 1])
#plt.plot([0.34,0.38],[0,1],'k--')
#plt.plot([0.34,0.63],[0,0.1],'k--')
#plt.plot([0.83,0.63],[1,0.1],'k--')
#
#plt.plot([0.42,0.455],[0,0.6],'k--')
#plt.plot([0.455,0.1],[0.6,1],'k--')
#
#plt.plot([0.77,0.61],[1,0.61],'k--')
#plt.plot([0.61,0.65],[0.61,0.35],'k--')
#plt.plot([0.65,1],[0.35,0.7],'k--')
#plt.savefig('boundary3.eps')


#%%

#plt.figure()
#ax = plt.subplot(111)
#
#for i in range(wine_2d.shape[0]-k_max):
#    ax.text(wine_2d[i, 0], wine_2d[i, 1], str(int(cluster[i,1])),
#             color=plt.cm.Set1(np.argmax(py_x[:,i])*0.25),
#             fontdict={'size': 7})
#    
#for i in range(2):
#    ax.text(wine_2d[n+i, 0], wine_2d[n+i, 1], str(i),
#             color=plt.cm.Set1(i*0.25),
#             fontdict={'weight': 'bold', 'size': 24})
#plt.title("Cluster assignments")
#ax.axes.xaxis.set_ticklabels([])
#ax.axes.yaxis.set_ticklabels([])
#plt.savefig('clustering2.eps')

#ax.text(wine_2d[397, 0], wine_2d[397, 1], str('.'),fontdict={'weight': 'bold', 'size': 22})
#ax.text(wine_2d[422, 0], wine_2d[422, 1], str('.'),fontdict={'weight': 'bold', 'size': 22})
#ax.text(wine_2d[439, 0], wine_2d[439, 1], str('.'),fontdict={'weight': 'bold', 'size': 22})
#ax.text(wine_2d[440, 0], wine_2d[440, 1], str('.'),fontdict={'weight': 'bold', 'size': 22})

#ax.text(wine_2d[279, 0], wine_2d[279, 1], str('o'),fontdict={'weight': 'bold', 'size': 22})
#ax.text(wine_2d[395, 0], wine_2d[395, 1], str('o'),fontdict={'weight': 'bold', 'size': 22})
#ax.text(wine_2d[499, 0], wine_2d[499, 1], str('o'),fontdict={'weight': 'bold', 'size': 22})
#
#ax.text(wine_2d[257, 0], wine_2d[257, 1], str('*'),fontdict={'weight': 'bold', 'size': 22})
#ax.text(wine_2d[300, 0], wine_2d[300, 1], str('*'),fontdict={'weight': 'bold', 'size': 22})
#ax.text(wine_2d[353, 0], wine_2d[353, 1], str('*'),fontdict={'weight': 'bold', 'size': 22})
#ax.text(wine_2d[396, 0], wine_2d[396, 1], str('*'),fontdict={'weight': 'bold', 'size': 22})
#ax.text(wine_2d[424, 0], wine_2d[424, 1], str('*'),fontdict={'weight': 'bold', 'size': 22})
#ax.text(wine_2d[432, 0], wine_2d[432, 1], str('*'),fontdict={'weight': 'bold', 'size': 22})
#ax.text(wine_2d[441, 0], wine_2d[441, 1], str('*'),fontdict={'weight': 'bold', 'size': 22})
#ax.text(wine_2d[471, 0], wine_2d[471, 1], str('*'),fontdict={'weight': 'bold', 'size': 22})


#ax.text(wine_2d[315, 0], wine_2d[315, 1], str('.'),fontdict={'weight': 'bold', 'size': 22})

#plt.plot([0.5,0.5],[1,0])
#ax.axis([0, 1, 0, 1])
#plt.plot([0.34,0.38],[0,1],'k')
#plt.plot([0.34,0.63],[0,0.1],'k')
#plt.plot([0.83,0.63],[1,0.1],'k')

#plt.plot([0.42,0.455],[0,0.6],'k')
#plt.plot([0.455,0.1],[0.6,1],'k')
#
#plt.plot([0.77,0.61],[1,0.61],'k')
#plt.plot([0.61,0.65],[0.61,0.35],'k')
#plt.plot([0.65,1],[0.35,0.7],'k')





