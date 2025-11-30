import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform, cdist

# D: Dimension
# N: population of wolves
# lb,ub lowerbound and upper bound
def initial(N,D,lb,ub):
  # 2 cases sames bounds for all dimensions or not
  # isinstance(UB,(list,np.ndarray)) is checking if UB is neither list or tuple
  # if SS_Boundary = 1 -> random 2D array from (0:1] PopxD
  SS_Boundary = len(lb) if isinstance(ub,(list,np.ndarray)) else 1
  if SS_Boundary == 1:
    Positions = np.random.rand(N,D)*(ub - lb) + lb
  else:
    Positions = np.zeros((N,D))
    for i in range (D):
        Positions[:,i] = np.random.rand(N)*(ub[i]-lb[i])+lb[i]
  return Positions

def bound_constraint(newpos, oldpos, lu):
    # lu[0, :] is lower bound, lu[1, :] is upper bound
    newpos = np.clip(newpos, lu[0, :], lu[1, :])
    return newpos

def IGWO(D, N, Max_iter, lb, ub, fobj):
    # boundary matrix
    if isinstance(lb, (int, float)):
        lu = np.array([np.ones(D) * lb, np.ones(D) * ub])
    else:
        lu = np.array([lb, ub])
    
    # Initialize alpha, beta, and delta positions
    AlphaPos = BetaPos = DeltaPos = np.zeros(D)
    AlphaFit = BetaFit = DeltaFit = np.inf  
    
    # Initialize the positions of wolves
    Positions = initial(N, D, ub, lb)
    Positions = bound_constraint(Positions, Positions, lu)
    
    # Calculate fitness for all woves
    Fit = np.zeros(N)
    for i in range(N):
        Fit[i] = fobj(Positions[i, :])
    
    # best fitness and position for each wolf
    pBestScore = Fit.copy()
    pBest = Positions.copy()
    
    # Initialize neighbor matrix
    neighbor = np.zeros((N, N))
    
    Convergence_curve = np.zeros(Max_iter)
    # Main loop
    iter = 0
    while iter < Max_iter:
        
        # Update Alpha, Beta, and Delta
        for i in range(N):
            fitness = Fit[i]
            
            if fitness < AlphaFit:
                AlphaFit = fitness
                AlphaPos = Positions[i, :].copy()
            elif fitness<BetaFit:
                BetaFit = fitness
                BetaPos = Positions[i, :].copy()
            elif fitness<DeltaFit:
                DeltaFit = fitness
                DeltaPos = Positions[i, :].copy()
        
        # XI - GWO
        a = 2 - iter * (2 / Max_iter)
        X_GWO = np.zeros((N, D))
        Fit_GWO = np.zeros(N)
        
        #Position update
        for i in range(N):
            for j in range(D):
                
                #Wolf Alpha
                r1 = np.random.rand()
                r2 = np.random.rand()
                A1 = 2*a*r1 - a
                C1 = 2*r2
                D_Alpha = abs(C1*AlphaPos[j]-Positions[i,j])
                X1 = AlphaPos[j] - A1*D_Alpha
                
                # Wolf Beta
                r1 = np.random.rand()
                r2 = np.random.rand()
                A2 = 2*a*r1-a
                C2 = 2*r2
                D_Beta = abs(C2*BetaPos[j]-Positions[i,j])
                X2 = BetaPos[j] - A2*D_Beta
                
                # Wolf Delta
                r1=np.random.random()
                r2=np.random.random()
                A3 = 2*a*r1-a
                C3 = 2*r2
                D_Delta = abs(C3*DeltaPos[j]-Positions[i,j])
                X3 = DeltaPos[j] - A3*D_Delta
              
                # X GWO
                X_GWO[i, j] = (X1 + X2 + X3)/3
            
            # boundary constraint
            X_GWO[i, :] = bound_constraint(X_GWO[i, :], Positions[i, :], lu)
            Fit_GWO[i] = fobj(X_GWO[i, :])
        
        # Xi-DLH 
        R = cdist(Positions, X_GWO, 'euclidean').diagonal()
        
        # Calculate distance
        dist_Position = squareform(pdist(Positions, 'euclidean'))
        
        # Random one wolf in population
        r1 = np.random.permutation(N)
        
        # Initialize X_DLH and Fit_DLH
        X_DLH = np.zeros((N, D))
        Fit_DLH = np.zeros(N)
        
        for t in range(N):
            # find neighbors
            neighbor[t, :] = (dist_Position[t, :] <= R[t])
            Id = np.where(neighbor[t, :] == 1)[0]
            
            # pick 1 from neighbors
            random_Id = np.random.randint(0, len(Id), size=D)
            
            # DLH
            for d in range(D):
                X_DLH[t, d] = Positions[t, d] + np.random.rand()*(Positions[Id[random_Id[d]], d] - Positions[r1[t], d])
            
            # boundary 
            X_DLH[t, :] = bound_constraint(X_DLH[t, :], Positions[t, :], lu)
            Fit_DLH[t] = fobj(X_DLH[t, :])
        
        # Selection between X gwo or X DLH
        tmp = Fit_GWO < Fit_DLH
        tmpFit = tmp * Fit_GWO + (~tmp) * Fit_DLH
        tmpPositions = np.where(tmp[:, None], X_GWO, X_DLH)
        
        # Update
        tmp = tmpFit < pBestScore 
        pBestScore = tmp * tmpFit + (~tmp) * pBestScore
        pBest = np.where(tmp[:, None], tmpPositions, pBest)
        # Update current fitness, positions
        Fit = pBestScore.copy()
        Positions = pBest.copy()
        
        #for i in range(N):
        #    if Fit[i] < AlphaFit:
        #        AlphaFit = Fit[i]
        #        AlphaPos = Positions[i, :].copy()
        #    elif Fit[i] < BetaFit:
        #        BetaFit = Fit[i]
        #        BetaPos = Positions[i, :].copy()
        #    elif Fit[i] < DeltaFit:
        #        DeltaFit = Fit[i]
        #        DeltaPos = Positions[i, :].copy()
        # Store convergence
        iter += 1
        Convergence_curve[iter - 1] = AlphaFit
            
    return AlphaFit, AlphaPos, Convergence_curve

# Define Objective Function
if __name__ == "__main__":
    def F1(x):
        return np.sum(x**2)
    Demo_Fun = F1
    lb = -100
    ub = 100
    D = 30
    N = 100
    Max_iter = 100

    bestfit, bestpos, convergence_curve = IGWO(D,N,Max_iter,lb,ub,Demo_Fun)
    print("Best fitness = ",bestfit)
    print("Best position =",bestpos)
    #Plot
    plt.plot(convergence_curve)
    plt.xlabel("Iterations")
    plt.ylabel("Fitness")
    plt.title("IGWO Convergence Curve")
    plt.grid(True)
    plt.show()