import numpy as np
import matplotlib.pyplot as plt

# D: Dimension
# PopulationSize: population of wolves
# LB,UB lowerbound and upper bound
def initial(PopulationSize,D,LB,UB):
  # 2 cases sames bounds for all dimensions or not
  # isinstance(UB,(list,np.ndarray)) is checking if UB is neither list or tuple
  # if SS_Boundary = 1 -> random 2D array from (0:1] PopxD
  SS_Boundary = len(LB) if isinstance(UB,(list,np.ndarray)) else 1
  if SS_Boundary == 1:
    Positions = np.random.rand(PopulationSize,D)*(UB - LB) + LB
  else:
    Positions = np.zeros((PopulationSize,D))
    for i in range (D):
        Positions[:,i] = np.random.rand(PopulationSize)*(UB[i]-LB[i])+LB[i]
  return Positions

from ast import Del


# GWO Main Loop
def GWO(PopulationSize,maxIn,LB,UB,D,ObjF):
  # Initialize alpha, beta, and delta positions
  AlphaPos = BetaPos = DeltaPos = np.zeros(D)
  AlphaFit= BetaFit = DeltaFit = np.inf

  # Initialize the positions of wolves
  Positions = initial(PopulationSize,D,LB,UB)
  ConvergenceCurve = np.zeros(maxIn)

  iteration=0
  while iteration < maxIn:
    for i in range(Positions.shape[0]):
        BB_UB = Positions[i,:]>UB
        BB_LB = Positions[i,:]<LB
        Positions[i,:] = (Positions[i,:]*(~(BB_UB+BB_LB))) + UB*BB_UB + LB*BB_LB
        Fitness = ObjF(Positions[i,:])

        # Update Xa, Xb, Xd
        if Fitness < AlphaFit:
            AlphaFit = Fitness
            AlphaPos = Positions[i,:]
        elif Fitness<BetaFit:
            BetaFit = Fitness
            BetaPos = Positions[i,:]
        elif Fitness<DeltaFit:
            DeltaFit = Fitness
            DeltaPos = Positions[i,:]
    a = 2 - 1*(2/maxIn)
  #Position update
    for i in range (Positions.shape[0]):
      for j in range(Positions.shape[1]):

        #Wolf Alpha
        r1=np.random.random()
        r2=np.random.random()

        A1 = 2*a*r1 - a
        C1 = 2*r2

        D_Alpha = abs(C1*AlphaPos[j]-Positions[i,j])
        X1 = AlphaPos[j] - A1*D_Alpha

        #Wolf Beta
        r1=np.random.random()
        r2=np.random.random()

        A2 = 2*a*r1-a
        C2 = 2*r2

        D_Beta = abs(C2*BetaPos[j]-Positions[i,j])
        X2 = BetaPos[j] - A2*D_Beta

        #Wolf Delta
        r1=np.random.random()
        r2=np.random.random()

        A3 = 2*a*r1-a
        C3 = 2*r2

        D_Delta = abs(C3*DeltaPos[j]-Positions[i,j])
        X3 = DeltaPos[j] - A3*D_Delta

        Positions[i,j] = (X1 + X2 + X3)/3
    iteration += 1
    ConvergenceCurve[iteration-1] = AlphaFit
  return AlphaFit, AlphaPos, ConvergenceCurve

# Define Objective Function
if __name__ == "__main__":
    def F1(x):
        return np.sum(x**2)
    Demo_Fun = F1
    LB = -100
    UB = 100
    D = 30
    PopulationSize = 100
    maxIn = 100

    bestfit, bestpos, convergence_curve = GWO(PopulationSize,maxIn,LB,UB,D,Demo_Fun)
    print("Best fitness = ",bestfit)
    print("Best position =",bestpos)
    #Plot
    plt.plot(convergence_curve)
    plt.xlabel("Iterations")
    plt.ylabel("Fitness")
    plt.title("GWO Convergence Curve")
    plt.grid(True)
    plt.show()