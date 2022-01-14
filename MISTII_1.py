#from pagailleIO import saveEdf,openImage,openSeq,save3D_Edf
from numpy.fft import fftshift as fftshift
from numpy.fft import ifftshift as ifftshift
from numpy.fft import fft2 as fft2
from numpy.fft import ifft2 as ifft2
import numpy as np
#from matplotlib import cm
#import matplotlib as mpl
from scipy.ndimage.filters import  median_filter#gaussian_filter,
#from scipy.sparse.linalg import lsmr
#from matplotlib import pyplot as plt
#import colorsys
from getk import getk
import multiprocessing

def MISTII_1(sampleImages, refImages, dataDict,nbImages):
    """
    Calculates the tensors of the dark field and the thickness of a phase object from the acquisitions
    """
        
    Nz, Nx, Ny  = refImages.shape
    beta        = dataDict["beta"]
    #gamma_mat   = dataDict["delta"]/beta
    distSampDet = dataDict["distOD"]
    pixSize     = dataDict["pixel"]
    #Lambda      = 1.2398/dataDict["energy"]*1e-9
    k           = getk(dataDict["energy"]*1000)
    
    LHS = np.ones(((nbImages, Nx, Ny)))
    RHS = np.ones((((nbImages,4, Nx, Ny))))
    FirstTermRHS = np.ones((Nx,Ny))
    solution = np.ones(((4, Nx, Ny)))
    
    #Prepare system matrices
    for i in range(nbImages):
        #Left hand Side
        IsIr = sampleImages[i]/refImages[i]
        
        #Right handSide
        gX_IrIr,gY_IrIr   = np.gradient(refImages[i],pixSize)
        gXX_IrIr,gYX_IrIr = np.gradient(gX_IrIr,pixSize)
        gXY_IrIr,gYY_IrIr = np.gradient(gY_IrIr,pixSize)
        
        gXX_IrIr = gXX_IrIr/refImages[i]
        gXY_IrIr = gXY_IrIr/refImages[i]
        gYX_IrIr = gYX_IrIr/refImages[i]
        gYY_IrIr = gYY_IrIr/refImages[i]
        
        RHS[i] = [FirstTermRHS,gXX_IrIr, gYY_IrIr,gXY_IrIr]
        LHS[i] = 1 -IsIr


#    Solving system for each pixel 
    #PART PARALLELIZED
    num_cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=num_cores)
    # needed for some platform
    #os.system("taskset -p 0xff %d" % os.getpid())
    mescore = "Using %s cores to solve the pixel systems of equations" % num_cores
    print(mescore)

    # Loop through pixels excluding borders equal to half the (correlation size - 1)
    list_lineData = []

    #boundsM = np.linspace(0, Nx, round(Nx/num_cores))
    for idk in range(0,Nx,round(Nx/num_cores)):
        RHSl = RHS[:,:,idk:idk+round(Nx/num_cores),:]
        LHSl = LHS[:,idk:idk+round(Nx/num_cores),:]
        list_lineData.append([idk,RHSl,LHSl])
	    

    G1 = np.zeros((Nx, Ny))
    G2 = np.zeros((Nx, Ny))
    G3 = np.zeros((Nx, Ny))
    G4 = np.zeros((Nx, Ny))

    print("Starting parralel chunck of the code")
    #chunks_idx_pairs = np.array_split(list(idx_pairs), num_cores)

    solution = pool.map(sysSolvePix, list_lineData)
    # the solution with joblib seems touse only one worker
    #solution = Parallel(n_jobs=18)(delayed(sysSolvePix)(kLineData) for kLineData in list_lineData)

    # closing the pool for the sake of memmory
    pool.close()
    pool.join()
    
    print("Finito parallel chunck of code, will collect solution var and reshape")
    solution = sum(solution, [])

    print("Solution arrays collected")
    for item in solution:
        # G1
        G1[item[4]] = item[0] 
        # G2
        G2[item[4]] = item[1]
        # G3
        G3[item[4]] = item[2]
	# G4
        G4[item[4]] = item[3]        

    
    sig_scale=dataDict['sigmaRegularization']
    if sig_scale==0:
        beta=1
    else:
        dqx = 2 * np.pi / (Nx)
        dqy = 2 * np.pi / (Ny)
        Qx, Qy = np.meshgrid((np.arange(0, Ny) - np.floor(Ny / 2) - 1) * dqy, (np.arange(0, Nx) - np.floor(Nx / 2) - 1) * dqx) #frequency ranges of the images in fqcy space
    
        #building filters
        sigmaX = dqx / 1. * np.power(sig_scale,2)
        sigmaY = dqy / 1. * np.power(sig_scale,2)
        #sigmaX=sig_scale
        #sigmaY = sig_scale
    
        g = np.exp(-(((Qx)**2) / 2. / sigmaX + ((Qy)**2) / 2. / sigmaY))
        #g = np.exp(-(((np.power(Qx, 2)) / 2) / sigmaX + ((np.power(Qy, 2)) / 2) / sigmaY))
        beta = 1 - g;
    
    #Calculation of the thickness of the object
    u, v = np.meshgrid(np.arange(0, Nx), np.arange(0, Ny))
    u = (u - (Nx / 2))
    v = (v - (Ny / 2))
    u_m = u / (Nx * pixSize)
    v_m = v / (Ny * pixSize)
    uv_sqr = np.transpose(u_m ** 2 + v_m ** 2)  # ie (u2+v2)
    uv_sqr[uv_sqr==0] = 1
    
    #Calculation of absorption image
    phi = k/distSampDet*ifft2(ifftshift(fftshift(fft2(G1))*beta/(-4*np.pi*uv_sqr))).real                                                                                                                                                                                                                           

    Deff_xx = -G2/distSampDet
    Deff_yy = -G3/distSampDet
    Deff_xy = -G4/distSampDet
    
    return phi, Deff_xx,Deff_yy,Deff_xy


def sysSolvePix(DataIn):    
    """Function solving the system of equations for the pixel in the line kLine
    """
    kLine = DataIn[0]
    RHS = DataIn[1]
    LHS = DataIn[2]   
    

    list_out = []
    lNz, lNo, lNx, lNy = RHS.shape
    #print("size 1 %s" % lNy)
    
    for j in range(0,lNy):
        for k in range(0,lNx):
            # Define single-pixel system
            a = RHS[:,:,k,j]
            b = LHS[:,k,j]
            Q,R = np.linalg.qr(a) # qr decomposition of A
            if R[2,2]==0 or R[1,1]==0 or R[0,0]==0 or R[3,3]==0:#undetermine system
                solut = [1,1,1,1]
            else:
                Qb  = np.dot(Q.T,b) # computing Q^T*b (project b onto the range of A)
                solut = np.linalg.solve(R,Qb) # solving R*x = Q^T*b

            list_out.append([solut[0], solut[1], solut[2], solut[3], ((k+kLine),j)])
        
    return list_out

def processProjectionMISTII_1(Is,Ir,expParam):
    """
    This function calls PavlovDirDF to compute the tensors of the directional dark field and the phase of the sample
    The function should also convert the tensor into a coloured image
    """
    nbImages, Nx, Ny= Is.shape
    
    #Calculate directional darl field
    phi, Deff_xx,Deff_yy,Deff_xy=MISTII_1(Is,Ir,expParam,nbImages)
    
    #Post processing tests
    #Median filter
    medFiltSize=expParam['DirDF_MedFilt']
    if medFiltSize!=0:
        phi=median_filter(phi, medFiltSize)
        Deff_xx = median_filter(Deff_xx, medFiltSize)
        Deff_yy = median_filter(Deff_yy, medFiltSize)
        Deff_xy = median_filter(Deff_xy, medFiltSize)
    
    #Normalization of the result and restrict to thresholds
    a1 = np.mean(np.mean([Deff_xx,Deff_yy,Deff_xy]))
    b1 = np.mean(np.std([Deff_xx,Deff_yy,Deff_xy]))
    Deff_xx = ((Deff_xx-a1)/(3*b1))
    Deff_yy = ((Deff_yy-a1)/(3*b1))
    Deff_xy = ((Deff_xy-a1)/(3*b1))
    #print(b1)
    Deff_xx[Deff_xx>1] = 1
    Deff_yy[Deff_yy>1] = 1
    Deff_xy[Deff_xy>1] = 1
    Deff_xx[Deff_xx<0] = 0
    Deff_yy[Deff_yy<0] = 0
    Deff_xy[Deff_xy<0] = 0
    
    #Trying to create a coloured image from tensor (method probably wrong for now)
    colouredImage=np.zeros((( Nx, Ny,3)))
    colouredImage[:,:,0] = abs(Deff_xx)
    colouredImage[:,:,1] = abs(Deff_yy)
    colouredImage[:,:,2] = abs(Deff_xy)
    colouredImage[colouredImage>1] = 1
    
    #colouredImage=mpl.colors.hsv_to_rgb(colouredImage)

    return {'phi': phi, 'Deff_xx': Deff_xx, 'Deff_yy': Deff_yy, 'Deff_xy': Deff_xy, 'ColoredDeff': colouredImage}

