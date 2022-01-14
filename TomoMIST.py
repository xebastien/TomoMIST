#! /usr/bin/env python3

# to submit to oar oarsub --array 5  -S "./test.oar toto"

from pagailleIO import saveEdf, openImage, test_err,check_input#, openSeq, save3D_Edf
#import glob
#import random
import os
from scipy.ndimage.filters import gaussian_filter
from MISTII_1 import processProjectionMISTII_1
from MISTII_2 import processProjectionMISTII_2
#import time
import datetime
from matplotlib import pyplot as plt
import numpy as np
import sys


def preProcessAndPadImages(Is, Ir, expDict):
    """
    Simply pads images in Is and Ir using parameters in expDict
    Returns Is and Ir padded
    Will eventually do more (Deconvolution, shot noise filtering...)
    """
    nbImages, width, height = Ir.shape
    padSize=expDict['padding']
    IrToReturn = np.zeros((nbImages, width + 2 * padSize, height + 2 * padSize))
    IsToReturn = np.zeros((nbImages, width + 2 * padSize, height + 2 * padSize))
    for i in range(nbImages):
        IrToReturn[i] = np.pad(Ir[i], ((padSize, padSize), (padSize, padSize)),mode=expDict['padType'])  # voir is edge mieux que reflect
        IsToReturn[i] = np.pad(Is[i], ((padSize, padSize), (padSize, padSize)),mode=expDict['padType'])  # voir is edge mieux que reflect
    return IsToReturn, IrToReturn
    
  
def readStudiedCase(sCase, expParam ,imNumber):
    """
    This function contains and reads the data specific to an experiment,
    opens the acquisitions and normalizes
    Arguments: 
        sCase [string]: the experiment name
        nbImages [int]: the number of pairs of acquisitions to take into account
        machinePrefix [string]: the name of the machine you are working on
    Outputs:
        Is [numpy array]: contains the nbImages sample images
        Ir [numpy array]: contains the nbImages reference images
    """
    
    gridImages = expParam["gridImages"].tolist()

    refName = []
    sampName = []
    
    refPath0 = expParam['expFolder'] + '/'+ expParam['FolderBasis']+ str(gridImages[0]).zfill(expParam['gridDigit']) + expParam['middlePartStr'] + str(gridImages[2]).zfill(expParam['gridDigit']) + '_/'
    refName0 = refPath0 + expParam['refstartN'] + '.edf'

    # catch image size from first image
    data = openImage(refName0)
    height,width = data.shape
    
    NbIm = len(range(gridImages[0],gridImages[1]+1)) * len(range(gridImages[2],gridImages[3]+1))
    # malloc 3D array
    Is = np.zeros((NbIm, height, width),dtype=np.float32)   
    Ir = np.zeros((NbIm, height, width),dtype=np.float32)
    # open and store image
    itCounter = 0    
    for ks in range(gridImages[0],gridImages[1]+1):       # slow axis
        for kf in range(gridImages[2],gridImages[3]+1):     # fast axis
            refPath0 = expParam['expFolder'] + '/'+ expParam['FolderBasis'] + str(ks).zfill(expParam['gridDigit']) + expParam['middlePartStr'] + str(kf).zfill(expParam['gridDigit']) +'_/'
            if imNumber < (expParam['nbProj']/2):
                refName1 = refPath0 + expParam['refstartN'] + '.edf'
            else:
                refName1 = refPath0 + expParam['refstopN'] + '.edf'
            
            darkName =  refPath0 + expParam['darkName'] + '.edf'
            darkImage = openImage(str(darkName))
            
            refName.append(refName1)  
            dataRefIm = openImage(str(refName1))
            Ir[itCounter,:,:] = dataRefIm - darkImage
            
            sampName1 = refPath0 + expParam['FolderBasis'] + str(ks).zfill(expParam['gridDigit']) + expParam['middlePartStr'] + str(kf).zfill(expParam['gridDigit']) + '_' + str(imNumber).zfill(4) + '.edf'
            
            sampName.append(sampName1)  
            dataSampIm = openImage(str(sampName1))
            Is[itCounter,:,:] = dataSampIm - darkImage
            
            itCounter += 1


        
    # On cree un white a partir de la reference pour normaliser 
        # surtout le float64 tres important ici
    white = gaussian_filter(np.mean(Ir, axis=0),30)
    Ir = np.asarray(Ir/white, dtype=np.float64)
    Is = np.asarray(Is/white, dtype=np.float64)
    
    #Ir = np.subtract(Ir[0:-1,:,:],Ir[1:,:,:],dtype=np.float64)
    # Is = np.subtract(Is[0:-1,:,:],Is[1:,:,:],dtype=np.float64)
    # stolen from Ruxandra Swarp: one pixel is concatenaton of 4 neibogring
    if expParam['NeighboursPix'] > 1:
        # To have 1x1, 2x2, 3x3 windows: template_window is NOT a half-size!
        dw = int(Is.shape[2] - expParam['NeighboursPix']+1)
        dh = int(Is.shape[1] - expParam['NeighboursPix']+1)
        #dd = Is.shape[0]
        sample_big = np.empty((0, dh, dw))
        ref_big = np.empty((0, dh, dw))
        
        for kw in range(0, expParam['NeighboursPix']):
            for kh in range(0, expParam['NeighboursPix']):
                ii = kh
                ie = - expParam['NeighboursPix'] + 1 + kh
                if (ie == 0):
                    ie = None
                ji = kw
                je = - expParam['NeighboursPix'] + 1 + kw
                if (je == 0):
                    je = None
    
                sample_big = np.concatenate((sample_big, Is[:, ii:ie, ji:je]), axis=0)
                ref_big = np.concatenate((ref_big, Ir[:, ii:ie, ji:je]), axis=0)

        Is = sample_big
        Ir = ref_big

    
    if expParam['cropOn']:
        Ir = Ir[:, expParam['cropDebX']:expParam['cropEndX'], expParam['cropDebY']:expParam['cropEndY']]
        Is = Is[:, expParam['cropDebX']:expParam['cropEndX'], expParam['cropDebY']:expParam['cropEndY']]

    return Is, Ir

def processMISTII_2(sampleImage, referenceImage, ddict,kImage):
    """
    this function calls processProjectionLCS_v2() in its file, 
    crops the results of the padds added in pre-processin 
    and saves the retrieved images.
    """
    result = processProjectionMISTII_2(sampleImage, referenceImage, expParam=ddict)
    thickness = result['phi']
    Deff_xx = result['Deff_xx']
    Deff_yy = result['Deff_yy']
    Deff_xy = result['Deff_xy']
    colouredDeff = result['ColoredDeff']
    NbIm, Nx, Ny = sampleImage.shape
    
    padSize = ddict['padding']
    if padSize > 0:
        width, height = thickness.shape
        thickness = thickness[padSize:padSize + width - 2 * padSize, padSize:padSize + height - 2 * padSize]
        Deff_xx = Deff_xx[padSize:padSize + width - 2 * padSize, padSize:padSize + height - 2 * padSize]
        Deff_yy = Deff_yy[padSize:padSize + width - 2 * padSize, padSize:padSize + height - 2 * padSize]
        Deff_xy = Deff_xy[padSize:padSize + width - 2 * padSize, padSize:padSize + height - 2 * padSize]
    # saveEdf(phi, ddict['outputFolder']     + '/phi_'+ str(kImage).zfill(4) +'.edf')
    saveEdf(Deff_xx, ddict['outputFolder'] + '/Deff_xx_'+ str(kImage).zfill(4) + '.edf')
    saveEdf(Deff_yy, ddict['outputFolder'] + '/Deff_yy_'+ str(kImage).zfill(4) + '.edf')
    saveEdf(Deff_xy, ddict['outputFolder'] + '/Deff_xy_'+ str(kImage).zfill(4) + '.edf')
    plt.imsave(ddict['outputFolder'] + '/ColoredDeff_'+str(kImage).zfill(4)+'.tiff',colouredDeff,format='tiff')
    return 

def processMISTII_1(sampleImage, referenceImage, ddict,kImage):
    """
    this function calls processProjectionLCS_v2() in its file, 
    crops the results of the padds added in pre-processin 
pyder    and saves the retrieved images.
    """
    result = processProjectionMISTII_1(sampleImage, referenceImage, expParam=ddict)
    phi = result['phi']
    Deff_xx = result['Deff_xx']
    Deff_yy = result['Deff_yy']
    Deff_xy = result['Deff_xy']
    colouredDeff = result['ColoredDeff']
    NbIm, Nx, Ny = sampleImage.shape
    
    padSize = ddict['padding']
    if padSize > 0:
        width, height = phi.shape
        phi = phi[padSize:padSize + width - 2 * padSize, padSize:padSize + height - 2 * padSize]
        Deff_xx = Deff_xx[padSize:padSize + width - 2 * padSize, padSize:padSize + height - 2 * padSize]
        Deff_yy = Deff_yy[padSize:padSize + width - 2 * padSize, padSize:padSize + height - 2 * padSize]
        Deff_xy = Deff_xy[padSize:padSize + width - 2 * padSize, padSize:padSize + height - 2 * padSize]
    saveEdf(phi, ddict['outputFolder']     + '/phi_'+str(kImage).zfill(4) +'.edf')
    saveEdf(Deff_xx, ddict['outputFolder'] + '/Deff_xx_'+ str(kImage).zfill(4) + '.edf')
    saveEdf(Deff_yy, ddict['outputFolder'] + '/Deff_yy_'+ str(kImage).zfill(4) + '.edf')
    saveEdf(Deff_xy, ddict['outputFolder'] + '/Deff_xy_'+ str(kImage).zfill(4) + '.edf')
    plt.imsave(ddict['outputFolder'] + '/ColoredDeff_'+str(kImage).zfill(4)+'.tiff',colouredDeff,format='tiff')
    return 


if __name__ == "__main__":
    
    #######not anymore# I put here in hard for now
    path_ini = sys.argv[1]#"/data/bm05/bm05staff/seb/library/laurene/CodeTomo/Tomo_MIST.ini" 
    #Parameters to tune
    print("Th .ini file read will be:")
    print(path_ini)
    print("Image numbers are from {} to {} included".format(int(sys.argv[3]),int(sys.argv[4])))

    #firstImNum = int(sys.argv[1])
    #lastImNum = int(sys.argv[2])
    imNumbers = np.arange(int(sys.argv[3]),int(sys.argv[4])+1)

    #expDict["gridImages"] =  [0,3,1,4]       #slowAxis [Init, SlowAxisEnd], [FastAxisInit, FastAxisend]=> range by range should give you the number of nbImages
    #expDict["nbImages"] = 500                # number of maximum pair images
    studiedCase = sys.argv[2] #'LadafKidney1'  # name of the experiment we want to work on
    #expDict["machinePrefix"] ='ESRFcluster'  #name of the machine we are working on
    #expDict["doPavlovDirDF"]  = True
    #expDict["doPavlovDirDFa"] = False	# True
    
    expDict = check_input(studiedCase,path_ini)    
      
    nbImages      = expDict["nbImages"]
    doMISTII_1    = expDict["MISTII_1"]
    doMISTII_2    = expDict["MISTII_2"]
    
    if expDict['outputFolder'][-1] == '/':    
        expDict['outputFolder'] = expDict['outputFolder'][0:-1]
        
    if not os.path.exists(expDict['outputFolder']):
        locFolder = os.path.basename(expDict['outputFolder'])
        baseFolder = os.path.dirname(expDict['outputFolder'])
        
        nobackupFolder = baseFolder + '/' + '.' + locFolder + '_nobackup'
        os.mkdir(nobackupFolder) # test for existence   
        os.symlink(nobackupFolder, expDict['outputFolder'])
        #os.mkdir(expParam['outputFolder']) # test for existence 
        
    # Create write permissions for output directory
    write_err2 = ('Output directory ' + expDict['outputFolder'] + 'does not have write permissions')
    test_err(os.access(expDict['outputFolder'], os.W_OK), 'check_input', write_err2)
    
    for kImage in imNumbers:
    #Load images and parameters:
        Is, Ir = readStudiedCase(studiedCase, expDict, kImage)
        
        Nz = len(Is)
        print("Calculating image   {} using stack of size {} / {} acqui".format(str(kImage).zfill(4),str(Nz),str((Nz//expDict["NeighboursPix"]/expDict["NeighboursPix"]))))
        nbOfPoint           = min(nbImages, Nz)

        Is, Ir = preProcessAndPadImages(Is, Ir, expDict)
        
        ## PHASE RETRIEVAL
        #Compute directional dark field /!\ requires at least 4 points
        if doMISTII_1:
            if nbOfPoint < 4 :
                raise Exception('Not enough points to compute directional dark field. Required at least 4. Given ', expDict['nbOfPoint']) 
            else:
                processMISTII_1(Is, Ir, expDict,kImage)
        if doMISTII_2:
            if nbOfPoint <4 :
                raise Exception('Not enough points to compute directional dark field. Required at least 4. Given ', expDict['nbOfPoint']) 
            else:
                processMISTII_2(Is, Ir, expDict,kImage)
       
        print('Image  {}  successfully processed'.format(str(kImage).zfill(4)))

