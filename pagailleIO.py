
import fabio
import fabio.edfimage as edf
import fabio.tifimage as tif
#import edfimage

#from PIL import Image
import numpy as np
import sys
import os
import configparser as ConfigParser


def openImage(filename):
    filename=str(filename)
    im=fabio.open(filename)
    imarray=im.data
    return imarray

def getHeader(filename):
    im = fabio.open(filename)
    header= im.header
    return header



def saveTiff16bit(data,filename,minIm=0,maxIm=0,header=None):
    if(minIm==maxIm):
        minIm=np.amin(data)
        maxIm= np.amax(data)
    datatoStore=65536*(data-minIm)/(maxIm-minIm)
    datatoStore[datatoStore>65635]=65535
    datatoStore[datatoStore <0] = 0
    datatoStore=np.asarray(datatoStore,np.uint16)

    if(header!=None):
        tif.TifImage(data=datatoStore,header=header).write(filename)
    else:
        tif.TifImage(data=datatoStore).write(filename)





def openSeq(filenames):
    if len(filenames) >0 :
        data=openImage(str(filenames[0]))
        height,width=data.shape
        toReturn = np.zeros((len(filenames), height, width),dtype=np.float32)
        i=0
        for file in filenames:
            data=openImage(str(file))
            toReturn[i,:,:]=data
            i+=1
        return toReturn
    raise Exception('spytlabIOError')

# ****************************************************************************
# ********************** Interpreting the ini file
# ****************************************************************************

def replace_default(key, value, data_type, use_default, default_val, condition, reason):
    """Function that deals with replacing inputed values with default values
   when this is needed (and possible)
   """
    
    value_out = value
    
    if (use_default):
        if (default_val):
            type_warn = (str(reason) + ' Parameter value for key ' + str(key) + ' of data type ' + str(data_type)
                         + ' has been replaced with default value ' + str(default_val) + ' . Initially inputed value: '
                         + str(value))
            test_warn(condition, 'check_key', type_warn)
        value_out = default_val
    else:
        type_err = (str(reason) + ' Parameter value for key ' + str(key) + ' of data type ' + str(data_type)
                    + ' is NEEDED and there is no default value for this.' + ' Initially inputed value: ' + str(value))
        test_err(condition, 'check_key', type_err)
        
    return value_out


def test_warn(assertion, func='function', message='False assertion', verbose=True):
    """Tests if assertion can be interpreted as False. If this is the case, print a warning message, but continue the
    execution of the program. Returns True if problem found. This can be used in an if clause to perform additional
    actions to compensate for the problem. (e.g. asign a default value to some variable).
    """
    
    result = not bool(assertion)

    if (result and verbose):
        print('Function ' + func + ': ' + message)

    return result


def check_key(key, config, section, data_type='str', default_val=None, use_default=True, positive=False, acc_val=None):
    """Check if key exists in dictionary. If it does not exist and no default value given, use error function.
    Otherwise print warning and use default. This function will change the input dictionary when needed, as it is
    a mutable object. If positive is True for a int/float, the code will check if the number is positive.
    If positive is True for a path, then the code will check if the path points to a true location, otherwise it will
    replace it with None
    """

    # Check if key exists in dictionary. If not, replace with default
    # if possible or call error
    
    reason = 'Key does not exist in INI file.'
    condition = True if key in config[section] else False
    value = None
    
    if(condition):
        value = config.get(section, key)
    else:
        replace_default(key, value, data_type, use_default, default_val, condition, reason)

    # Remove leading and final quotes/apostrophes in case they were
    # introduced by the user
    try:
      basestring
    except NameError:
      basestring = str
    
    if(isinstance(value, str) or isinstance(value, basestring)):
        if value.startswith('\"') and value.endswith('\"'):
            value = value[1:-1]
            
        if value.startswith('\'') and value.endswith('\''):
             value = value[1:-1]
             
    # Check if inputed value is None or 'none'
    if (value == 'none'):
        value = None
    
    # Force correct type. If it cannot be done, replace with default
    # if possible or call error            
    try:
        if(value):
            if (data_type == 'float'):
                value = float(value)
            elif (data_type == 'int'):
                value = int(value)
            elif (data_type == 'bool'):
                if (str(config.get(section, key)).lower() == 'true'):
                    value = True
                elif (str(value).lower() == 'false' or int(value) == 0):
                    value = False
                else:
                    value = bool(value)
                
               
            elif (data_type == 'grid'):
                roi_list = value.split()
                roi_err1 = ('The string for ' + key + ' must be \'none\' or must contain 4 positive integers separated '
                            'by a  space using this format \'xmin xmax ymin ymax\'')
                roi_err2 = ('At least one of the 4 values in the inputed  string for ' + key + ' cannot be interpreted '
                            'as INTEGER')
                roi_err3 = ('The 4 values in the inputed string for ' + key + ' must all be POSITIVE INTEGERS')
                roi_err4 = 'In ' + key + ', check if xmin < xmax'
                roi_err5 = 'In ' + key + ', check if ymin < ymax'
                # roi_err6 = 'In ' + key + ', check if ymin < ymax'

                test_err(len(roi_list) == 4, 'check_key', roi_err1)
                
                try: 
                    roi_list = [int(i) for i in roi_list]
                    value = roi_list
                except:
                    test_err(False, 'check_key', roi_err2)
                    
                test_err(any(item >= 0 for item in roi_list), 'check_key', roi_err3)
                test_err(roi_list[0] < roi_list[1], 'check_key', roi_err4)
                test_err(roi_list[2] < roi_list[3], 'check_key', roi_err5)
                
            elif (data_type == 'path'):
                value = os.path.abspath(value)
                if (not os.path.exists(value)):
                    if(positive):
                        warn = ('Path in key ' + str(key) + ' with value: ' + str(value) + ' does not exist on disk. '
                                + ' Will try to create it.')
                        test_warn(False, 'check_input', warn)
                        #value = None
                    #else:
                        # Create path, check write permissions
                    #    try:
                    #        os.makedirs(value)
                    #    except:
                    #        write_err1 = ('Cannot create ' + value + '. Check write permissions')
                    #        test_err(False, 'check_input', write_err1)                     
        else:
            if (use_default):
                value = default_val
                exist_warn = ('Key ' + str(key) + ' does not exist or has ' + 'value None in INI file. It will created '
                              + 'using the default value ' + str(default_val))
                if (test_warn(key in config.options(section), 'check_key', exist_warn)):
                    config.set(section, key, default_val)
            else:
                exist_err = ('Key ' + str(key) + ' does not exist or has ' + 'value None in INI file. '
                              + 'It MUST be included with a valid value. ')
                test_err(key in config.options(section), 'check_key', exist_err)
    except (ValueError, TypeError):
        reason = 'Input value cannot be parsed into correct data type.'
        condition = False
        replace_default(key, value, data_type, use_default, default_val, condition, reason)
            
    # Check if value is within accepted set of values
    if (acc_val):
        acc_list = acc_val.split()
        reason = 'Input value not in accepted list of values for key.'
        condition = bool(str(value) in acc_list)
        replace_default(key, value, data_type, use_default, default_val, condition, reason)
            
    # If int or float, test whether it is a positive value (if required)
    if(value and data_type in ['int', 'float'] and positive):
        pos_err = ('Key ' + str(key) + ' with input value ' + str(config.get(section, key)) + ' must be a POSITIVE ' 
                     'number of type ' + data_type)
        test_err(value >= 0, 'check_key', pos_err)
    
    return value



def check_input(sCase,path_ini):
    """Function to load, check and correct if possible the contents of the INI file for the detectorDistortion script.
    When possible, it will use default values if input ones are absent. It will also ensure that directory paths are
    int the absolute format and create output directories when and if needed. It can call warnings or errors, depending
    on how severe the issue is. Returns a dictionary to be used in the rest of the script
    """
    
    # Create and load config object
    config = ConfigParser.ConfigParser()
    config.read(path_ini)

    error_msg = ('The provided INI file does not contain the ' + str(sCase) + ' tomoCase. All relevant user parameters'
                 ' must be present in the INI file under that section/case.')
    test_err(sCase in config.sections(), 'check_input', error_msg)
    
    section_out = sCase + '_checked'

    
    # Prepare list of [key, data_type, default_val, use_default, positive, acc_value]
    key_list = [['nbImages',        'int',     1000,      True,   True,   None],
                ['gridDigit',       'int',     3,         True,   True,   None],
                ['gridImages',      'grid',    None,      False,  True,   None],
                ['middlePartStr',   'str',     '__',      True,   True,   None],
                ['darkName',        'str',    'dark',     True,   True,   None],
                ['refstartN',       'str',  'refHST0000', True,   True,   None],
                ['refstopN',        'str',  'refHST2800', True,   True,   None],
                ['machinePrefix',   'str', 'ESRFcluster', True,   True,   None],
                ['MISTII_1',        'bool',    True,      True,   True,   None],
                ['MISTII_2',        'bool',    True,      False,  True,   None],
        		 ['computeEllipse',  'bool',    True,      False,  True,   None],
                ['expFolder',       'path',    None,      False,  True,   None],
                ['outputFolder',    'path',    None,      False,  True,   None],
                ['FolderBasis',     'str',     None,      True,   True,   None],
                ['cropOn',          'bool',    True,      True,   True,   None],
                ['cropDebX',        'int',     0,         True,   True,   None],
                ['cropDebY',        'int',     0,         True,   True,   None],
                ['cropEndX',        'int',     1850,      True,   True,   None],
                ['cropEndY',        'int',     2048,      True,   True,   None],
                ['NeighboursPix',   'int',     1,         True,   True,   None],
                ['pixel',           'float',   5.8E-6,    True,   True,   None],
                ['nbProj',          'int',     3000,      True,   True,   None],
                ['distOD',          'int',     1,         True,   True,   None],
                ['distSO',          'int',     140,       True,   True,   None],
                ['energy',          'float',   30,        True,   True,   None],
                ['delta',           'float',   8E-7,      True,   True,   None],
                ['beta',            'float',   7E-10,     True,   True,   None],
                ['sourceSize',      'float',   1E-6,      True,   True,   None],
                ['detectorPSF',     'float',   1.5,       True,   True,   None],
                ['padding',         'int',     0,         True,   True,   None],
                ['padType',         'str',    'reflect',  True,   True,   None],
                ['Deconvolution',   'bool',    False,     True,   True,   None],
                ['umpaMaxShift',    'int',     4,         True,   True,   None],
                ['LCS_med_filter'  ,'int',     0,         True,   True,   None],
                ['LCS_gauss_filter','int',     0,         True,   True,   None],
                ['DeconvType',      'str', 'unsupervised',True,   True,   None],
                ['DirDF_MedFilt',   'int',     1,         True,   True,   None],
                ['timePavlovDirDF', 'int',     0,         True,   True,   None],
                ['timeLCSv2',       'int',     0,         True,   True,   None],
                ['sigmaRegularization','int',  0,         True,   True,   None],
                ]
                    

    # Initialize parameter dictionary
    param = {}
    
    # Add new section to config object
    config.add_section(section_out)
    
    # Check parameters related to files and directories
    for [key, data_type, default_value, use_default, positive, acc_value] in key_list:
        # SHOULD BE IMPLEMENTED FOR ROBUSTNESS
        value = check_key(key, config, sCase, data_type, default_value, use_default, positive, acc_value)
                          
        # config.set(section_out, key, str(value))
        
        param[key] = value
        
        # For the ROI_default parameter, convert string to array of integers
        # Already done in check_key
        if((data_type == 'grid') and value):
            param[key] = np.array(value)
            
                                      
    # Print a modified INI file in output directory (with new section)
    #cfg_file = open(os.path.join(param['dir_out'], 'input_modified_' + dt + '.ini'), 'w')
    #config.write(cfg_file)
    #cfg_file.close()
    
    # Return dictionary containing needed parameters and their values
    return param
    
    
    
def test_err(assertion, func='function', message='False assertion', verbose = True):
    """Tests if assertion can be interpreted as False. If this is the  case, print an error message and stop the
    execution of the program.
    """

    result = not bool(assertion)

    if (result):
        print('Function ' + func + ': ' + message)
        sys.exit(1)

    return result



# ****************************************************************************
# ********************** Motors
# ****************************************************************************


def makeDarkMean(Darkfiedls):
    nbslices, height, width = Darkfiedls.shape
    meanSlice = np.mean(Darkfiedls, axis=0)
    print ('-----------------------  mean Dark calculation done ------------------------- ')
    OutputFileName = '/Users/helene/PycharmProjects/spytlab/meanDarkTest.edf'
    outputEdf = edf.EdfFile(OutputFileName, access='wb+')
    outputEdf.WriteImage({}, meanSlice)
    return meanSlice


def saveEdf(data,filename):
    print(filename)
    dataToStore=data.astype(np.float32)
    edf.EdfImage(data=dataToStore).write(filename)


def save3D_Edf(data,filename):
    nbslices,height,width=data.shape
    for i in range(nbslices):
        textSlice='%4.4d'%i
        dataToSave=data[i,:,:]
        filenameSlice=filename+textSlice+'.edf'
        saveEdf(dataToSave,filenameSlice)



#def savePNG(data,filename,min=0,max=0):
    #if min == max:
    #    min=np.amin(data)
    #    max= np.amax(data)
    #data16bit=data-min/(max-min)
    #data16bit=np.asarray(data16bit,dtype=np.uint16)

    #scipy.misc.imsave(filename,data16bit)




if __name__ == "__main__":

    # filename='ref1-1.edf'
    # filenames=glob.glob('*.edf')
    # data=openImage(filename)
    # savePNG(data,'ref.png',100,450)
    # print( data.shape)
    #
    #
    # rootfolder = '/Volumes/VISITOR/md1097/id17/Phantoms/TwoDimensionalPhantom/GrilleFils/Absorption52keV/'
    # referencesFilenames = glob.glob(rootfolder + 'Projref/*.edf')
    # sampleFilenames = glob.glob(rootfolder + 'Proj/*.edf')
    # referencesFilenames.sort()
    # sampleFilenames.sort()
    # print(' lalalal ')
    # print (referencesFilenames)
    # print (sampleFilenames)

    inputImageFilename = '/Volumes/ID17/speckle/md1097/id17/Phantoms/ThreeDimensionalPhantom/OpticalFlow/dx32/dx_Speckle_Foam1_52keV_6um_xss_bis_012_0000.edf'
    data=openImage(inputImageFilename)
    print(data.dtype)
    print(data)
    outputImageFilename = '/Volumes/ID17/speckle/md1097/id17/Phantoms/ThreeDimensionalPhantom/OpticalFlowTest26Apr/dx0001_32bit.edf'
    saveEdf(data,outputImageFilename)
    print(data)
    print('At the end '+str(data.dtype))

