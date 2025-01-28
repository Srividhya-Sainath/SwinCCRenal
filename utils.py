import os,sys,h5py,yaml
from PIL import Image

import numpy as np

def count_trainable_params(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable_params

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def LoadHdf5Data(hdf5Filename):    
  hdf5Data={}    
  with h5py.File(hdf5Filename, 'r') as f:
    hdf5Data['classLabels']=np.array(f['classes'][:],'U')
    hdf5Data['patchCenters']=f['patchCenters'][:]
    for k in f['patches'].attrs.keys():
      hdf5Data[k]=f['patches'].attrs[k]
    hdf5Data['patchData']=[]
    for s in np.arange(hdf5Data['numberOfScales']):
      temp={}
      temp['patches']=f['patches'+'/'+str(s)][:]
      temp['patchSize']=f['patches'+'/'+str(s)].attrs['patchSize']
      temp['downSampleFactor']=f['patches'+'/'+str(s)].attrs['downSampleFactor']
      hdf5Data['patchData'].append(temp)
  return hdf5Data    

def LoadPatchData(hdf5FileList,classDict=None,returnSampleNumbers=False,returnPatchCenters=False):
    if len(hdf5FileList) == 0:
      print("Empty hdf5 list supplied. Exiting... \n")
      os._exit(1)
    # Find number of scales/patch dimensions
    numberOfPatches=0

    if classDict is None:
      uniqueLabels=set()
    else:
      uniqueLabels=np.array([k for k in classDict.keys()])
    
    patchSizes=[]
    for fileCounter,hdf5File in enumerate(hdf5FileList):
      with h5py.File(hdf5File, 'r') as f:
        if fileCounter==0:
          numberOfScales=f['patches'].attrs['numberOfScales']
          for s in np.arange(numberOfScales): #numberofScales yahaan 2 hai. Remember ek 5x aur ek 20x 
            patchSizes.append(f['patches'+'/'+str(s)].attrs['patchSize']) # aur har scale mein patch size jo hai
              
        classes=np.array(f['classes'][:],'U')
        if classDict is None:
          numberOfPatches=numberOfPatches+f['patches'].attrs['numberOfPatches']
          uniqueLabels.update(set(classes))
        else:
          isInDict= np.array([s in uniqueLabels for s in classes])
          numberOfPatches=numberOfPatches+np.sum(isInDict)
       
    if classDict is None:
      classDict={}
      for num,name in enumerate(np.sort(list(uniqueLabels))):
        classDict[name]=num

    # Change variable type to int
    numberOfPatches = int(numberOfPatches)
    
    #print(np.array(uniqueLabels))     
    patchData=[]
    #preAllocate patches across scales and classes 
    patchClasses=np.zeros((numberOfPatches))
    sampleNumbers=np.zeros((numberOfPatches))
    patchCenters=np.zeros((numberOfPatches,2))

    for scale in range(numberOfScales):
      patchData.append(np.zeros((numberOfPatches,patchSizes[scale],patchSizes[scale],3),dtype=np.uint8))
    #print('Initialized data in '+ str(time.time() - startTime)+' seconds')
    patchCounter=0  
    # bar=progressbar.ProgressBar(maxval=len(hdf5FileList))
    # bar.start()
    for fileCounter,hdf5File in enumerate(hdf5FileList):
      
      hdf5Data=LoadHdf5Data(hdf5File)
      isInDict= np.array([s in uniqueLabels for s in hdf5Data['classLabels']])
      nValidPatches=np.sum(isInDict)
      if(nValidPatches>0):
      #print(nValidPatches)
          try:
              patchClasses[np.uint32(np.arange(patchCounter,patchCounter+nValidPatches))]=np.array([classDict[c] for c in hdf5Data['classLabels'][isInDict]])
          except Exception as e:
              print(nValidPatches)
              print(np.arange(patchCounter,patchCounter+nValidPatches))
              print(e)
              sys.exit()
          sampleNumbers[np.uint32(np.arange(patchCounter,patchCounter+nValidPatches))]=fileCounter
          patchCenters[np.uint32(np.arange(patchCounter,patchCounter+nValidPatches)),:]=hdf5Data['patchCenters'][isInDict]
          for scale in range(numberOfScales):
            patchData[scale][np.uint32(np.arange(patchCounter,patchCounter+nValidPatches)),:,:,:]=hdf5Data['patchData'][scale]['patches'][isInDict]
            #print(hdf5Data['patchData'][scale]['patches'][isInDict])
      patchCounter=patchCounter+nValidPatches  
    #   bar.update(fileCounter)
    # bar.finish()   
    if returnPatchCenters:
      return patchData, patchClasses,classDict,sampleNumbers,patchCenters
    elif returnSampleNumbers:
      return patchData, patchClasses,classDict,sampleNumbers
    else:
      return patchData, patchClasses,classDict