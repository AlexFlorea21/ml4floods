import os
import torch
import rasterio
import numpy as np
from ml4floods.models.model_setup import get_channel_configuration_bands
from ml4floods.data.worldfloods.configs import CHANNELS_CONFIGURATIONS, SENTINEL2_NORMALIZATION, CHANNELS_CONFIGURATIONS_LANDSAT
from torchvision.models.feature_extraction import create_feature_extractor
from patchify import patchify, unpatchify
import time
import math
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
from scipy.special import softmax
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, confusion_matrix
import rasterio
import os
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import shutil
import cv2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
from sklearn.preprocessing import Normalizer
from matplotlib.ticker import NullLocator

class xDNNClassifier:
    def train(self, Input):

        Images = Input['Images']
        Features = Input['Features']
        Labels = Input['Labels']
        CN = 3
        Prototypes = self.PrototypesIdentification(Images,Features,Labels,CN)
        Output = {}
        Output['xDNNParms'] = {}
        Output['xDNNParms']['Parameters'] = Prototypes
        MemberLabels = {}
        for i in range(0,CN):
            MemberLabels[i]=Input['Labels'][Input['Labels']==i] 
        Output['xDNNParms']['CurrentNumberofClass']=CN
        Output['xDNNParms']['OriginalNumberofClass']=CN
        Output['xDNNParms']['MemberLabels']=MemberLabels
        return Output

    def updating(self, Input):

        Images = Input['Images']
        Features = Input['Features']
        Labels = Input['Labels']
        CN = 3
        Param = Input['Prototypes']
        Prototypes = self.PrototypesUpdating(Images,Features,Labels,CN,Param)
        Output = {}
        Output['xDNNParms'] = {}
        Output['xDNNParms']['Parameters'] = Prototypes
        MemberLabels = Input['MemberLabels']
        for i in range(0,CN): 
            MemberLabels[i] = np.hstack((MemberLabels[i],Input['Labels'][Input['Labels']==i]))
        Output['xDNNParms']['CurrentNumberofClass']=Input['CurrentNumberofClass']
        Output['xDNNParms']['OriginalNumberofClass']=CN
        Output['xDNNParms']['MemberLabels']=MemberLabels
        return Output
    
    
    def predict(self,Input):
    
        Params=Input['xDNNParms']
        datates=Input['Features']
        Test_Results = self.DecisionMaking(Params,datates)
        EstimatedLabels = Test_Results['EstimatedLabels'] 
        Scores = Test_Results['Scores']
        Output = {}
        Output['EstLabs'] = EstimatedLabels
        # Output['Scores'] = Scores
        # Output['ConfMa'] = confusion_matrix(Input['Labels'],Output['EstLabs'])
        # Output['ClassAcc'] = np.sum(Output['ConfMa']*np.identity(len(Output['ConfMa'])))/len(Input['Labels'])
        return Output

    def PrototypesIdentification(self, Image,GlobalFeature,LABEL,CL):
        data = {}
        image = {}
        label = {}
        Prototypes = {}
        for i in range(0,CL):
            seq = np.argwhere(LABEL==i)
            data[i]=GlobalFeature[seq]
            image[i] = {}
            for j in range(0, len(seq)):
                image[i][j] = Image[seq[j][0]]
            label[i] = np.ones((len(seq),1))*i
        for i in range(0, CL):
            if data[i].size == 0:
                Prototypes[i] = {}
                Prototypes[i]['Noc'] =  0
                Prototypes[i]['Centre'] =  np.empty((0,0))
                Prototypes[i]['Support'] =  np.empty((0,0))
                Prototypes[i]['Radius'] =  np.empty((0,0))
                Prototypes[i]['GMean'] =  np.empty((0,0))
                Prototypes[i]['Prototype'] = {}
                Prototypes[i]['L'] = np.empty((0,0))
                Prototypes[i]['X'] = np.empty((0,0))
                Prototypes[i]['CDmax'] = np.empty((0,0))
                Prototypes[i]['CDmin'] = np.empty((0,0))
                continue
            else:
                Prototypes[i] = self.xDNNclassifier(data[i],image[i])
        
        return Prototypes

    def PrototypesUpdating(self, Image,GlobalFeature,LABEL,CL, Param):
        data = {}
        image = {}
        label = {}
        Prototypes = {}
        for i in range(0,CL):
            seq = np.argwhere(LABEL==i)
            data[i]=GlobalFeature[seq]
            image[i] = {}
            for j in range(0, len(seq)):
                image[i][j] = Image[seq[j][0]]
            label[i] = np.ones((len(seq),1))*i
        for i in range(0, CL):

            if image[i] :
                if Param[i]['Noc'] == 0:

                    Prototypes[i] = self.xDNNclassifier(data[i],image[i])
                else:

                    Prototypes[i] = self.xDNNclassifier_updating(data[i],image[i],Param[i])
            else:
                Prototypes[i] = Param[i]
               
        return Prototypes
    
        
    def xDNNclassifier(self, Data,Image):
        L, N, W = np.shape(Data)
        radius =1 - math.cos(math.pi/6)
        Data_2 = Data**2
        Data_2 = Data_2.reshape(-1, 64)
        Xnorm = np.sqrt(np.sum(Data_2,axis=1))
        
        data = Data.reshape(-1,64) / (Xnorm.reshape(-1,1))*(np.ones((1,W)))
        Centre = data[0,]
        Centre = Centre.reshape(-1,64)
        Center_power = np.power(Centre,2)
        X = np.array([np.sum(Center_power)])
        Support =np.array([1])
        Noc = 1
        GMean = Centre.copy()
        Radius = np.array([radius])
        ND = 1
        VisualPrototype = {}
        VisualPrototype[1] = Image[0]
        Global_X = 1
        for i in range(2,L+1):
            GMean = (i-1)/i*GMean+data[i-1,]/i
            GDelta=Global_X-np.sum(GMean**2,axis = 1)
            
            # CDmax=max(CentreDensity)
            # CDmin=min(CentreDensity)
            DataDensity=1/(1+np.sum((data[i-1,] - GMean) ** 2)/GDelta)
            if i == 2:
                CentreDensity=1/(1+np.sum(((Centre-np.kron(np.ones((Noc,1)),GMean))**2),axis=1)/GDelta)
                CDmax = max(CentreDensity)
                CDmin = min(CentreDensity)
                distance = cdist(data[i-1,].reshape(1,-1),Centre.reshape(1,-1),'euclidean')[0]
            else:
                distance = cdist(data[i-1,].reshape(1,-1),Centre,'euclidean')[0]
            value,position= distance.min(0),distance.argmin(0)
            if (DataDensity > CDmax or DataDensity < CDmin):
                if (DataDensity > CDmax):
                    CDmax = DataDensity
                elif (DataDensity < CDmin):
                    CDmin = DataDensity
            # if (DataDensity > CDmax or DataDensity < CDmin) or value >2*Radius[position]:
                Centre=np.vstack((Centre,data[i-1,]))
                Noc=Noc+1
                VisualPrototype[Noc]=Image[i-1]
                X=np.vstack((X,ND))
                Support=np.vstack((Support, 1))
                Radius=np.vstack((Radius, radius))
            else:
                Centre[position,] = Centre[position,]*(Support[position]/(Support[position]+1))+data[i-1]/(Support[position]+1)
                Support[position]=Support[position]+1
                Radius[position]=0.5*Radius[position]+0.5*(X[position,]-sum(Centre[position,]**2))/2  
        dic = {}
        dic['Noc'] =  Noc
        dic['Centre'] =  Centre
        dic['Support'] =  Support
        dic['Radius'] =  Radius
        dic['GMean'] =  GMean
        dic['Prototype'] = VisualPrototype
        dic['L'] =  L
        dic['X'] =  X
        dic['CDmax'] = CDmax
        dic['CDmin'] = CDmin

        return dic
        
    def xDNNclassifier_updating(self, Data,Image,PARAM):
        L, N, W = np.shape(Data)
        radius =1 - math.cos(math.pi/6)
        Data_2 = Data**2
        Data_2 = Data_2.reshape(-1, 64)
        Xnorm = np.sqrt(np.sum(Data_2,axis=1))
        
        data = Data.reshape(-1,64) / (Xnorm.reshape(-1,1))*(np.ones((1,W)))
        # Centre = data[0,]
        # Centre = Centre.reshape(-1,64)
        # Center_power = np.power(Centre,2)
        # X = np.array([np.sum(Center_power)])
        
        Noc = PARAM['Noc']
        Centre=PARAM['Centre']
        Support=PARAM['Support']
        Radius=PARAM['Radius']
        GMean=PARAM['GMean']
        ND = 1
        VisualPrototype=PARAM['Prototype']
        K=PARAM['L']
        X=PARAM['X']
        CDmax = PARAM['CDmax']
        CDmin = PARAM['CDmin']
        Global_X = 1
        for i in range(K+2,L+K+1):
            GMean = (i-1)/i*GMean+data[i-K-1,]/i
            GDelta=Global_X-np.sum(GMean**2,axis = 1)
            DataDensity=1/(1+np.sum((data[i-K-1,] - GMean) ** 2)/GDelta)
            distance = cdist(data[i-K-1,].reshape(1,-1),Centre,'euclidean')[0]
            value,position= distance.min(0),distance.argmin(0)
            if (DataDensity > CDmax or DataDensity < CDmin):
            # if (DataDensity > CDmax or DataDensity < CDmin) or value >2*Radius[position]:
                if (DataDensity > CDmax):
                    CDmax = DataDensity
                elif (DataDensity < CDmin):
                    CDmin = DataDensity
                Centre=np.vstack((Centre,data[i-K-1,]))
                Noc=Noc+1
                VisualPrototype[Noc]=Image[i-K-1]
                X=np.vstack((X,ND))
                Support=np.vstack((Support, 1))
                Radius=np.vstack((Radius, radius))
            else:
                Centre[position,] = Centre[position,]*(Support[position]/(Support[position]+1))+data[i-K-1]/(Support[position]+1)
                Support[position]=Support[position]+1
                Radius[position]=0.5*Radius[position]+0.5*(X[position,]-sum(Centre[position,]**2))/2  
        dic = {}
        dic['Noc'] =  Noc
        dic['Centre'] =  Centre
        dic['Support'] =  Support
        dic['Radius'] =  Radius
        dic['GMean'] =  GMean
        dic['Prototype'] = VisualPrototype
        dic['L'] =  L+K
        dic['X'] =  X
        dic['CDmax'] = CDmax
        dic['CDmin'] = CDmin
        return dic
        
        

    def DecisionMaking(self, Params,datates,NN=10):
        
        
        PARAM=Params['Parameters']
        # convert dictionary to array
        features_prop = np.concatenate([PARAM[i] for i in range(len(PARAM))])
        labels_prop = np.concatenate([np.ones(PARAM[i].shape[0])*i for i in range(len(PARAM))]).reshape(-1,1)
        
        winner_model = KNeighborsClassifier(n_neighbors=NN)
        winner_model.fit(features_prop, labels_prop)
        
        results = winner_model.predict(datates)
        test = winner_model.predict_proba(datates)
        
        dic = {}
        dic['EstimatedLabels'] = results
        
        return dic
        
            
    def save_model(self, model, name='xDNN_model'):
        with open(name, 'wb') as file:
            pickle.dump(model, file)

    def load_model(self, name='xDNN_model'):
        with open(name, 'rb') as file:
            pickle_model = pickle.load(file)
        return pickle_model

    def results(self, predicted,y_test_labels):

        accuracy = accuracy_score(y_test_labels , predicted)
        print('Accuracy: %f' % accuracy)
        # precision tp / (tp + fp)
        precision = precision_score(y_test_labels ,predicted, average='weighted')
        print('Precision: %f' % precision)
        # recall: tp / (tp + fn)
        recall = recall_score(y_test_labels , predicted,average='weighted')
        print('Recall: %f' % recall)
        # f1: 2 tp / (2 tp + fp + fn)
        f1 = f1_score(y_test_labels , predicted, average='weighted')
        print('F1 score: %f' % f1)
        # kappa
        kappa = cohen_kappa_score(y_test_labels , predicted)
        print('Cohens kappa: %f' % kappa)
        # confusion matrix
        matrix = confusion_matrix(y_test_labels , predicted)
        print("Confusion Matrix: ",matrix)
def normalize(batch_image):

    channel_configuration_bands = get_channel_configuration_bands(config.model_params.hyperparameters.channel_configuration)

    mean_batch = SENTINEL2_NORMALIZATION[channel_configuration_bands, 0]
    mean_batch = torch.tensor(mean_batch[None, :, None, None])  # (1, num_channels, 1, 1)

    std_batch = SENTINEL2_NORMALIZATION[channel_configuration_bands, 1]
    std_batch = torch.tensor(std_batch[None, :, None, None])  # (1, num_channels, 1, 1)
    assert batch_image.ndim == 4, "Expected 4d tensor"
    return (batch_image - mean_batch) / (std_batch + 1e-6)

def centralisation(data):
    CentralizedData = data.copy()

    rows = data.shape[0]
    cols = data.shape[1]

    for j in range(cols):
        mu = np.mean(data[:,j])
        for i in range(rows):
            CentralizedData[i,j] = data[i,j] - mu
    return CentralizedData


def fit_model():


    # water_model_dir = r'/mmfs1/scratch/hpc/00/zhangz65/IDSS/Code/models/5_th/EMSR279_05ZARAGOZA_DEL_MONIT02_v1_observed_event_a.tif_5_water_model.sav'

    # land_model_dir = r'/mmfs1/scratch/hpc/00/zhangz65/IDSS/Code/models/5_th/EMSR279_05ZARAGOZA_DEL_MONIT02_v1_observed_event_a.tif_5_land_model.sav'

    # cloud_model_dir = r'/mmfs1/scratch/hpc/00/zhangz65/IDSS/Code/models/5_th/EMSR279_05ZARAGOZA_DEL_MONIT02_v1_observed_event_a.tif_5_cloud_model.sav'

    water_model_dir = r'/home/zhangz65/IDSS/code/IDSS_code/models/ST1_20161014_WaterExtent_BinhDinh_Lake.tif_5_water_model.sav'
    land_model_dir = r'/home/zhangz65/IDSS/code/IDSS_code/models/ST1_20161014_WaterExtent_BinhDinh_Lake.tif_5_land_model.sav'
    cloud_model_dir = r'/home/zhangz65/IDSS/code/IDSS_code/models/ST1_20161014_WaterExtent_BinhDinh_Lake.tif_5_cloud_model.sav'


    water_model = pickle.load(open(water_model_dir, 'rb'))
    land_model = pickle.load(open(land_model_dir, 'rb'))
    cloud_model = pickle.load(open(cloud_model_dir, 'rb'))

    water_prop = water_model.cluster_centers_

    water_label = np.empty((water_prop.shape[0], 1))

    water_label.fill(2)



    land_prop = land_model.cluster_centers_

    land_label = np.empty((land_prop.shape[0], 1))

    land_label.fill(1)

    cloud_prop = cloud_model.cluster_centers_

    cloud_label = np.empty((cloud_prop.shape[0], 1))

    cloud_label.fill(0)

    features = np.concatenate((cloud_prop, land_prop, water_prop), axis=0)
    
    # features = features[:, 0:64]

    label = np.concatenate((cloud_label, land_label, water_label), axis=0).astype(int)
    
    # features = features[:, 0:64]
    
    # PCA

    
    # features = StandardScaler().fit_transform(features)
    
    # transformer = Normalizer(norm = 'l2').fit(features)
    
    # features_norm = transformer.transform(features)
    
    # features_cen = centralisation(features)
    
    # pca = PCA(n_components=2)
    # X_pca = pca.fit_transform(features_cen)
    
    # print(pca.explained_variance_ratio_)
  
    # cdict = {0: 'yellow', 1: 'green', 2: 'blue'}
    # plt.figure(figsize=(10, 5))
    
    # for g in np.unique(label):
    #     ix = np.where(label == g)
    #     plt.scatter(X_pca[ix[0], 0], X_pca[ix[0], 1], c = cdict[g], label = g, cmap="jet")
    # plt.legend()
    # plt.savefig('pca.png', dpi=600)
    
    
    
    


    model_KNN = KNeighborsClassifier(n_neighbors=10)



   
    model_KNN.fit(features, label)

    return model_KNN,features,label


def rgb_sentinel_2(rgb, factor=255, clip_range = (0, 1)):
    rgb = np.clip(rgb/4000, *clip_range)
    rgb = rgb*factor
    rgb = rgb.astype(np.uint8)
    return rgb




def prediction(features,model_KNN,features_prop, labels_prop):
    # modelFile_name = r'/mmfs1/scratch/hpc/00/zhangz65/train/water_patches/Code/U_net/model2/xDNN_online_ST1_20161014_WaterExtent_BinhDinh_Laketile_512-512.tif_1130.sav'

    # model = xDNNClassifier()
    # # model_file = model.load_model(modelFile_name)

    # TestData = {}
    # TestData['xDNNParms'] = {}
    # TestData['xDNNParms']['Parameters'] = {}
    # TestData['xDNNParms']['Parameters'][0] = {}
    # TestData['xDNNParms']['Parameters'][1] = {}
    # TestData['xDNNParms']['Parameters'][2] = {}
    # TestData['xDNNParms']['MemberLabels'] = {}
    # TestData['xDNNParms']['CurrentNumberofClass'] = {}

    # water_model_dir = r'/mmfs1/scratch/hpc/00/zhangz65/train/water_patches/Code/k_means/SSCI/Model/water/EMSR279_05ZARAGOZA_DEL_MONIT02_v1_observed_event_a.tif_5_water_model.sav'

    # land_model_dir = r'/mmfs1/scratch/hpc/00/zhangz65/train/water_patches/Code/k_means/SSCI/Model/land/EMSR279_05ZARAGOZA_DEL_MONIT02_v1_observed_event_a.tif_5_land_model.sav'

    # cloud_model_dir = r'/mmfs1/scratch/hpc/00/zhangz65/train/water_patches/Code/k_means/SSCI/Model/cloud/EMSR279_05ZARAGOZA_DEL_MONIT02_v1_observed_event_a.tif_5_cloud_model.sav'

    # water_model_dir = r'E:\Lancaster_PhD\PHD\Data\WORLDFLOODS\Code\7.12\ssci\model\EMSR279_05ZARAGOZA_DEL_MONIT02_v1_observed_event_a.tif_5_water_model.sav'

    # land_model_dir = r'E:\Lancaster_PhD\PHD\Data\WORLDFLOODS\Code\7.12\ssci\model\EMSR279_05ZARAGOZA_DEL_MONIT02_v1_observed_event_a.tif_5_land_model.sav'

    # cloud_model_dir = r'E:\Lancaster_PhD\PHD\Data\WORLDFLOODS\Code\7.12\ssci\model\EMSR279_05ZARAGOZA_DEL_MONIT02_v1_observed_event_a.tif_5_cloud_model.sav'


    # water_model = pickle.load(open(water_model_dir, 'rb'))
    # land_model = pickle.load(open(land_model_dir, 'rb'))
    # cloud_model = pickle.load(open(cloud_model_dir, 'rb'))

    # water_prop = water_model.cluster_centers_

    # water_label = np.empty((water_prop.shape[0], 1))

    # water_label.fill(2)



    # land_prop = land_model.cluster_centers_

    # land_label = np.empty((land_prop.shape[0], 1))

    # land_label.fill(1)

    # cloud_prop = cloud_model.cluster_centers_

    # cloud_label = np.empty((cloud_prop.shape[0], 1))

    # cloud_label.fill(0)

    # features = np.concatenate((cloud_prop, land_prop, water_prop), axis=0)

    # label = np.concatenate((cloud_label, land_label, water_label), axis=0)


    # model_KNN = KNeighborsClassifier(n_neighbors=10)



   
    # model.fit(features, label)


    # TestData['xDNNParms']['Parameters'][0]['Centre'] = cloud_prop
    # TestData['xDNNParms']['Parameters'][1]['Centre'] = land_prop
    # TestData['xDNNParms']['Parameters'][2]['Centre'] = water_prop

    # TestData['xDNNParms']['CurrentNumberofClass'] = 3
    # TestData['xDNNParms']['MemberLabels'][0] = 0
    # TestData['xDNNParms']['MemberLabels'][1] = 1
    # TestData['xDNNParms']['MemberLabels'][2] = 2
    

    
    # TestData = {}
    
    # TestData ['xDNNParms'] =  model_file['xDNNParms']
 
    # TestData ['Features'] = features
    

    start = time.time()
    # xDNN Predict
    pred= KNN_model.predict(features)
    # print(np.unique(pred,return_counts=True))
    
    # rgb_test = features[1, 65:68]*1000000000
    # rgb_test = rgb_sentinel_2(rgb_test).reshape(1,1,3)
    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    # plt.margins(0, 0)
    # plt.gca().xaxis.set_major_locator(NullLocator())
    # plt.gca().yaxis.set_major_locator(NullLocator())
    # plt.imshow(rgb_test)

    # # plt.imshow(rgb_test)
    # # plt.show()
    # # rgb_test =cv2.cvtColor(rgb_test, cv2.COLOR_RGB2BGR)
    # rgb_test_label = pred[1]
    # # cv2.imwrite(r'E:\Lancaster_PhD\PHD\Data\WORLDFLOODS\Code\IDSS\new_version\rgb_visilization\sample1\rgb_test_'+str(rgb_test_label)+'.png',rgb_test)
    # plt.savefig(r'E:\Lancaster_PhD\PHD\Data\WORLDFLOODS\Code\IDSS\new_version\rgb_visilization\sample3\rgb_test_'+str(rgb_test_label)+'.png', dpi=600)
    
    # distances, indices = KNN_model.kneighbors(features[3].reshape(1, -1))
    # idx = [-1,-2,-3]
    # for _,(j,i) in enumerate(zip(idx,indices[0][-3:])):
    #     rgb_1 = features_prop[i, 65:68]*100000000
    #     rgb_1 = rgb_sentinel_2(rgb_1).reshape(1,1,3)
    #     distance = distances[0][j]
    #     similarity = np.exp(-distance)
    #     label1 = labels_prop[i]
    #     plt.imshow(rgb_1)
    #     plt.savefig(r'E:\Lancaster_PhD\PHD\Data\WORLDFLOODS\Code\IDSS\new_version\rgb_visilization\sample3\rgb_dissimilar'+str(label1)+ str(similarity)+'.png', dpi=600)
        # plt.show()
        # print(rgb_1)
        
    # idx = [0,1,2]
    # for _,(j,i) in enumerate(zip(idx,indices[0][:3])):
    #     rgb_1 = features_prop[i, 65:68]*100000000
    #     rgb_1 = rgb_sentinel_2(rgb_1).reshape(1,1,3)
    #     distance = distances[0][j]
    #     similarity = np.exp(-distance)
    #     label1 = labels_prop[i]
    #     plt.imshow(rgb_1)
    #     plt.savefig(r'E:\Lancaster_PhD\PHD\Data\WORLDFLOODS\Code\IDSS\new_version\rgb_visilization\sample3\rgb_similar'+str(label1)+ str(similarity)+'.png', dpi=600)
    #     # plt.show()
    #     # print(rgb_1)    
        
    
    pred_confidence = KNN_model.predict_proba(features)
    
    
    pred_confidence = pred_confidence.max(axis=1)
    end = time.time()
    print("Validation Time: ",round(end - start,2), "seconds")
    
    return pred,pred_confidence

def prediction_batches(patch,original_patch,KNN_model,features_prop, labels_prop):

    # print(patch.shape[:2])
    original = original_patch.reshape(-1,13)/100000000

    patch_reshape = patch.reshape(-1,64)
    
    patch_reshape = patch_reshape/ np.sqrt(np.sum(patch_reshape**2,axis=1)).reshape(-1,1)
    
    final_patch = np.concatenate((patch_reshape,original),axis=1)
    
    pred,pred_confidence = prediction(final_patch,KNN_model,features_prop, labels_prop)
    pred_labels = pred.reshape(patch.shape[:2])
    pred_labels_confidence = pred_confidence.reshape(patch.shape[:2])
    
    final_pred = pred_labels

    return final_pred



print(torch.__version__)

try:
    from google.colab import drive
    drive.mount('/content/drive')
    assert os.path.exists('/content/drive/My Drive/Public WorldFloods Dataset'), "Add a shortcut to the publice Google Drive folder: https://drive.google.com/drive/u/0/folders/1dqFYWetX614r49kuVE3CbZwVO6qHvRVH"
    google_colab = True
    path_to_dataset_folder = '/content/drive/My Drive/Public WorldFloods Dataset'
    dataset_folder = os.path.join(path_to_dataset_folder, "worldfloods_v1_0_sample")
    experiment_name = "WFV1_unet"
    folder_name_model_weights = os.path.join(path_to_dataset_folder, experiment_name)
except ImportError as e:
    print(e)
    print("Setting google colab to false, it will need to install the gdown package!")
    google_colab = False


# Download pre-trained model from Google Drive folder
if not google_colab:
    experiment_name = "WFV1_unet"
    # Download val folder
    path_to_dataset_folder = '.'
    # dataset_folder = os.path.join(path_to_dataset_folder, "worldfloods_v1_0_sample")
    # val_folder = os.path.join(dataset_folder, "val")
    
    folder_name_model_weights = os.path.join(path_to_dataset_folder, experiment_name)
    
    if not os.path.exists(folder_name_model_weights):
        import gdown
        gdown.download_folder(id="1Oup-qVD1U-re3lIQkw7TOKJsdu90blsk", quiet=False, use_cookies=False,
                              output=folder_name_model_weights)

    # if not os.path.exists(val_folder):
    #     import gdown
    #     os.makedirs(val_folder, exist_ok=True)
    #     # https://drive.google.com/drive/folders/1ogcNupGr0q6nLwS7BBQQ8PzILyONah12?usp=sharing
    #     gdown.download_folder(id="1ogcNupGr0q6nLwS7BBQQ8PzILyONah12", quiet=False, use_cookies=False,
    #                           output=val_folder)


from ml4floods.models.config_setup import get_default_config

config_fp = os.path.join(folder_name_model_weights, "config.json")

config = get_default_config(config_fp)

# The max_tile_size param controls the max size of patches that are fed to the NN. If you're in a memory contrained environment set this value to 128
config["model_params"]["max_tile_size"] = 128

from ml4floods.models.model_setup import get_model

model_folder = os.path.dirname(folder_name_model_weights)
if model_folder == "":
    model_folder = "."

config["model_params"]['model_folder'] = model_folder
config["model_params"]['test'] = True
model = get_model(config.model_params, experiment_name)

model.eval()

# print(model)

pytorch_total_params = sum(p.numel() for p in model.parameters())

print("Total number of parameters: ", pytorch_total_params)

pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(pytorch_total_params)
model.to("cuda") # comment this line if your machine does not have GPU

return_nodes = {
    # node_name: user-specified key for output dict
    
    'network.dconv_up1.3': 'layer1',
}


model = create_feature_extractor(model, return_nodes=return_nodes)

# tiff_s2 = r'E:\Lancaster_PhD\PHD\Data\WORLDFLOODS\test\S2\tif'

tiff_s2 = r'/home/zhangz65/IDSS/data/test/S2/tif'
# tiff_s2 = r'/mmfs1/scratch/hpc/00/zhangz65/IDSS/Dataset/test/S2/tif/'

contents = os.listdir(tiff_s2)

patch_size = 256

KNN_model,features_prop, labels_prop = fit_model()

for index, each in enumerate(contents):

    imgdir = os.path.join(tiff_s2,each)
    img_rasterio = rasterio.open(imgdir)
    img = img_rasterio.read(range(1,14))
    img = img.transpose(1,2,0)
    img = np.nan_to_num(img)



    # img = torch.unsqueeze(torch.from_numpy(img.astype(np.float32)),0)
    SIZE_X = (img.shape[0]//patch_size+1)*patch_size #Nearest size divisible by our patch size(Larger than the original size)
    SIZE_Y = (img.shape[1]//patch_size+1)*patch_size #Nearest size divisible by our patch size(Larger than the original size)

    img_padded = np.zeros((SIZE_X,SIZE_Y,13))
    img_padded[:img.shape[0],:img.shape[1],:] = img

 



   
  
    # print(out.shape)

    patches_img = patchify(img_padded, (256,256,13), step=256)

    patches_prediction = []
    print(f'The number of the patches is {str((patches_img.shape[0])*(patches_img.shape[1]))}')

    for i in range(patches_img.shape[0]):
            for j in range(patches_img.shape[1]):
            
                single_patch_img = patches_img[i,j,:,:,:]
                #single_patch_img = (single_patch_img.astype('float32')) / 255. #We will preprocess using one of the backbones
                single_patch_img = single_patch_img[0] #Drop the extra unecessary dimension that patchify adds.

                original_single_patch_img = single_patch_img.copy()
                

                print(f'single_patch is {single_patch_img.shape}')
                single_patch_img = np.transpose(single_patch_img,(2,0,1))
                
                # rgb_vis = np.transpose(single_patch_img,())
                # rgb_vis = single_patch_img[65:68]*1000000000
                # rgb_vis = rgb_sentinel_2(rgb_vis)
                # plt.imshow(rgb_vis)
                # plt.show()

                single_patch_img = torch.unsqueeze(torch.from_numpy(single_patch_img.astype(np.float32)),0)
                # single_patch_img = single_patch_img.transpose(single_patch_img,1,3)
                single_patch_img = normalize(single_patch_img)
                
                time1 = time.time()

                out = model(single_patch_img.cuda())
                

                
                out = torch.squeeze(out['layer1'],0).cpu().detach().numpy().transpose(1,2,0)
                
                time2 = time.time()
                
                print(f'inference time is {time2-time1}')
                
                print(f'out is {out.shape}')


                pred = prediction_batches(out,original_single_patch_img,KNN_model,features_prop, labels_prop)
            

                patches_prediction.append(pred)




                # pred = prediction_batches(out,original_single_patch_img,KNN_model)
            

                
                print(f"finish {str(i)}  {str(j)}")
    patches_prediction = np.array(patches_prediction)

    patches_prediction = np.reshape(patches_prediction, [patches_img.shape[0], patches_img.shape[1], 
                                                patches_img.shape[2], patches_img.shape[3],patches_img.shape[4],1])
    

    unpatched_prediction = unpatchify(patches_prediction, (img_padded.shape[0], img_padded.shape[1],1))
    unpatched_prediction = unpatched_prediction[0:img.shape[0], 0:img.shape[1]]
    
    # save unpatched_prediction

    # np.save(r'/mmfs1/scratch/hpc/00/zhangz65/IDSS/error_anlysis/confidence_prede_64/'+str(os.path.splitext(each)[0])+'.npy', unpatched_prediction)
    # np.save(r'E:\Lancaster_PhD\PHD\Data\WORLDFLOODS\Code\IDSS\new_version\error_analysis\confidenc_pred_64/'+str(os.path.splitext(each)[0])+'.npy', unpatched_prediction)
    
    
    

    rgb_conversion = np.zeros((unpatched_prediction.shape[0],unpatched_prediction.shape[1],3))
    
    start = time.time()
    
    

    # For unpatched_prediction == 1 land
    rgb_conversion[np.where(unpatched_prediction[:,:,0] == 1)] = [120, 183, 53]

    # For unpatched_prediction == 2 water
    rgb_conversion[np.where(unpatched_prediction[:,:,0] == 2)] = [141, 103, 48]

    # For unpatched_prediction == 0 cloud
    rgb_conversion[np.where(unpatched_prediction[:,:,0] == 0)] = [36, 231, 253]
    
    rgb_conversion = rgb_conversion.astype(int)
    
    B_04_invalid = img_rasterio.read(4)
    invalid_pixel = B_04_invalid == 0

    rgb_conversion[invalid_pixel] = [0,0,0]
    
    end = time.time()
    
    print(f'rgb conversion time is {end-start}')
    
    cv2.imwrite(r'/home/zhangz65/IDSS/pred_1000/'+str(os.path.splitext(each)[0])+'.png', rgb_conversion)
    # for i in range(rgb_conversion.shape[0]):
    #   for j in range(rgb_conversion.shape[1]):
    #     # mask 
    #     if np.logical_and(unpatched_prediction[i,j] >= 0.79, unpatched_prediction[i,j] <= 1.0):
    #         rgb_conversion[i,j] = [0,204,0]
    #     elif np.logical_and(unpatched_prediction[i,j] >= 0.59, unpatched_prediction[i,j] < 0.8):
    #         rgb_conversion[i,j] = [0,255,0]
    #     elif np.logical_and(unpatched_prediction[i,j] >= 0.39, unpatched_prediction[i,j] < 0.6):
    #         rgb_conversion[i,j] = [51,255,51]
    #     elif np.logical_and(unpatched_prediction[i,j] >= 0.19, unpatched_prediction[i,j] < 0.4):
    #         rgb_conversion[i,j] = [102,255,102]
    #     elif np.logical_and(unpatched_prediction[i,j] >= 0.0, unpatched_prediction[i,j] < 0.2):
    #         rgb_conversion[i,j] = [153,255,153]
    #     elif np.logical_and(unpatched_prediction[i,j] >= 1.79, unpatched_prediction[i,j] <= 2.0):
    #         rgb_conversion[i,j] = [0,102,204]
    #     elif np.logical_and(unpatched_prediction[i,j] >= 1.59, unpatched_prediction[i,j] < 1.8):
    #         rgb_conversion[i,j] = [0,128,255]
    #     elif np.logical_and(unpatched_prediction[i,j] >= 1.39, unpatched_prediction[i,j] < 1.6):
    #         rgb_conversion[i,j] = [51,153,255]
    #     elif np.logical_and(unpatched_prediction[i,j] >= 1.19, unpatched_prediction[i,j] < 1.4):
    #         rgb_conversion[i,j] = [102,178,255]
    #     elif np.logical_and(unpatched_prediction[i,j] >= 1.0, unpatched_prediction[i,j] < 1.2):
    #         rgb_conversion[i,j] = [153,204,255]
    #     elif np.logical_and(unpatched_prediction[i,j] >= 2.79, unpatched_prediction[i,j] <= 3.0):
    #         rgb_conversion[i,j] = [204,204,0]
    #     elif np.logical_and(unpatched_prediction[i,j] >= 2.59, unpatched_prediction[i,j] < 2.8):
    #         rgb_conversion[i,j] = [255,255,0]
    #     elif np.logical_and(unpatched_prediction[i,j] >= 2.39, unpatched_prediction[i,j] < 2.6):
    #         rgb_conversion[i,j] = [255,255,51]
    #     elif np.logical_and(unpatched_prediction[i,j] >= 2.19, unpatched_prediction[i,j] < 2.4):
    #         rgb_conversion[i,j] = [255,255,102]
    #     elif np.logical_and(unpatched_prediction[i,j] >= 2.0, unpatched_prediction[i,j] < 2.2):
    #         rgb_conversion[i,j] = [255,255,153]
    # rgb_conversion = rgb_conversion.astype(int)

    # cv2.imwrite(r'/mmfs1/scratch/hpc/00/zhangz65/IDSS/Code/pred_new/'+str(os.path.splitext(each)[0])+'.png', rgb_conversion)




    



