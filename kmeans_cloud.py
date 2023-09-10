from sklearn.cluster import MiniBatchKMeans
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
from rasterio import windows
from itertools import product

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
        Output['Scores'] = Scores
        Output['ConfMa'] = confusion_matrix(Input['Labels'],Output['EstLabs'])
        Output['ClassAcc'] = np.sum(Output['ConfMa']*np.identity(len(Output['ConfMa'])))/len(Input['Labels'])
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
            CentreDensity=1/(1+np.sum(((Centre-GMean)**2),axis=1)/GDelta)
            CDmax = CentreDensity.max()
            CDmin =CentreDensity.min()
            DataDensity=1/(1+np.sum((data[i-1,] - GMean) ** 2)/GDelta)
            if i == 2:
                distance = cdist(data[i-1,].reshape(1,-1),Centre.reshape(1,-1),'euclidean')[0]
            else:
                distance = cdist(data[i-1,].reshape(1,-1),Centre,'euclidean')[0]
            value,position= distance.min(0),distance.argmin(0)
            if (DataDensity > CDmax or DataDensity < CDmin):
            #     if (DataDensity > CDmax):
            #         CDmax = DataDensity
            #     elif (DataDensity < CDmin):
            #         CDmin = DataDensity
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
            CentreDensity=1/(1+np.sum(((Centre-GMean)**2),axis=1)/GDelta)
            CDmax = CentreDensity.max()
            CDmin =CentreDensity.min()
            distance = cdist(data[i-K-1,].reshape(1,-1),Centre,'euclidean')[0]
            value,position= distance.min(0),distance.argmin(0)
            if (DataDensity > CDmax or DataDensity < CDmin):
            # if (DataDensity > CDmax or DataDensity < CDmin) or value >2*Radius[position]:
                # if (DataDensity > CDmax):
                #     CDmax = DataDensity
                # elif (DataDensity < CDmin):
                #     CDmin = DataDensity
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
        
        

    def DecisionMaking(self, Params,datates,NN=1):
        PARAM=Params['Parameters']
        CurrentNC=Params['CurrentNumberofClass']
        LAB=Params['MemberLabels']
        VV = 1
        LTes=np.shape(datates)[0]
        EstimatedLabels = np.zeros((LTes))
        Scores=np.zeros((LTes,CurrentNC))
        for i in range(1,LTes + 1):
            data = datates[i-1,]
            Data_2 = data**2
            Data_2 = Data_2.reshape(-1, 64)
            Xnorm = np.sqrt(np.sum(Data_2,axis=1))
            data = data/Xnorm
            R=np.zeros((VV,CurrentNC))
            numPrototypes = 0
            for j in range(CurrentNC):
                numPrototypes = numPrototypes+PARAM[j]['Noc']
            for k in range(0,CurrentNC):
                if k == 0:
                    #distance=np.sort(cdist(data.reshape(1, -1),PARAM[k]['Centre'],'minkowski',8))[0]
                    # test = np.exp(-1*cdist(data.reshape(1, -1),PARAM[k]['Centre'],'euclidean')**2)
                    # distance=np.sort(cdist(data.reshape(1, -1),PARAM[k]['Centre'],'euclidean'))[0]

                    # distance = np.sort(np.exp(-1*cdist(data.reshape(1, -1),PARAM[k]['Centre'],'euclidean')**2)).T
                    distance = np.exp(-1*cdist(data.reshape(1, -1),PARAM[k]['Centre'],'euclidean')**2).T

                    label = np.full(distance.shape,k)

                else:
                    distance_new = np.exp(-1*cdist(data.reshape(1, -1),PARAM[k]['Centre'],'euclidean')**2).T
                    label_new = np.full(distance_new.shape,k)

                    distance = np.vstack((distance,distance_new))
                    label = np.vstack((label,label_new))

            distance_label = np.hstack((distance,label))

            distance_label = distance_label[distance_label[:,0].argsort()]

            distance_label = distance_label[-NN:,:]

           
            EstimatedLabels[i-1]=np.argmax(np.bincount(distance_label[:,1].astype(int)))


        LABEL1=np.zeros((CurrentNC,1))
        
        

        for i in range(0,CurrentNC): 
            LABEL1[i] = np.unique(LAB[i])

        EstimatedLabels = EstimatedLabels.astype(int)
        EstimatedLabels = LABEL1[EstimatedLabels]   
        dic = {}
        dic['EstimatedLabels'] = EstimatedLabels
        dic['Scores'] = Scores

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





def get_tiles(ds, width=256, height=256):
    nols, nrows = ds.meta['width'], ds.meta['height']
    offsets = product(range(0, nols, width), range(0, nrows, height))
    big_window = windows.Window(col_off=0, row_off=0, width=nols, height=nrows)
    for col_off, row_off in  offsets:
        window =windows.Window(col_off=col_off, row_off=row_off, width=width, height=height).intersection(big_window)
        transform = windows.transform(window, ds.transform)
        yield window, transform

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
    dataset_folder = os.path.join(path_to_dataset_folder, "worldfloods_v1_0_sample")
    val_folder = os.path.join(dataset_folder, "val")
    
    folder_name_model_weights = os.path.join(path_to_dataset_folder, experiment_name)
    
    if not os.path.exists(folder_name_model_weights):
        import gdown
        gdown.download_folder(id="1Oup-qVD1U-re3lIQkw7TOKJsdu90blsk", quiet=False, use_cookies=False,
                              output=folder_name_model_weights)

    if not os.path.exists(val_folder):
        import gdown
        os.makedirs(val_folder, exist_ok=True)
        # https://drive.google.com/drive/folders/1ogcNupGr0q6nLwS7BBQQ8PzILyONah12?usp=sharing
        gdown.download_folder(id="1ogcNupGr0q6nLwS7BBQQ8PzILyONah12", quiet=False, use_cookies=False,
                              output=val_folder)


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
model_Unet = get_model(config.model_params, experiment_name)

model_Unet.eval()
# model.to("cuda") # comment this line if your machine does not have GPU
return_nodes = {
    # node_name: user-specified key for output dict
    
    'network.dconv_up1.3': 'layer1',
}


model_Unet = create_feature_extractor(model_Unet, return_nodes=return_nodes)

# The training images: you can use only 6 images from validation set right now
img_in_path = r'E:\SCC\SCC3\TYP\ml4floods\worldfloods_v1_0_sample\val\S2'
gt_in_path = r'E:\SCC\SCC3\TYP\ml4floods\worldfloods_v1_0_sample\val\gt'


contents = os.listdir(img_in_path)

contents.sort()
kmeans = MiniBatchKMeans(n_clusters=1, random_state=0)


for count,each in enumerate(contents):
    all_bands = np.empty((0,0))
    
    input_filename = each
    stem, suffix = os.path.splitext(input_filename)


    img = rasterio.open(os.path.join(img_in_path, input_filename))
    img_gt = rasterio.open(os.path.join(gt_in_path, input_filename))




    
    tile_width, tile_height = 256, 256

    meta_img = img.meta.copy()
    meta_gt = img_gt.meta.copy()

    




    for window, transform in get_tiles(img):

        
        

        meta_img['transform'] = transform
        meta_img['width'], meta_img['height'] = window.width, window.height
        

        if window.height == 256 and window.width == 256:

   
            img_window = img.read(window=window)
            img_gt_window = img_gt.read(window=window)


            B_bands = torch.unsqueeze(torch.from_numpy(img_window.astype(np.float32)),0)

            
            B_bands_original = np.transpose((img_window/100000000).reshape(13,-1))
            

    

            B_bands = normalize(B_bands)


            B_bands = model_Unet(B_bands)
            B_bands = torch.squeeze(B_bands['layer1'],0).detach().numpy()


        
    
    
            #Reshape the bands from (13,n,m) to (13, n*m)
            B_bands = B_bands.reshape(64,-1)
            
            
            gt_bands = img_gt_window.reshape(1,-1)
            
            #Transpose the array from (13, n*m) to (n*m,13)

            B_bands = np.transpose(B_bands)
            gt_bands = np.transpose(gt_bands)

            

            # Create the DataFrames
            B_bands = pd.DataFrame(B_bands, columns=[str(i) for i in range(1, 65)])
            gt_bands = pd.DataFrame(gt_bands, columns=['65'])
            B_bands_original = pd.DataFrame(B_bands_original, columns=[str(i) for i in range(1, 14)])

            # Concatenate gt_bands to B_bands and B_bands_original
            B_bands = pd.concat([B_bands, gt_bands], axis=1)
            B_bands_original = pd.concat([B_bands_original, gt_bands], axis=1)

            # Filter out rows where column '65' is not equal to 1
            B_bands = B_bands[B_bands['65'] == 3]
            B_bands_original = B_bands_original[B_bands_original['65'] == 3]

            # Drop column '65'
            B_bands = B_bands.drop(columns=['65'])
            B_bands_original = B_bands_original.drop(columns=['65'])

            # Convert to numpy arrays
            B_bands = B_bands.to_numpy()
            B_bands_original = B_bands_original.to_numpy()

            if window.height == 256 and window.width == 256 and window.col_off == 0 and window.row_off == 0: 

                all_bands = B_bands

                all_bands_original = B_bands_original

            else:

                all_bands = np.concatenate((all_bands,B_bands),axis = 0)

                all_bands_original = np.concatenate((all_bands_original,B_bands_original),axis = 0)

                # gt_bands = gt_bands['65']

                # gt_bands = gt_bands.to_numpy()

                

                # kmeans = kmeans.partial_fit(B_bands)

                # print(kmeans.cluster_centers_.shape)


        else:
            continue


    print(f'Finish {count} the image is {each}')

    # print(all_bands.shape)

    Xnorm = np.sqrt(np.sum(all_bands**2,axis=1))
        
    all_bands = all_bands/ np.sqrt(np.sum(all_bands**2,axis=1)).reshape(-1,1)

    all_bands = np.concatenate((all_bands,all_bands_original),axis = 1)


    kmeans = kmeans.partial_fit(all_bands)
    # print(kmeans.cluster_centers_.shape)

    #Save xDNN model (optional)

    filename = 'E:/SCC/SCC3/TYP/ml4floods/codes/models/'+str(each)+'_'+str(count)+'_cloud_model.sav'
    pickle.dump(kmeans, open(filename, 'wb'))
   

   


   

