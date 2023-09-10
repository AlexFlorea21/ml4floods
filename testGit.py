from ml4floods.models.config_setup import get_default_config
from ml4floods.models.model_setup import get_model
from ml4floods.models.model_setup import get_model_inference_function

from ml4floods.models.model_setup import get_channel_configuration_bands
from ml4floods.visualization import plot_utils
from ml4floods.data.worldfloods import dataset
import torch
import matplotlib.pyplot as plt

from ml4floods.models import postprocess
from ml4floods.visualization import plot_utils
import geopandas as gpd
import numpy as np

import os

#get config
experiment_name="WFV1_unet"
config_fp = f"worldfloods_v1_0_sample/checkpoints/{experiment_name}/config.json"
config = get_default_config(config_fp)

# The max_tile_size param controls the max size of patches that are fed to the NN. If you're in a memory contrained environment set this value to 128
config["model_params"]["max_tile_size"] = 128

#load model
model_folder = os.path.dirname( f"worldfloods_v1_0_sample/checkpoints/")
if model_folder == "":
    model_folder = "."

config["model_params"]['model_folder'] = model_folder
config["model_params"]['test'] = True
model = get_model(config.model_params, experiment_name)

model.eval()
#model.to("cuda") # comment this line if your machine does not have GPU

#get inference function

inference_function = get_model_inference_function(model, config,apply_normalization=True)

#run inference

channel_configuration = config.model_params.hyperparameters.channel_configuration

# dataset_folder = gs://ml4cc_data_lake/2_PROD/2_Mart/worldfloods_v1_0/
event_id = "RS2_20161008_Water_Extent_Corail_Pestel.tif"
tiff_s2 = os.path.join(dataset_folder, "val", "S2", event_id)
tiff_gt = os.path.join(dataset_folder, "val", "gt", event_id)
tiff_permanentwaterjrc = os.path.join(dataset_folder, "val", "PERMANENTWATERJRC", event_id)
window = None
channels = get_channel_configuration_bands(channel_configuration)

# Read inputs
torch_inputs, transform = dataset.load_input(tiff_s2, window=window, channels=channels)

# Make predictions
outputs = inference_function(torch_inputs.unsqueeze(0))[0] # (num_classes, h, w)
prediction = torch.argmax(outputs, dim=0).long() # (h, w)

# Mask invalid pixels
mask_invalid = torch.all(torch_inputs == 0, dim=0)
prediction+=1
prediction[mask_invalid] = 0

# Load GT and permanent water for plotting
torch_targets, _ = dataset.load_input(tiff_gt, window=window, channels=[0])
torch_permanent_water, _ = dataset.load_input(tiff_permanentwaterjrc, window=window, channels=[0])


# Plot
fig, axs = plt.subplots(2,2, figsize=(16,16))
plot_utils.plot_rgb_image(torch_inputs, transform=transform, ax=axs[0,0])
axs[0,0].set_title("RGB Composite")
plot_utils.plot_swirnirred_image(torch_inputs, transform=transform, ax=axs[0,1])
axs[0,1].set_title("SWIR1,NIR,R Composite")
plot_utils.plot_gt_v1_with_permanent(torch_targets, torch_permanent_water, window=window, transform=transform, ax=axs[1,0])
axs[1,0].set_title("Ground Truth with JRC Permanent")
plot_utils.plot_gt_v1(prediction.unsqueeze(0),transform=transform, ax=axs[1,1])
axs[1,1].set_title("Model prediction")
plt.tight_layout()

#vectorise water and plot

prob_water_mask = outputs[1].cpu().numpy()
binary_water_mask = prob_water_mask>.5

geoms_polygons = postprocess.get_water_polygons(binary_water_mask, transform=transform)

data_out = gpd.GeoDataFrame({"geometry": geoms_polygons, "id": np.arange(len(geoms_polygons))})
fig, ax = plt.subplots(1,1, figsize=(12, 12))
data_out.plot("id",legend=True,categorical=True,ax=ax,facecolor="None",edgecolor=None,linewidth=3)
plot_utils.plot_rgb_image(torch_inputs, transform=transform, ax=ax, alpha=.6,
                             channel_configuration=channel_configuration)