import sys, os
from pathlib import Path

import fsspec
from pytorch_lightning import seed_everything
import pkg_resources

from ml4floods.models.config_setup import get_default_config
from ml4floods.models.model_setup import get_model,get_model_inference_function,get_channel_configuration_bands
import
import warnings
warnings.filterwarnings("ignore")

@torch.no_grad() #Deactivate autograd engine
def read_inference_pair(tiff_inputs:str, folder_ground_truth:str, 
                        window:Optional[Union[rasterio.windows.Window, Tuple[slice,slice]]], 
                        return_ground_truth: bool=False, channels:bool=None, 
                        folder_permanent_water:Optional[str]=None,
                        cache_folder:Optional[str]=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, rasterio.Affine]:
    """
    Read a pair of layers from the worldfloods bucket and return them as Tensors to pass to a model, return the transform for plotting with lat/long
    
    Args:
        tiff_inputs: filename for layer in worldfloods bucket
        folder_ground_truth: folder name to be replaced by S2 in the input
        window: window of layer to use
        return_ground_truth: flag to indicate if paired gt layer should be returned
        channels: list of channels to read from the image
        return_permanent_water: Read permanent water layer raster
    
    Returns:
        (torch_inputs, torch_targets, transform): inputs Tensor, gt Tensor, transform for plotting with lat/long
    """
    
    if cache_folder is not None and tiff_inputs.startswith("gs"):
        tiff_inputs = download_tiff(cache_folder, tiff_inputs, folder_ground_truth, folder_permanent_water)
    
    tiff_targets = tiff_inputs.replace("/S2/", folder_ground_truth)

    with rasterio.open(tiff_inputs, "r") as rst:
        inputs = rst.read((np.array(channels) + 1).tolist(), window=window)
        # Shifted transform based on the given window (used for plotting)
        transform = rst.transform if window is None else rasterio.windows.transform(window, rst.transform)
        torch_inputs = torch.Tensor(inputs.astype(np.float32)).unsqueeze(0)
    
    if folder_permanent_water is not None:
        tiff_permanent_water = tiff_inputs.replace("/S2/", folder_permanent_water)
        with rasterio.open(tiff_permanent_water, "r") as rst:
            permanent_water = rst.read(1, window=window) 
            torch_permanent_water = torch.tensor(permanent_water.astype(np.int16))
    else:
        torch_permanent_water = torch.zeros_like(torch_inputs)
        
    if return_ground_truth:
        with rasterio.open(tiff_targets, "r") as rst:
            targets = rst.read(1, window=window)
        
        torch_targets = torch.tensor(targets.astype(np.int16)).unsqueeze(0)
    else:
        torch_targets = torch.zeros_like(torch_inputs)
    
    return torch_inputs, torch_targets, torch_permanent_water, transform

experiment_name = "WFV1_unet"

config_fp = f"worldfloods_v1_0_sample/checkpoints/{experiment_name}/config.json"
config = get_default_config(config_fp)

config["model_params"]["max_tile_size"] = 128

config["model_params"]['model_folder'] = 'worldfloods_v1_0_sample/checkpoints'
config["model_params"]['test'] = True

model = get_model(config.model_params, experiment_name=experiment_name)
model

inference_function = get_model_inference_function(model, config, apply_normalization=True)

tiff_s2 = "worldfloods_v1_0_sample/val/S2/RS2_20161008_Water_Extent_Corail_Pestel.tif"
window = None
channels = get_channel_configuration_bands(config.model_params.hyperparameters.channel_configuration)

torch_inputs, torch_targets, \
torch_permanent_water, transform = read_inference_pair(tiff_s2,folder_ground_truth="/gt/",  window=window, return_ground_truth=True, 
                                                          channels=channels, folder_permanent_water="/PERMANENTWATERJRC/", cache_folder=None)

outputs = inference_function(torch_inputs)
outputs.shape

prediction = torch.argmax(outputs, dim=1).long()
plot_inference_set(torch_inputs, torch_targets, prediction, torch_permanent_water, transform)

tiff_s2 = "worldfloods_v1_0_sample/val/S2/EMSR271_02FARKADONA_DEL_v1_observed_event_a.tif"
window = None
channels = get_channel_configuration_bands(config.model_params.hyperparameters.channel_configuration)

torch_inputs, torch_targets, torch_permanent_water, transform = read_inference_pair(tiff_s2, folder_ground_truth="/gt/", 
                                                                                    window=window, 
                                                                                    return_ground_truth=True, 
                                                                                    channels=channels,
                                                                                    folder_permanent_water="/PERMANENTWATERJRC/",
                                                                                    cache_folder=None)

outputs = inference_function(torch_inputs)
prediction = torch.argmax(outputs, dim=1).long()
plot_inference_set(torch_inputs, torch_targets, prediction, torch_permanent_water, transform)