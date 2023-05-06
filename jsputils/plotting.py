import cortex
import numpy as np
import matplotlib.pyplot as plt
from jsputils import nnutils, nsdorg
from IPython.core.debugger import set_trace

def plot_ROI_flatmap(subj, space, roi_group, included_voxels, mapper='nearest',vmin=0,vmax=1,cmap='magma',colorbar=True):

    if space == 'nativesurface':
        plot_data = cortex.Vertex(included_voxels, subj, cmap=cmap,
                                        vmin=vmin,
                                        vmax=vmax,recache=True)
    elif space == 'func1pt8mm':
        subj_dims = nsdorg.get_subj_dims(subj)
        included_voxels = np.swapaxes(included_voxels.reshape(subj_dims),0,2)
        plot_data = cortex.Volume(included_voxels, subj, xfmname='func1pt8_to_anat0pt8_autoFSbbr', cmap=cmap,vmin=vmin,vmax=vmax)

    plt.figure()
    cortex.quickshow(plot_data,with_rois=False,with_labels=False,with_curvature=True,
                     curvature_contrast=0.3,
                     curvature_brightness=0.8,
                     curvature_threshold=True,
                     with_colorbar=colorbar,
                     recache=False)
    plt.title(roi_group,fontsize=44)
    plt.show()
    
    return plot_data

def plot_selective_unit_props(selective_unit_dict):
    
    domains = selective_unit_dict['floc_domains']
    model_name = selective_unit_dict['model_name']
    floc_imageset = selective_unit_dict['floc_imageset']
    FDR_p = selective_unit_dict['FDR_p']
    
    target_layers = list(selective_unit_dict[domains[0]].keys())

    colors = {'faces':'tomato',
              'bodies':'dodgerblue',
              'objects':'orange',
              'scenes':'limegreen',
              'characters':'purple',
              'scrambled':'navy'}

    floc_colors = [x[1] for x in colors]
    floc_domains = [x[0] for x in colors]
    
    plt.figure(figsize=(8,4))

    for domain in domains:
        domain_props = []
        
        for i in range(len(target_layers)):
            domain_props.append(np.mean(selective_unit_dict[domain][target_layers[i]]['mask']))

        plt.plot(domain_props,label=domain,color=colors[domain]);

    plt.xticks(np.arange(len(target_layers)),np.array(target_layers),rotation=90);
    plt.title(f'proportion of domain-selective units by layer (FDR_p = {FDR_p})\nmodel: {model_name}\nfloc set: {floc_imageset}')
    plt.grid('on')
    # get rid of the frame
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    plt.ylim([0,0.6])
    plt.legend()
    plt.show()
    
    return
