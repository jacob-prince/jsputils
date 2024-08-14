import cortex
import numpy as np
import matplotlib.pyplot as plt
from jsputils import nnutils, nsdorg
from mpl_toolkits.mplot3d import axes3d, art3d
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
              'scrambled':'navy',
             'scrambled-objects':'navy',
             'scrambled-words':'darkgray'}

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

def pca_plot(all_pcs, ax, title = None, cs = ['k'], ss = [1], als = [1], rot = 0, lim = None, elv = 20):

    for p, pcs in enumerate(all_pcs):
        x, y, z = pcs[:,0], pcs[:,1], pcs[:,2]

        ax.scatter3D(xs = x, ys = y, zs = z, c = cs[p],
                     s = ss[p], alpha = als[p])

    # remove axis labels
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    # remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    # remove axis lines
    ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

    # transparent background
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    # remove background entirely
    ax.set_facecolor((1.0, 1.0, 1.0, 0.0))

    def lims(mplotlims):
        scale = 1.021
        offset = (mplotlims[1] - mplotlims[0])*scale
        return mplotlims[1] - offset, mplotlims[0] + offset

    xlims, ylims, zlims = lims(ax.get_xlim()), lims(ax.get_ylim()), lims(ax.get_zlim())
    i = np.array([xlims[0], ylims[0], zlims[0]])
    f = np.array([xlims[0], ylims[1], zlims[0]])
    p = art3d.Poly3DCollection(np.array([[i, f]]))
    p.set_color('black')
    ax.add_collection3d(p)

    i = np.array([xlims[0], ylims[1], zlims[0]])
    f = np.array([xlims[0], ylims[1], zlims[1]])
    p = art3d.Poly3DCollection(np.array([[i, f]]))
    p.set_color('black')
    ax.add_collection3d(p)

    i = np.array([xlims[0], ylims[1], zlims[0]])
    f = np.array([xlims[1], ylims[1], zlims[0]])
    p = art3d.Poly3DCollection(np.array([[i, f]]))
    p.set_color('black')
    ax.add_collection3d(p)
    
    if lim:
        ax.set_xlim([-lim,lim])
        ax.set_ylim([-lim,lim])
        ax.set_zlim([-lim,lim])
        
    # Adjust text positions based on viewing angle
    if 0 <= rot < 90:
        # Adjust for first quadrant view
        ax.text(xlims[0], ylims[0], zlims[0], 'PC2', ha='right', va='top')
        ax.text(xlims[1], ylims[1], zlims[0], 'PC1', ha='left', va='top')
        ax.text(xlims[0], ylims[1], zlims[1], 'PC3', ha='center', va='bottom')
    elif 90 <= rot < 180:
        ax.text(xlims[0], ylims[0], zlims[0], 'PC2', ha='right', va='top')
        ax.text(xlims[1], ylims[1], zlims[0], 'PC1', ha='left', va='top')
        ax.text(xlims[0], ylims[1], zlims[1], 'PC3', ha='center', va='bottom')
    elif 180 <= rot < 270:
        ax.text(xlims[0], ylims[0], zlims[0], 'PC2', ha='right', va='top')
        ax.text(xlims[1], ylims[1], zlims[0], 'PC1', ha='left', va='top')
        ax.text(xlims[0], ylims[1], zlims[1], 'PC3', ha='center', va='bottom')
    else:
        ax.text(xlims[0], ylims[0], zlims[0], 'PC2', ha='right', va='top')
        ax.text(xlims[1], ylims[1], zlims[0], 'PC1', ha='left', va='top')
        ax.text(xlims[0], ylims[1], zlims[1], 'PC3', ha='center', va='bottom')

    ax.view_init(elev=elv, azim=rot)

    plt.gca().grid(False)
    if title:
        plt.title(title)