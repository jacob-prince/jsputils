import numpy as np
import xarray as xr

def get_ncsnr(resps, img_id, uimgid, nreps = 2):
    '''
        Calculate Noise Ceiling using methods of NSD paper.
          - If conditions have vastly different numbers of repeats, then use the minimum.
              (i.e. the num repeats of the condition with the fewest repeats).

        Arguments:
          - resps: (size: num_trials x num_channels) matrix of responses (prior to trial-averaging).
          - img_id: (length: num_trials) vector of image id shown on each trial.
          - uimgid: (length: num_images) vector of all unique images.
    '''
    arr = xr.DataArray(resps, dims = ('presentation', 'channel'), coords={'image_id': ('presentation', img_id)})
    num_reps = arr.groupby('image_id').count().loc[{'image_id': uimgid}].data[:,0]

    # Exclude conditions with less than 2 trials
    valid_uimgid = uimgid[num_reps >= nreps]
    num_reps_valid = num_reps[num_reps >= nreps]
    
    # Use only the minimum number of repeats for all conditions for fairness.
    # (i.e. if some conditions have more repeats, subsample trials)
    min_trials = np.min(num_reps_valid)

    def trim_trials(da, min_trials):
        return da.isel(presentation=np.random.choice(da.image_id.size, min_trials, replace=False))

    arr_trim = arr.sel(presentation=arr['image_id'].isin(valid_uimgid)).groupby('image_id').apply(trim_trials, min_trials=min_trials)
    
    print(arr_trim.shape)

    # Noise variance is the variance across repetitions
    noise_sd = np.sqrt(np.mean(np.power(arr_trim.groupby('image_id').std(ddof=1).loc[{'image_id': valid_uimgid}].data, 2), axis=0))
    
    # Total variance is variance across all trials
    total_var = np.power(np.std(resps, axis=0), 2)
    
    # Calculate signal variance by total variance minus noise variance.
    signal_var = total_var - np.power(noise_sd,2)
    signal_var = np.maximum(0, signal_var) # rectification
    signal_sd = np.sqrt(signal_var)
    
    # NCSNR is ratio is signal SD to noise SD.
    ncsnr = signal_sd / noise_sd
    noise_ceiling = 100*(np.power(ncsnr,2) / (np.power(ncsnr,2)+(1/min_trials)))

    return noise_sd, total_var, signal_var, ncsnr, noise_ceiling