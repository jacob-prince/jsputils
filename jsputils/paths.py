def base():
    return '/home/jovyan/work' 

def nsd():
    return f'{base()}/DataLocal-w/NSD'

def nsd_primate_data():
    return f'{base()}/DropboxSandbox/NSD-Primate'

def nsd_stimuli():
    return f'{nsd()}/nsddata_stimuli/stimuli/nsd'

def full_coco_annots():
    return f'{base()}/DataLocal-w/COCO/annotations'

def nsd_coco_annots():
    return f'{base()}/DropboxProjects/DNFFA/PROJECT_DNFFA/STIMULUS_SETS/NSD_COCO_annotations'

def cifar10():
    return f'{base()}/DataLocal-w/ffcv-cifar10/'

def pycortex_db_NSD():
    return f'{base()}//DataLocal-w/pycortex_db_NSD'

def ffcv_imagenet1k_trainset():
    return f'{base()}/DataLocal-ro/imagenet1k-ffcv/imagenet1k_train_jpg_q100_s256_lmax512_crop_includes_index.ffcv'

def ffcv_imagenet1k_valset():
    return f'{base()}/DataLocal-ro/imagenet1k-ffcv/imagenet1k_val_jpg_q100_s256_lmax512_crop_includes_index.ffcv'

def training_checkpoint_dir():
    return f'{base()}/DropboxProjects/DNFFA/PROJECT_DNFFA/EXPERIMENTS_MODELS/checkpoints'

def encoding_output_dir():
    return f'{base()}/DataLocal-w/NSD_encoding_models'    

def image_set_dir():
    return f'{base()}/DropboxProjects/DNFFA/PROJECT_DNFFA/STIMULUS_SETS'

def selective_unit_dir():
    return f'{base()}/DropboxProjects/DNFFA/PROJECT_DNFFA/EXPERIMENTS_MODELS/selective_units'

def figure_savedir():
    return f'{base()}/DropboxProjects/DROPOUT/PROJECT_DROPOUT/NOTEBOOKS/figure_outputs'

def weight_savedir():
    return f'{base()}/DropboxProjects/DNFFA/PROJECT_DNFFA/EXPERIMENTS_MODELS/weights'

def dnn_dropout_weightdir():
    return f'{base()}/DataExactitude-w/dnn-dropout'

def brain_region_savedir():
    return f'{base()}/DataLocal-w/NSD_processed_data'  

def imagenet_valdir():
    return f'/{base()}/DataLocal-ro/imagenet1k-orig/val'
    
