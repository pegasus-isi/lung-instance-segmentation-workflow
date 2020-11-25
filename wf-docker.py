#!/usr/bin/env python
import logging
from pathlib import Path
from Pegasus.api import *
import glob
import os
import pandas as pd
logging.basicConfig(level = logging.DEBUG)

#Properties
props = Properties()
props["dagman.retry"] = "100"
props["pegasus.transfer.arguments"] = "-m 1"
props.write()


# import os
# os.environ['KAGGLE_USERNAME'] = "vedula"
# os.environ['KAGGLE_KEY'] = "482a5c14ced45f63f3698eacb8fa0c62"

# import kaggle
# kaggle.api.dataset_download_files('nikhilpandey360/chest-xray-masks-and-labels/download', path='.', unzip=True)

tc = TransformationCatalog()

unet_wf = Container(
                "unet_wf",
                Container.DOCKER,
                image=str(Path(".").resolve()/"unet_cont.tar"),
                image_site="local"
            )


preprocess = Transformation(
                "preprocess",
                site="condorpool",
                pfn="/usr/bin/preprocess.py",
                is_stageable=False,
                container=unet_wf
            )

# preprocess.add_condor_profile(requirements = 'HAS_SINGULARITY == True')
# preprocess.add_profiles(Namespace.CONDOR, key="+SingularityImage", value='"/cvmfs/singularity.opensciencegrid.org/vedularaghu/unet_wf:latest"')


data_split = Transformation(
                "data_split",
                site="condorpool",
                pfn="/usr/bin/data_split.py",
                is_stageable=False,
                container=unet_wf
            )

# data_split.add_condor_profile(requirements = 'HAS_SINGULARITY == True')
# data_split.add_profiles(Namespace.CONDOR, key="+SingularityImage", value='"/cvmfs/singularity.opensciencegrid.org/vedularaghu/unet_wf:latest"')

train_model = Transformation( 
                "train_model",
                site="condorpool",
                pfn="/usr/bin/train_model.py",
                is_stageable=False,
                container=unet_wf
            )

# train_model.add_condor_profile(requirements = 'HAS_SINGULARITY == True')
# train_model.add_profiles(Namespace.CONDOR, key="+SingularityImage", value='"/cvmfs/singularity.opensciencegrid.org/vedularaghu/unet_wf:latest"')


tc.add_containers(unet_wf)
tc.add_transformations(preprocess, data_split, train_model)
tc.write()

file_list = []
output_list = []

rc = ReplicaCatalog()

for file in glob.glob("./train_images/*.png"):
    f = file.replace("./train_images/", '')
    file_list.append(File(f))
    rc.add_replica("local", File(f), Path("./train_images/").resolve() / f)
    
for file in glob.glob("./train_masks/*.png"):
    f = file.replace("./train_masks/", '')
    file_list.append(File(f))
    rc.add_replica("local", File(f), Path("./train_masks/").resolve() / f)
    
for file in glob.glob("./test/*.png"):
    f = file.replace("./test/", '')
    file_list.append(File(f))
    rc.add_replica("local", File(f), Path("./test/").resolve() / f)
    

    
checkpoint_file = "study_checkpoint.pkl"
if not os.path.isfile(checkpoint_file):
    df = pd.DataFrame(list())
    df.to_pickle(checkpoint_file)

rc.add_replica("local", checkpoint_file, Path(".").resolve() / checkpoint_file)
    
rc.write()

for filename in glob.glob("./train_images/*.png"):
    f = filename.replace("./train_images/", '').strip(".png")+"_norm.png"
    output_list.append(File(f))


for filename in glob.glob("./train_masks/*.png"):
    f = filename.replace("./train_masks/", '').strip(".png")+"_norm.png"
    output_list.append(File(f))

for filename in glob.glob("./test/*.png"):
    f = filename.replace("./test/", '').strip(".png")+"_norm.png"
    output_list.append(File(f))

        
wf = Workflow("preprocess")
    
job_preprocess = Job(preprocess).add_inputs(*file_list).add_outputs(*output_list)\

data_split_file = File("data_split.pkl")

job_data_split = Job(data_split).add_inputs(*output_list).add_outputs(data_split_file)

model = File("model.h5")


job_train = Job(train_model)\
            .add_checkpoint(File(checkpoint_file), stage_out=True)\
            .add_inputs(*output_list, data_split_file)\
            .add_outputs(model)


wf.add_jobs(job_preprocess, job_data_split, job_train)

try:
    wf.plan(submit=True)\
      .wait().analyze()\
      .statistics()
except PegasusClientError as e:
    print(e.output)  





