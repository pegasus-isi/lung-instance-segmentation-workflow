#!/usr/bin/env python3
import logging as log
import math
import sys
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

from Pegasus.api import *

log.basicConfig(level=log.DEBUG)

# --- Get input files ----------------------------------------------------------
parser = ArgumentParser(description="Generates and runs lung instance segmentation workflow")
parser.add_argument(
    "lung_img_dir",
    help="Path to directory containing lung images for training and validation"
)
parser.add_argument(
    "lung_mask_img_dir",
    help="Path to directory containing lung mask images for training and validation"
)
parser.add_argument(
    "test_img_dir",
    help="Path to directory containing test lung images."
)
parser.add_argument(
    "--num-process-jobs",
    default=1,
    type=int,
    help="Number of pre-processing jobs. Input files are divided evenly amongst"
    "the number of pre-processing jobs. If this value exceeds the number input files"
    "extra jobs will not be added".
)
args = parser.parse_args(sys.argv[1:])
if args.num_process_jobs < 1:
    raise ValueError("--num-process-jobs must be >= 1")

LUNG_IMG_DIR = Path(args.lung_img_dir)
LUNG_MASK_IMG_DIR = Path(args.lung_mask_img_dir)
TEST_IMG_DIR = Path(args.test_img_dir)

# --- Write Properties ---------------------------------------------------------
props = Properties()
#props["dagman.retry"] = "100"
props["pegasus.mode"] = "development"

log.info("writing properties: {}".format(props._props))
props.write()



# --- Write TransformationCatalog ----------------------------------------------
tc = TransformationCatalog()

# all jobs to be run in container
unet_wf_cont = Container(
                "unet_wf",
                Container.DOCKER,
                image="docker://vedularaghu/unet_wf:latest"
            )

tc.add_containers(unet_wf_cont)

preprocess = Transformation(
                "preprocess",
                site="condorpool",
                pfn="/usr/bin/preprocess.py",
                is_stageable=False,
                container=unet_wf_cont
            )

train_model = Transformation( 
                "train_model",
                site="condorpool",
                pfn="/usr/bin/train_model.py",
                is_stageable=False,
                container=unet_wf_cont
            )

predict_masks = Transformation( 
                "predict_masks",
                site="condorpool",
                pfn="/usr/bin/prediction.py",
                is_stageable=False,
                container=unet_wf_cont
            )

tc.add_transformations(preprocess, train_model, predict_masks)

log.info("writing tc with transformations: {}, containers: {}".format([k for k in tc.transformations], [k for k in tc.containers]))
tc.write()

# --- Write ReplicaCatalog -----------------------------------------------------
training_input_files = []
test_input_files = []

rc = ReplicaCatalog()

for _dir, _list in [
        (LUNG_IMG_DIR, training_input_files), 
        (LUNG_MASK_IMG_DIR, training_input_files), 
        (TEST_IMG_DIR, test_input_files)
    ]:
    for f in _dir.iterdir():
        if f.name.endswith(".png"):
            _list.append(File(f.name))
            rc.add_replica(site="local", lfn=f.name, pfn=f.resolve())

# train job checkpoint file (empty one should be given if none exists)
p = Path(__file__).parent.resolve() / "study_checkpoint.pkl"
if not p.exists():
    df = pd.DataFrame(list())
    df.to_pickle(p.name)


checkpoint = File(p.name)
rc.add_replica(site="local", lfn=checkpoint, pfn=p.resolve())

log.info("writing rc with {} files collected from: {}".format(len(training_input_files) + len(test_input_files), [LUNG_IMG_DIR, LUNG_MASK_IMG_DIR, TEST_IMG_DIR]))
rc.write()

# --- Generate and run Workflow ------------------------------------------------
wf = Workflow("lung-instance-segmentation-wf")

# all input files to be processed
input_files = training_input_files + test_input_files

# create num-process-jobs number of jobs, where each job gets an equal number
# of files to process (with the exception of the last job in some cases)
num_img_per_job = len(input_files) // args.num_process_jobs
# create at most len(input_files) number of process jobs
if num_img_per_job < 1:
    num_img_per_job = 1
start = 0
end = start + num_img_per_job

log.info("assigning {} input files to each of the {} process job".format(num_img_per_job, args.num_process_jobs))

while start < len(input_files):
    job_in_files = input_files[start:end]
    job_out_files = [File(f.lfn.replace(".png", "_norm.png")) for f in job_in_files]

    process_job = Job(preprocess)\
                    .add_inputs(*job_in_files)\
                    .add_outputs(*job_out_files)
    
    wf.add_jobs(process_job)

log.info("{} process jobs generated".format(len(wf.jobs)))

# files to be used for training/valid (lung imgs w/mask imgs)
processed_training_files = [File(f.lfn.replace(".png", "_norm.png")) for f in training_input_files]

# files to be used for prediction
processed_test_files = [File(f.lfn.replace(".png", "_norm.png")) for f in test_input_files]

# create training job
log.info("generating train_model job")
model = File("model.h5")
train_job = Job(train_model)\
                .add_inputs(*processed_training_files)\
                .add_outputs(model)\
                .add_checkpoint(checkpoint)

wf.add_jobs(train_job)

# create mask prediction job
log.info("generating prediction job; using {} test lung images".format(len(processed_test_files)))
predicted_masks = [File(f.lfn.replace(".png", "_mask.png")) for f in processed_test_files]
predict_job = Job(predict_masks)\
                .add_inputs(model, *processed_test_files)\
                .add_outputs(*predicted_masks)

wf.add_jobs(predict_job)

# run workflow
log.info("begin workflow execution")
wf.plan(submit=True, dir="runs")\
    .wait()\
    .analyze()\
    .statistics()