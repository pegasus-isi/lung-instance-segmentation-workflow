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
    "--lung-img-dir",
    default=Path(__file__).parent / "inputs/train_images",
    help="Path to directory containing lung images for training and validation"
)
parser.add_argument(
    "--lung-mask-img-dir",
    default=Path(__file__).parent / "inputs/train_masks",
    help="Path to directory containing lung mask images for training and validation"
)
parser.add_argument(
    "--test-img-dir",
    default=Path(__file__).parent / "inputs/test_images",
    help="Path to directory containing test lung images."
)
# parser.add_argument(
#     "--num-process-jobs",
#     default=1,
#     type=int,
#     help="""Number of pre-processing jobs. Input files are divided evenly amongst
#     the number of pre-processing jobs. If this value exceeds the number input files
#     extra jobs will not be added. If the number of input files cannot be divided evenly
#     amongs each process job, one process job will be assigned extra files to process.
#     For example, given 3 input files, and num-process-jobs=2, 2 process jobs will
#     be created where one job gets a single file and the other job gets 2 files."""
# )
args = parser.parse_args(sys.argv[1:])
# if args.num_process_jobs < 1:
#     raise ValueError("--num-process-jobs must be >= 1")

LUNG_IMG_DIR = Path(args.lung_img_dir)
LUNG_MASK_IMG_DIR = Path(args.lung_mask_img_dir)
TEST_IMG_DIR = Path(args.test_img_dir)

# --- Write Properties ---------------------------------------------------------
props = Properties()
#props["dagman.retry"] = "100"
props["pegasus.mode"] = "development"
props.write()



# --- Write TransformationCatalog ----------------------------------------------
tc = TransformationCatalog()

# all jobs to be run in container
unet_wf_cont = Container(
                "unet_wf",
                Container.DOCKER,
                image="docker:///aditi1208/lung-segmentation:latest"
            )

tc.add_containers(unet_wf_cont)

preprocess = Transformation(
                "preprocess",
                site="condorpool",
                pfn="/usr/bin/preprocess.py",
                is_stageable=False,
                container=unet_wf_cont
            )
'''
unet_class = Transformation(
		"unet"
		site="condorpool"
		pfn="/usr/bin/unet.py",
		is_stageable=False,
		container=unet_wf_cont
	)
'''
hpo_task = Transformation( 
                "hpo",
                site="condorpool",
                pfn="/usr/bin/hpo.py",
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

tc.add_transformations(preprocess, hpo_task, train_model, predict_masks)

log.info("writing tc with transformations: {}, containers: {}".format([k for k in tc.transformations], [k for k in tc.containers]))
tc.write()

# --- Write ReplicaCatalog -----------------------------------------------------
training_input_files = []
mask_files = []
# test_input_files = []

rc = ReplicaCatalog()

for _dir, _list in [
        (LUNG_IMG_DIR, training_input_files), 
         (LUNG_MASK_IMG_DIR, mask_files), 
#         (TEST_IMG_DIR, test_input_files)
    ]:
    for f in _dir.iterdir():
        if f.name.endswith(".png"):
            _list.append(File(f.name))
            rc.add_replica(site="local", lfn=f.name, pfn=f.resolve())

# hpo job checkpoint file (empty one should be given if none exists)
p = Path(__file__).parent.resolve() / "study_checkpoint.pkl"
if not p.exists():
    df = pd.DataFrame(list())
    df.to_pickle(p.name)


checkpoint = File(p.name)
rc.add_replica(site="local", lfn=checkpoint, pfn=p.resolve())

log.info("writing rc with {} files collected from: {}".format(len(training_input_files)+len(mask_files), [LUNG_IMG_DIR, LUNG_MASK_IMG_DIR]))
rc.write()

# --- Generate and run Workflow ------------------------------------------------
wf = Workflow("lung-instance-segmentation-wf")

# all input files to be processed
# input_files = training_input_files + test_input_files

# create at most len(input_files) number of process jobs
# num_process_jobs = min(args.num_process_jobs, len(input_files))

# create the preproces jobs
process_jobs = [Job(preprocess).add_args("--type", group) for group in ["train", "val", "test"]]
processed_training_files = []
processed_val_files = []
processed_test_files = []
l = len(training_input_files)

for i, f in enumerate(training_input_files):
    if i+1 <= 0.7*l:
        process_jobs[0].add_inputs(f)
        op_file = File("train_"+f.lfn.replace(".png", "_norm.png"))
        process_jobs[0].add_outputs(op_file)
        processed_training_files.append(op_file)
    elif i+1 <= 0.9*l:
        process_jobs[1].add_inputs(f)
        op_file = File("val_"+f.lfn.replace(".png", "_norm.png"))
        process_jobs[1].add_outputs(op_file)
        processed_val_files.append(op_file)
    else:
        process_jobs[2].add_inputs(f)
        op_file = File("test_"+f.lfn.replace(".png", "_norm.png"))
        process_jobs[2].add_outputs(op_file)
        processed_test_files.append(op_file)

wf.add_jobs(*process_jobs)
log.info("generated 3 preprocess jobs")

# files to be used for training/valid (lung imgs w/mask imgs)
# processed_training_files = [File(f.lfn.replace(".png", "_norm.png")) for f in training_input_files]

# files to be used for prediction
# processed_test_files = [File(f.lfn.replace(".png", "_norm.png")) for f in test_input_files]

#creating hpo job
log.info("generating hpo job")
study = File("study_checkpoint.pkl")
hpo_job = Job(hpo_task)\
                .add_inputs(*processed_training_files, *processed_val_files, *processed_test_files, *mask_files)\
                .add_outputs(study)
#                 .add_checkpoint(checkpoint)

wf.add_jobs(hpo_job)

# create training job
log.info("generating train_model job")
model = File("model.h5")
train_job = Job(train_model)\
                .add_inputs(study, *processed_training_files, *processed_val_files, *processed_test_files, *mask_files)\
                .add_outputs(model)\
#                .add_checkpoint(model)

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
#wf.write()
wf.plan(submit=True, dir="runs")\
    .wait()\
    .analyze()\
    .statistics()


#wf.graph(include_files=True, label="xform-id", output="wf.dot")
