#!/usr/bin/env python3
import logging as log
import math
import sys, os
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import json
import random
import numpy as np


from Pegasus.api import *

log.basicConfig(level=log.DEBUG)
IGNORE_IMAGES = ['CHNCXR_0025_0.png', 'CHNCXR_0036_0.png', 'CHNCXR_0037_0.png', 'CHNCXR_0038_0.png', 'CHNCXR_0039_0.png', 'CHNCXR_0040_0.png', 'CHNCXR_0065_0.png', 'CHNCXR_0181_0.png', 'CHNCXR_0182_0.png', 'CHNCXR_0183_0.png', 'CHNCXR_0184_0.png', 'CHNCXR_0185_0.png', 'CHNCXR_0186_0.png', 'CHNCXR_0187_0.png', 'CHNCXR_0188_0.png', 'CHNCXR_0189_0.png', 'CHNCXR_0190_0.png', 'CHNCXR_0191_0.png', 'CHNCXR_0192_0.png', 'CHNCXR_0193_0.png', 'CHNCXR_0194_0.png', 'CHNCXR_0195_0.png', 'CHNCXR_0196_0.png', 'CHNCXR_0197_0.png', 'CHNCXR_0198_0.png', 'CHNCXR_0199_0.png', 'CHNCXR_0200_0.png', 'CHNCXR_0201_0.png', 'CHNCXR_0202_0.png', 'CHNCXR_0203_0.png', 'CHNCXR_0204_0.png', 'CHNCXR_0205_0.png', 'CHNCXR_0206_0.png', 'CHNCXR_0207_0.png', 'CHNCXR_0208_0.png', 'CHNCXR_0209_0.png', 'CHNCXR_0210_0.png', 'CHNCXR_0211_0.png', 'CHNCXR_0212_0.png', 'CHNCXR_0213_0.png', 'CHNCXR_0214_0.png', 'CHNCXR_0215_0.png', 'CHNCXR_0216_0.png', 'CHNCXR_0217_0.png', 'CHNCXR_0218_0.png', 'CHNCXR_0219_0.png', 'CHNCXR_0220_0.png', 'CHNCXR_0336_1.png', 'CHNCXR_0341_1.png', 'CHNCXR_0342_1.png', 'CHNCXR_0343_1.png', 'CHNCXR_0344_1.png', 'CHNCXR_0345_1.png', 'CHNCXR_0346_1.png', 'CHNCXR_0347_1.png', 'CHNCXR_0348_1.png', 'CHNCXR_0349_1.png', 'CHNCXR_0350_1.png', 'CHNCXR_0351_1.png', 'CHNCXR_0352_1.png', 'CHNCXR_0353_1.png', 'CHNCXR_0354_1.png', 'CHNCXR_0355_1.png', 'CHNCXR_0356_1.png', 'CHNCXR_0357_1.png', 'CHNCXR_0358_1.png', 'CHNCXR_0359_1.png', 'CHNCXR_0360_1.png', 'CHNCXR_0481_1.png', 'CHNCXR_0482_1.png', 'CHNCXR_0483_1.png', 'CHNCXR_0484_1.png', 'CHNCXR_0485_1.png', 'CHNCXR_0486_1.png', 'CHNCXR_0487_1.png', 'CHNCXR_0488_1.png', 'CHNCXR_0489_1.png', 'CHNCXR_0490_1.png', 'CHNCXR_0491_1.png', 'CHNCXR_0492_1.png', 'CHNCXR_0493_1.png', 'CHNCXR_0494_1.png', 'CHNCXR_0495_1.png', 'CHNCXR_0496_1.png', 'CHNCXR_0497_1.png', 'CHNCXR_0498_1.png', 'CHNCXR_0499_1.png', 'CHNCXR_0500_1.png', 'CHNCXR_0502_1.png', 'CHNCXR_0505_1.png', 'CHNCXR_0560_1.png', 'CHNCXR_0561_1.png', 'CHNCXR_0562_1.png', 'CHNCXR_0563_1.png', 'CHNCXR_0564_1.png', 'CHNCXR_0565_1.png']

# --- Get input files ----------------------------------------------------------
parser = ArgumentParser(description="Generates and runs lung instance segmentation workflow")
parser.add_argument(
    "--lung-img-dir",
    default=Path(__file__).parent / "img/lung-images",
    help="Path to directory containing lung images for training and validation"
)
parser.add_argument(
    "--lung-mask-img-dir",
    default=Path(__file__).parent / "img/lung-masks",
    help="Path to directory containing lung mask images for training and validation"
)

def train_test_val_split(preprocess, training_input_files, mask_files, 
        processed_training_files, processed_val_files, processed_test_files):
    np.random.seed(4)
    process_jobs = [Job(preprocess).add_args("--type", group) for group in ["train", "val", "test"]]
    augmented_masks = []
    random.shuffle(training_input_files)

    l = len(training_input_files)
    for i, f in enumerate(training_input_files):
        if i+1 <= 0.7*l:
            process_jobs[0].add_inputs(f)
            op_file1 = File("train_"+f.lfn.replace(".png", "_norm.png"))
            op_file2 = File("train_"+f.lfn.replace(".png", "_0_norm.png"))
            op_file3 = File("train_"+f.lfn.replace(".png", "_1_norm.png"))
            op_mask1 = File(f.lfn.replace(".png", "_0_mask.png"))
            op_mask2 = File(f.lfn.replace(".png", "_1_mask.png"))
            process_jobs[0].add_outputs(op_file1, op_file2, op_file3, op_mask1, op_mask2)
            augmented_masks.extend([op_mask1, op_mask2])
            processed_training_files.extend([op_file1, op_file2, op_file3])
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

    for preprocess_job in process_jobs:
        preprocess_job.add_inputs(*mask_files)
        
    mask_files.extend(augmented_masks)
    return process_jobs

def run_workflow():

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
                    image="docker:///aditi1208/lung-segmentation:latest",
                    arguments="--runtime=nvidia --shm-size=15gb"
                )

    tc.add_containers(unet_wf_cont)

    top_dir = Path(__file__).parent.resolve()

    preprocess = Transformation(
                    "preprocess",
                    site="local",
                    pfn=top_dir / "bin/preprocess.py",
                    is_stageable=True,
                    container=unet_wf_cont
                )

    unet = Transformation(
                    "unet",
                    site="local",
                    pfn=top_dir / "bin/unet.py",
                    is_stageable=True,
                    container=unet_wf_cont
                )
    utils = Transformation(
                    "utils",
                    site="local",
                    pfn=top_dir / "bin/utils.py",
                    is_stageable=True,
                    container=unet_wf_cont
                )

    hpo_task = Transformation( 
                    "hpo",
                    site="local",
                    pfn=top_dir / "bin/hpo.py",
                    is_stageable=True,
                    container=unet_wf_cont
                )
    hpo_task.add_requirement(unet)

    train_model = Transformation( 
                    "train_model",
                    site="local",
                    pfn=top_dir / "bin/train_model.py",
                    is_stageable=True,
                    container=unet_wf_cont
                )
    train_model.add_requirement(unet)
    train_model.add_requirement(utils)

    predict_masks = Transformation( 
                    "predict_masks",
                    site="local",
                    pfn=top_dir / "bin/prediction.py",
                    is_stageable=True,
                    container=unet_wf_cont
                )
    predict_masks.add_requirement(unet)

    evaluate_model = Transformation( 
                    "evaluate",
                    site="local",
                    pfn=top_dir / "bin/evaluate.py",
                    is_stageable=True,
                    container=unet_wf_cont
                )
    evaluate_model.add_requirement(unet)

    tc.add_transformations(preprocess, hpo_task, train_model, predict_masks, evaluate_model, unet, utils)

    log.info("writing tc with transformations: {}, containers: {}".format([k for k in tc.transformations], [k for k in tc.containers]))
    tc.write()

    # --- Write ReplicaCatalog -----------------------------------------------------
    training_input_files = []
    mask_files = []

    rc = ReplicaCatalog()

    for _dir, _list in [
            (LUNG_IMG_DIR, training_input_files), 
            (LUNG_MASK_IMG_DIR, mask_files), 
        ]:
        for f in _dir.iterdir():
            if f.name.endswith(".png"):
                if f.name in IGNORE_IMAGES: continue
                _list.append(File(f.name))
                rc.add_replica(site="local", lfn=f.name, pfn=f.resolve())

    #checkpoint files  and results (empty one should be given if none exists) 
    for fname in ["study_checkpoint.pkl", "model.h5", "bin/unet.py", "bin/utils.py"]:
        p = Path(__file__).parent.resolve() / fname
        if not p.exists():
            with open(p, "w") as dummyFile:
                dummyFile.write("")

        replicaFile = File(p.name)
        rc.add_replica(site="local", lfn=replicaFile, pfn=p)

    log.info("writing rc with {} files collected from: {}".format(len(training_input_files)+len(mask_files), [LUNG_IMG_DIR, LUNG_MASK_IMG_DIR]))
    rc.write()

    # --- Generate and run Workflow ------------------------------------------------
    wf = Workflow("lung-instance-segmentation-wf")

    #create preprocess job
    processed_training_files = []
    processed_val_files = []
    processed_test_files = []
    process_jobs = train_test_val_split(preprocess, training_input_files, mask_files,
     processed_training_files, processed_val_files, processed_test_files)
    wf.add_jobs(*process_jobs)
    log.info("generated 3 preprocess jobs")


    #create hpo job
    log.info("generating hpo job")
    hpo_checkpoint_result = File("study_checkpoint.pkl")
    study_result = File("study_results.txt")
    unet_file = File("unet.py")
    hpo_job = Job(hpo_task)\
                    .add_inputs(*processed_training_files, *processed_val_files, *mask_files, unet_file)\
                    .add_outputs(study_result)\
                    .add_checkpoint(hpo_checkpoint_result)\
                    .add_profiles(Namespace.DAGMAN, key="retry", value=1000)\
                    .add_profiles(Namespace.PEGASUS, key="checkpoint.time", value=10)\
                    .add_profiles(Namespace.PEGASUS, key="maxwalltime", value=11)

    wf.add_jobs(hpo_job)

    # create training job
    log.info("generating train_model job")
    model = File("model.h5")
    model_copy = File("model_copy.h5")
    utils_file = File("utils.py")
    train_job = Job(train_model)\
                    .add_inputs(study_result, *processed_training_files, *processed_val_files, *mask_files, unet_file, utils_file)\
                    .add_checkpoint(model)\
                    .add_outputs(model_copy)

    wf.add_jobs(train_job)

    # create mask prediction job
    log.info("generating prediction job; using {} test lung images".format(len(processed_test_files)))
    predicted_masks = [File("pred_"+f.lfn.replace(".png", "_mask.png")[5:]) for f in processed_test_files]
    predict_job = Job(predict_masks)\
                    .add_inputs(model_copy, *processed_test_files, unet_file)\
                    .add_outputs(*predicted_masks)

    wf.add_jobs(predict_job)

    #create evalute job
    evaluate_job = Job(evaluate_model)\
                    .add_inputs(*processed_training_files, *processed_test_files, *predicted_masks, *mask_files, unet_file)\
                    # .add_outputs(pdf_analysis)

    wf.add_jobs(evaluate_job)

    # run workflow
    log.info("begin workflow execution")
    wf.plan(submit=True, dir="runs", verbose=3)\
        .wait()\
        .analyze()\
        .statistics()

    wf.graph(include_files=True, no_simplify=True, label="xform-id", output="graph.dot")


if __name__ == "__main__":
    global LUNG_IMG_DIR
    global LUNG_MASK_IMG_DIR
    args = parser.parse_args(sys.argv[1:])

    LUNG_IMG_DIR = Path(args.lung_img_dir)
    LUNG_MASK_IMG_DIR = Path(args.lung_mask_img_dir)
    run_workflow()
