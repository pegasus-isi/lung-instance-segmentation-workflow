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

log.basicConfig(level=log.INFO)
IGNORE_IMAGES = {'CHNCXR_0025_0.png', 'CHNCXR_0036_0.png', 'CHNCXR_0037_0.png', 'CHNCXR_0038_0.png', 'CHNCXR_0039_0.png', 'CHNCXR_0040_0.png', 'CHNCXR_0065_0.png', 'CHNCXR_0181_0.png', 'CHNCXR_0182_0.png', 'CHNCXR_0183_0.png', 'CHNCXR_0184_0.png', 'CHNCXR_0185_0.png', 'CHNCXR_0186_0.png', 'CHNCXR_0187_0.png', 'CHNCXR_0188_0.png', 'CHNCXR_0189_0.png', 'CHNCXR_0190_0.png', 'CHNCXR_0191_0.png', 'CHNCXR_0192_0.png', 'CHNCXR_0193_0.png', 'CHNCXR_0194_0.png', 'CHNCXR_0195_0.png', 'CHNCXR_0196_0.png', 'CHNCXR_0197_0.png', 'CHNCXR_0198_0.png', 'CHNCXR_0199_0.png', 'CHNCXR_0200_0.png', 'CHNCXR_0201_0.png', 'CHNCXR_0202_0.png', 'CHNCXR_0203_0.png', 'CHNCXR_0204_0.png', 'CHNCXR_0205_0.png', 'CHNCXR_0206_0.png', 'CHNCXR_0207_0.png', 'CHNCXR_0208_0.png', 'CHNCXR_0209_0.png', 'CHNCXR_0210_0.png', 'CHNCXR_0211_0.png', 'CHNCXR_0212_0.png', 'CHNCXR_0213_0.png', 'CHNCXR_0214_0.png', 'CHNCXR_0215_0.png', 'CHNCXR_0216_0.png', 'CHNCXR_0217_0.png', 'CHNCXR_0218_0.png', 'CHNCXR_0219_0.png', 'CHNCXR_0220_0.png', 'CHNCXR_0336_1.png', 'CHNCXR_0341_1.png', 'CHNCXR_0342_1.png', 'CHNCXR_0343_1.png', 'CHNCXR_0344_1.png', 'CHNCXR_0345_1.png', 'CHNCXR_0346_1.png', 'CHNCXR_0347_1.png', 'CHNCXR_0348_1.png', 'CHNCXR_0349_1.png', 'CHNCXR_0350_1.png', 'CHNCXR_0351_1.png', 'CHNCXR_0352_1.png', 'CHNCXR_0353_1.png', 'CHNCXR_0354_1.png', 'CHNCXR_0355_1.png', 'CHNCXR_0356_1.png', 'CHNCXR_0357_1.png', 'CHNCXR_0358_1.png', 'CHNCXR_0359_1.png', 'CHNCXR_0360_1.png', 'CHNCXR_0481_1.png', 'CHNCXR_0482_1.png', 'CHNCXR_0483_1.png', 'CHNCXR_0484_1.png', 'CHNCXR_0485_1.png', 'CHNCXR_0486_1.png', 'CHNCXR_0487_1.png', 'CHNCXR_0488_1.png', 'CHNCXR_0489_1.png', 'CHNCXR_0490_1.png', 'CHNCXR_0491_1.png', 'CHNCXR_0492_1.png', 'CHNCXR_0493_1.png', 'CHNCXR_0494_1.png', 'CHNCXR_0495_1.png', 'CHNCXR_0496_1.png', 'CHNCXR_0497_1.png', 'CHNCXR_0498_1.png', 'CHNCXR_0499_1.png', 'CHNCXR_0500_1.png', 'CHNCXR_0502_1.png', 'CHNCXR_0505_1.png', 'CHNCXR_0560_1.png', 'CHNCXR_0561_1.png', 'CHNCXR_0562_1.png', 'CHNCXR_0563_1.png', 'CHNCXR_0564_1.png', 'CHNCXR_0565_1.png'}

DONUT_USER_HOME = "/nas/home/jaditi"
# --- Get input files ----------------------------------------------------------
parser = ArgumentParser(description="Generates and runs lung instance segmentation workflow")
parser.add_argument(
    "--lung-img-dir",
    default=Path(__file__).parent / "data/LungSegmentation/CXR_png",
    help="Path to directory containing lung images for training and validation"
)
parser.add_argument(
    "--lung-mask-img-dir",
    default=Path(__file__).parent / "data/LungSegmentation/masks",
    help="Path to directory containing lung mask images for training and validation"
)
parser.add_argument(
    "--num-inputs",
    type=int,
    default=-1,
    help="Number of files to use as input (-1) to use the complete dataset"
)
parser.add_argument(
    "--hpo-jobs",
    type=int,
    default=1,
    help="Number of HPO Jobs you want to start simultaneously"
)
parser.add_argument(
    "--hpo-storage",
    type=str,
    default="",
    help="Storage location for parallel hpo jobs. Ex: mysql://username:pass@hostname/db"
)
parser.add_argument(
    "--gpus",
    action='store_true',
    help="Flag to request gpus on ML jobs"
)
parser.add_argument(
    "--donut",
    action='store_true',
    help="Flag to run the workflow on donut cluster"
)

top_dir = Path(__file__).parent.resolve()

def train_test_val_split(preprocess, training_input_files, mask_files, processed_training_files, processed_val_files, processed_test_files, training_masks, val_masks, test_masks, num_inputs):
    np.random.seed(4)
    process_jobs = [Job(preprocess).add_args("--type", group) for group in ["train", "val", "test"]]
    augmented_masks = []


    # --- Write ReplicaCatalog -----------------------------------------------------
    rc = ReplicaCatalog()

    # add mask images to rc
    for f in LUNG_MASK_IMG_DIR.iterdir():
        if f.name.endswith(".png"):
            if f.name in IGNORE_IMAGES:
                continue
            
            mask_files.append(File(f.name))
            rc.add_replica(site="local", lfn=f.name, pfn=f.resolve())
    
    #add an empty(probably checkpoint file
    #checkpoint files  and results (empty one should be given if none exists)
    for fname in ["inputs/checkpoints/study_checkpoint.pkl", "bin/model/unet.py", "bin/model/utils.py"]:
        p = Path(__file__).parent.resolve() / fname
        if not p.exists():
            with open(p, "w") as dummyFile:
                dummyFile.write("")
        replicaFile = File(p.name)
        rc.add_replica(site="local", lfn=replicaFile, pfn=p)

    for f in LUNG_IMG_DIR.iterdir():
        if f.name.endswith(".png") and ("mask" not in f.name.lower()) and (f.name not in IGNORE_IMAGES):
            training_input_files.append(f)

    random.shuffle(training_input_files)
    l = len(training_input_files) if num_inputs == -1 else num_inputs
    print('Length ', l)

    i = 0
    for file in training_input_files:
        if i+1 <= 0.7*l:
            f = File("train_{}".format(file.name))
            rc.add_replica(site="local", lfn=f, pfn=file.resolve()) 

            process_jobs[0].add_inputs(f)
            log.info("preprocess_train adding input {}".format(f))
            op_file1 = File(f.lfn.replace(".png", "_norm.png"))
            op_file2 = File(f.lfn.replace(".png", "_0_norm.png"))
            op_file3 = File(f.lfn.replace(".png", "_1_norm.png"))
            op_mask2 = File(file.name.replace(".png", "_0_mask.png"))
            op_mask3 = File(file.name.replace(".png", "_1_mask.png"))

            for m in mask_files:
                mname = m.lfn[0:-9]
                if file.name[0:-4] == mname:
                    training_masks.append(m)
                    break

            process_jobs[0].add_outputs(op_file1, op_file2, op_file3, op_mask2, op_mask3)
            augmented_masks.extend([op_mask2, op_mask3])
            processed_training_files.extend([op_file1, op_file2, op_file3])

        elif i+1 <= 0.9*l:
            f = File("val_{}".format(file.name))
            rc.add_replica(site="local", lfn=f, pfn=file.resolve())

            process_jobs[1].add_inputs(f)
            log.info("preprocess_val adding input {}".format(f))
            op_file = File(f.lfn.replace(".png", "_norm.png"))
            for m in mask_files:
                mname = m.lfn[0:-9]
                if file.name[0:-4] == mname:
                    val_masks.append(m)
                    break
                    
            process_jobs[1].add_outputs(op_file)
            processed_val_files.append(op_file)

        else:
            f = File("test_{}".format(file.name))
            rc.add_replica(site="local", lfn=f, pfn=file.resolve())

            process_jobs[2].add_inputs(f)
            op_file = File(f.lfn.replace(".png", "_norm.png"))
            for m in mask_files:
                mname = m.lfn[0:-9]
                if file.name[0:-4] == mname:
                    test_masks.append(m)

            process_jobs[2].add_outputs(op_file)
            log.info("preprocess_test adding input {}".format(f))
            processed_test_files.append(op_file)

        i += 1

    log.info("writing rc with {} files collected from: {}".format(len(training_input_files)+len(mask_files), [LUNG_IMG_DIR, LUNG_MASK_IMG_DIR]))
    rc.write()

    # for preprocess_job in process_jobs:
    #     preprocess_job.add_inputs(*mask_files)
    process_jobs[0].add_inputs(*training_masks)
    training_masks.extend(augmented_masks)
    return process_jobs

def create_site_catalog():
    sc = SiteCatalog()

    shared_scratch_dir = os.path.join(top_dir, "scratch")
    local_storage_dir = os.path.join(top_dir, "output")

    local = Site("local")\
                .add_directories(
                    Directory(Directory.SHARED_SCRATCH, shared_scratch_dir)
                        .add_file_servers(FileServer("file://" + shared_scratch_dir, Operation.ALL)),
                    Directory(Directory.LOCAL_STORAGE, local_storage_dir)
                        .add_file_servers(FileServer("file://" + local_storage_dir, Operation.ALL))
                )

    condorpool = Site("condorpool")\
                    .add_pegasus_profile(
                        style="condor",
                        data_configuration="condorio"
                    )\
                    .add_condor_profile(universe="vanilla")\
                    .add_profiles(Namespace.PEGASUS, key="data.configuration", value="condorio")
    
    donut = Site("donut")\
            .add_grids(
                Grid(grid_type=Grid.BATCH, scheduler_type=Scheduler.SLURM, contact="${DONUT_USER}@donut-submit01", job_type=SupportedJobs.COMPUTE),
                Grid(grid_type=Grid.BATCH, scheduler_type=Scheduler.SLURM, contact="${DONUT_USER}@donut-submit01", job_type=SupportedJobs.AUXILLARY)
            )\
            .add_directories(
                Directory(Directory.SHARED_SCRATCH, "/nas/home/${DONUT_USER}/pegasus/scratch")
                    .add_file_servers(FileServer("scp://${DONUT_USER}@donut-submit01${DONUT_USER_HOME}/pegasus/scratch", Operation.ALL)),
                Directory(Directory.SHARED_STORAGE, "/nas/home/${DONUT_USER}/pegasus/storage")
                    .add_file_servers(FileServer("scp://${DONUT_USER}@donut-submit01${DONUT_USER_HOME}/pegasus/storage", Operation.ALL))
            )\
            .add_pegasus_profile(
                style="ssh",
                data_configuration="nonsharedfs",
                change_dir="true",
                queue="donut-default",
                cores=1,
                runtime=1800
            )\
            .add_profiles(Namespace.PEGASUS, key="SSH_PRIVATE_KEY", value="/home/pegasus/.ssh/bosco_key.rsa")\
            .add_env(key="PEGASUS_HOME", value="${DONUT_USER_HOME}/${PEGASUS_VERSION}")

    sc.add_sites(local, donut, condorpool)
    return sc

def run_workflow(args):

    # --- Write Properties ---------------------------------------------------------
    props = Properties()
    props["pegasus.mode"] = "development"
    if (args.donut):
        props["pegasus.transfer.links"] = "true"
        props["pegasus.transfer.threads"] = "8"
    props.write()



    # --- Write TransformationCatalog ----------------------------------------------
    tc = TransformationCatalog()

    # all jobs to be run in container

    if (args.donut):
        unet_wf_preprocess_cont = Container(
                        "unet_wf_pre",	
	                    Container.SINGULARITY,	
	                    image=str(Path(".").parent.resolve() / "lungseg_pre.sif"),	
	                    #image="docker:///papajim/lung-segmentation:latest",	
	                    image_site="local",	
	                    mounts=["${DONUT_USER_HOME}:${DONUT_USER_HOME}"]	
	                )
        
        unet_wf_cont = Container(
                        "unet_wf_model",
	                    Container.SINGULARITY,	
	                    image=str(Path(".").parent.resolve() / "lungseg_model.sif"),	
	                    #image="docker:///papajim/lung-segmentation:latest",	
	                    image_site="local",	
	                    mounts=["${DONUT_USER_HOME}:${DONUT_USER_HOME}"]	
	                )	
    else:
        unet_wf_cont = Container(	
	                    "unet_wf_pre",	
	                    Container.SINGULARITY,	
                            image="docker:///aditi1208/lung-segmentation-preprocess:latest",
                            image_site="docker_hub"
	                )
        
        unet_wf_preprocess_cont = Container(	
	                    "unet_wf_model",	
	                    Container.SINGULARITY,	
                            image="docker:///papajim/lung-segmentation-model:latest",
                            image_site="docker_hub"
	                )

    tc.add_containers(unet_wf_cont, unet_wf_preprocess_cont)

    preprocess = Transformation(
                    "preprocess",
                    site="local",
                    pfn=top_dir / "bin/preprocess/preprocess.py",
                    is_stageable=True,
                    container=unet_wf_preprocess_cont
                )

    unet = Transformation(
                    "unet",
                    site="local",
                    pfn=top_dir / "bin/model/unet.py",
                    is_stageable=True,
                    container=unet_wf_cont
                )

    utils = Transformation(
                    "utils",
                    site="local",
                    pfn=top_dir / "bin/model/utils.py",
                    is_stageable=True,
                    container=unet_wf_cont
                )

    hpo_task = Transformation( 
                    "hpo",
                    site="local",
                    pfn=top_dir / "bin/model/hpo.py",
                    is_stageable=True,
                    container=unet_wf_cont
                ).add_pegasus_profile(cores=8, runtime=14400)


    train_model = Transformation( 
                    "train_model",
                    site="local",
                    pfn=top_dir / "bin/model/train_model.py",
                    is_stageable=True,
                    container=unet_wf_cont
                ).add_pegasus_profile(cores=8, runtime=7200)

    predict_masks = Transformation( 
                    "predict_masks",
                    site="local",
                    pfn=top_dir / "bin/model/prediction.py",
                    is_stageable=True,
                    container=unet_wf_cont
                ).add_pegasus_profile(cores=8, runtime=3600)

    evaluate_model = Transformation( 
                    "evaluate",
                    site="local",
                    pfn=top_dir / "bin/model/evaluate.py",
                    is_stageable=True,
                    container=unet_wf_cont
                )
    
    if args.gpus:
        hpo_task.add_pegasus_profile(gpus=1)
        train_model.add_pegasus_profile(gpus=1)
        predict_masks.add_pegasus_profile(gpus=1)

    tc.add_transformations(preprocess, hpo_task, train_model, predict_masks, evaluate_model, unet, utils)

    log.info("writing tc with transformations: {}, containers: {}".format([k for k in tc.transformations], [k for k in tc.containers]))
    tc.write()

    if args.donut:
        log.info("using donut site catalog")
        sc = create_site_catalog()
        sc.write()

    # --- Generate and run Workflow ------------------------------------------------
    wf = Workflow("lung-instance-segmentation-wf")

    #create preprocess job
    training_input_files = []
    mask_files = []

    processed_training_files = []
    processed_val_files = []
    processed_test_files = []
    training_masks = []
    val_masks = []
    test_masks = []
    process_jobs = train_test_val_split(preprocess, training_input_files, mask_files, processed_training_files, processed_val_files, processed_test_files, training_masks, val_masks, test_masks, args.num_inputs)
    wf.add_jobs(*process_jobs)
    log.info("generated 3 preprocess jobs")

    #create hpo job
    log.info("generating hpo job")
    hpo_checkpoint_result = File(f"study_checkpoint.pkl")
    study_result_list = []
    unet_file = File("unet.py")
    if args.hpo_jobs > 1:
        for i in range(1,args.hpo_jobs+1):
            study_result = File(f"study_result_{i}.txt")
            study_result_list.append(study_result)
            hpo_job = Job(hpo_task)\
                    .add_args("--results_file", study_result, "--storage_path", args.hpo_storage)\
                    .add_inputs(*processed_training_files, *processed_val_files, *training_masks, *val_masks, unet_file)\
                    .add_outputs(study_result)\
                    .add_checkpoint(hpo_checkpoint_result)
            wf.add_jobs(hpo_job)
    else :
        study_result = File("study_results.txt")
        study_result_list.append(study_result)
        hpo_job = Job(hpo_task)\
                    .add_args("--results_file", study_result)\
                    .add_inputs(*processed_training_files, *processed_val_files, *training_masks, *val_masks, unet_file)\
                    .add_outputs(study_result)\
                    .add_checkpoint(hpo_checkpoint_result)

        wf.add_jobs(hpo_job)

    # create training job
    log.info("generating train_model job")
    model = File("model.h5")
    utils_file = File("utils.py")
    train_job = Job(train_model)\
                    .add_args("--results_file", study_result_list[0])\
                    .add_inputs(study_result_list[0], *processed_training_files, *processed_val_files, *training_masks, *val_masks, unet_file, utils_file)\
                    .add_outputs(model)

    wf.add_jobs(train_job)

    # create mask prediction job
    log.info("generating prediction job; using {} test lung images".format(len(processed_test_files)))
    predicted_masks = [File("pred_"+f.lfn.replace(".png", "_mask.png")[5:]) for f in processed_test_files]
    predict_job = Job(predict_masks)\
                    .add_inputs(model, *processed_test_files, unet_file)\
                    .add_outputs(*predicted_masks)

    wf.add_jobs(predict_job)

    #create evalute job
    pdf_analysis = File("EvaluationAnalysis.pdf")
    evaluate_job = Job(evaluate_model)\
                    .add_inputs(*processed_training_files, *processed_test_files, *predicted_masks, *test_masks, unet_file)\
                    .add_outputs(pdf_analysis)

    wf.add_jobs(evaluate_job)
    
    # run workflow
    log.info("begin workflow execution")
    if args.donut:
        wf.plan(submit=True, dir="runs", sites=["donut"], output_sites=["local"], verbose=3)
    else:
        wf.plan(submit=True, dir="runs", sites=["condorpool"], output_sites=["local"])

    #wf.graph(include_files=True, no_simplify=True, label="xform-id", output="graph.dot")


if __name__ == "__main__":
    global LUNG_IMG_DIR
    global LUNG_MASK_IMG_DIR
    args = parser.parse_args(sys.argv[1:])

    LUNG_IMG_DIR = Path(args.lung_img_dir)
    LUNG_MASK_IMG_DIR = Path(args.lung_mask_img_dir)
    
    if args.hpo_jobs < 1:
        log.warning("Number of hpo jobs must be > 0. Setting number of hpo jobs to 1.")
        args.hpo_jobs = 1
    elif args.hpo_jobs > 1 and args.hpo_storage == "":
        log.error("For more than 1 hpo jobs the --hpo-storage option needs to be set.")
        exit(1)

    run_workflow(args)
