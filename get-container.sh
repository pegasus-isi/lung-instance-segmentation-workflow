#!/usr/bin/env bash

mkdir containers
singularity pull containers/lung-segmentation_latest.sif docker://papajim/lung-segmentation:works21
