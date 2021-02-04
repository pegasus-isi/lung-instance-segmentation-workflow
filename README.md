# Lung-Instance-Segmentation-Workflow 
(Instance segmentation with U-Net/Mask R-CNN workflow using Keras &amp; Ray Tune)

Lung instance segmentation workflow uses [Chest X-ray](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4256233/) for predicting lung masks from the images using [U-Net](https://arxiv.org/abs/1505.04597) model. 

## Running the Workflow

* Clone the respository using the command `git clone <repository link>`
* `cd` into the `lung-instance-segmentation-workflow` directory
* Run the workflow script using the command `python3 workflow.py`
* Check the predicted masks, model.h5 file, and Checkpoint file in `wf-output` folder

## Executing Standalone Scripts

* Clone the respository using the command `git clone <repository link>`
* `cd` into the `lung-instance-segmentation-workflow/bin` directory
* Use the command `pip3 -r requirements.txt` to install the required packages
* Use the command `python3 <filename>` command to run the `preprocess.py`, `train_model.py`, and `prediction.py` files

## Contributing to the project

* Fork the repository and create a branch using `git checkout -b <branchname>` command
* Add your contributions to this branch and create a Pull Request to this repository. 

