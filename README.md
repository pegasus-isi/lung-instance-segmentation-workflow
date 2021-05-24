# Lung-Instance-Segmentation-Workflow 
(Instance segmentation with U-Net/Mask R-CNN workflow using Keras &amp; Ray Tune)

![workflow](img/graph.png)

![workflow-outline](img/workflow.jpg)

Lung instance segmentation workflow uses [Chest X-ray](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4256233/) for predicting lung masks from the images using [U-Net](https://arxiv.org/abs/1505.04597) model. 

## Running the Workflow

* Clone the respository using the command `git clone <repository link>`
* `cd` into the `lung-instance-segmentation-workflow` directory
*  [Optional] If you want to add your own docker image, go to `workflow.py` file and change the image in 

    ```python
    unet_wf_cont = Container(
                "unet_wf",
                Container.DOCKER,
                image="docker:///aditi1208/lung-segmentation:latest"
            )
    ``` 
    
    part, to the link to your docker image
* Run the workflow script using the command `python3 workflow.py`
* Check the predicted masks, model.h5 file, and the checkpoint file in the `wf-output` folder

## Executing Standalone Scripts

* Clone the respository using the command `git clone <repository link>`
* `cd` into the `lung-instance-segmentation-workflow/bin` directory
* Use the command `pip3 -r requirements.txt` to install the required packages
* Go back to the `lung-instance-segmentation-workflow` directory and ensure that you have a directory called `output` here
* Download the dataset by running the python script called "get-dataset.py" by `python get-datatset.py`
* Execute the `end-to-end.sh` script