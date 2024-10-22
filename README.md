# NLPProject

Check dependencies in environment.yaml  
All 3 models in project Folder.  
transformersProject.py uses GPU acceleration with CUDA version 12.4  
If you do not have an NVIDIA GPU available, the CPU will be used.  
Corpus in train.csv  

countVector.py = basic word frequency  
transformersProject.py = use distilbert to create embeddings  
approxNearestNeighborProject = an approximate k nearest neighbor  

To run. Import yaml file to virtual environment. Change relative references for the file name in the model. In command line: python3 {modelname.py}.  
You will have to install pytorch manually if it not already installed on your system.



