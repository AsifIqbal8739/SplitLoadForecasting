# SplitLoadForecasting
This repository contains the experimental code for the "Privacy-Preserving Collaborative Split Learning Framework for Smart Grid Load Forecasting" paper

# Get Started
- Install Python 3.6, PyTorch 1.9.0 to run the code.
- FEDformer is used as the central model 
- SplitGSSP is the SplitGlobal model discussed in the paper
- SplitPerson is the SplitPersonal model discussed in the paper
- To run the individual tests, run
  - run_Central.py to train a Central model 
  - run_SplitFramework to train the SplitGlobal model
  - run_SplitFrameworkPersonal to train the SplitPersonal model
 

# Acknowledgement
We appreciate the following github repos a lot for their valuable code base or datasets:

https://github.com/MAZiqing/FEDformer

https://github.com/thuml/Autoformer

https://github.com/zhouhaoyi/Informer2020

https://github.com/zhouhaoyi/ETDataset
