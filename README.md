# TransGraphEdit
Environment Requirements
python=3.8 \
pytorch==1.12.1 \
Other environmental references requirements.txt 


# Data preprocessing
1.generate the edit labels \
python preprocess.py --mode train/test/valid 

2.prepare the data for training \
python prepare_data.py --mode train/test/valid 

# Train
python train.py

# Evaluate
python eval.py

# Reproduce our results
python eval.py --dataset C-H Arylation --use_rxn_class False --experiments 15-11-2024--10-36-38
