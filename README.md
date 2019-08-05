# Single-Net Continual Learning with Progressive Segmented Training (PST)

### This is the code to reproduce the results in paper 'Single-Net Continual Learning with Progressive Segmented Training (PST)

### Submission ID: 2994

### Enviroments:

Pytorch: 1.0.1_post2

Python 3.6

CUDA 9.0+

For details, see requirements.txt



To run CIFAR-100 experiment in 'Experiment' section, which is incrementally learning 5/10/20/50 tasks, perform:
	
	python PST_main.py --classes_per_task 5/10/20/50
	
Alternatively, you can change the hyper-parameters by:

	python PST_main.py --classes_per_task 5 --memory_size 5000 --NA_C0 30
	
	