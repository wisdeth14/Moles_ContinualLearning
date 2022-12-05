# Moles_ContinualLearning
Project uses pytorch version 1.11.0 and Avalanche (a Continual Learning library) version 0.2.1.

Install Avalanche:
pip install avalanche-lib

Run program (baseline):
python icarl.py

Run program with moles:
python icarl.py --moles

Adjust seed through --seed flag and exemplar size through --ex flag.

Further edits to experiment setup (i.e. epochs, number of tasks, batchsize) can be edited with the config in main()
