# recfairness

### Creator Churn
To reproduce the experiment of the creator churn 
   * python3 CreatorChurn/creatorChurnModel.py

### Rec experiments
#### The randomized strategy
Prepare the data
   * python3 ../RecExepriments/Baseline/dataPreprocess.py

The randomized stratetgy on Data\_all
   * python3 ../RecExepriments/Baseline/baselineRandom.py

The randomized strategy on Data\_churn
   * python3 ../RecExepriments/Baseline/baselineRandomWithChurn.py

The randomized strategy on Data\_edm
   * python3 ../RecExepriments/Baseline/baselineRandomEDM.py

#### The CF algorithm
Prepare the data
   * python3 ../RecExepriments/CFExperiments/dataPreprocess.py

The CF stratetgy on Data\_all
   * python3 ../RecExepriments/CFExperiments/baselineCF.py

The CF strategy on Data\_churn
   * python3 ../RecExepriments/CFExperiments/ChurnBaselineCF.py

The CF strategy on Data\_edm
   * python3 ../RecExepriments/CFExperiments/EDMBaselineCF.py
#### The neuralCF algorithm
Prepare the data
   * python3 ../RecExepriments/NeuralCF/dataPreprocess.py

The neuralCF stratetgy on Data\_all
   * python3 ../RecExepriments/NeuralCF/baselineNCF.py

The neuralCF strategy on Data\_churn
   * python3 ../RecExepriments/NeuralCF/ChurnBaselineNCF.py

The neuralCF strategy on Data\_edm
   * python3 ../RecExepriments/NeuralCF/EDMBaselineNCF.py

