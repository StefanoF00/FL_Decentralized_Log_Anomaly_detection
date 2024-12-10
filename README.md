# Control Engineering Master's Thesis: "Anomaly Detection in Mobile Edge Computing Infrastructures using Application Logs: an Adaptive Federated Learning and Finite Time Consensus-based Approach"

As part of the [__NANCY__](https://nancy-project.eu) project funded by European Union’s Horizon Europe research and innovation programme, this work proposes two federated approaches for performing anomaly detection on log files produced by a Mobile Edge Computing infrastructure. The proposed algorithms, widely described in `Master_Thesis_Felli_2023_2024.pdf` file, preserve the privacy of the various clients participating in the infrastructure. 

## AdaLightLog Algorithm
This first framework, based on Federated Learning algorithm __AdaFed__[¹], relies on a central server for the parameter averaging procedure. This proposed algorithm was presented in the proceedings of the International [Conference on Critical Information Infrastructures Security (CRITIS2024)](https://critis2024.uniroma3.it) - submitted on 31/05/2024 and discussed on 18/09/2024 - and was a finalist for the Young CRITIS Award (YCA). 

## DecAdaLightLog Algorithm
It is a framework that, based on finite-time consensus protocol for multi-agent systems and Physics-Informed Neural Networks, has a totally distributed nature. It doesn’t depend on a central coordinating server that represents a point of failure for cyber-physical attacks. Given the innovative contribution and the surprising results of this algorithm, it is planned to submit the complete work for further publication in a scientific journal.

## Get Started
* Import `HDFS.log` file into the `/Raw_data_logs/HDFS` folder. Due to its high dimension (1.58GB), it is not present in the repository but can be found on [Kaggle](https://www.kaggle.com/datasets/ayenuryrr/loghub-hdfs-hadoop-distributed-file-system-data).
* Run all the cells in `Log_anomaly_Detection/main.ipynb`, which incorporate client definition, log processing, and training of FedAvg[²], AdaLightLog, and DecAdaLightLog algorithms.

---

## References
¹: Alessandro Giuseppi, Lucrezia Della Torre, Danilo Menegatti, and Antonio Pietrabissa. Adafed: Performance-based adaptive federated learning. In *2021 The 5th International Conference on Advances in Artificial Intelligence (ICAAI)*.

²: H. Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Agüera y Arcas. Communication-efficient learning of deep networks from decentralized data *Proceedings of the 20 th International Conference on
Artificial Intelligence and Statistics (AISTATS) 2017*.
