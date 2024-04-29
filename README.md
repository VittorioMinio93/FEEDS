# FEEDS: Framework for Evaluation of Early Detection Systems
FEEDS (Framework for Evaluation of Early Detection Systems) has been developed for the assessment of the generalization capability of a generic Early Detection (ED) system (e.g., Ripepe et al., 2021), which refers to the ability to provide reliable alarms for target events using data different from those used to develop the system. The framework consists of a Python package comprising functions and classes dedicated to the management of geophysical data, simulation in pseudo real-time (i.e., real-time simulation on previously acquired data), and the evaluation of the performance of an alert model. 
FEEDS assesses the generalization capability of an alert system, i.e., the ability to provide reliable alerts for target events using data different from those with which the system was developed. This evaluation can be quantified through several statistics on predictive parameters (e.g., Cannavò et al., 2017):

- TPR (True Positive Rate): represents the ratio between the number of true positives and the total number of eruptive events considered. In other words, it indicates the system's ability to correctly identify positive events. It varies from 0 to 1, respectively in the worst and best cases.
- FDR (False Discovery Rate): this parameter expresses the ratio between the number of false positives and the total number of alerts issued by the system. It helps to assess how much the system generates erroneously positive alarms. It varies from 0 to 1, respectively in the best and worst cases.
- LT (Lead Time): refers to the period of time elapsed between the issuance of the alert and the occurrence of the event. The greater this time, the better the alert system.
- FTA (Fraction Time Alert): indicates the fraction of total time during which the system is in an alert state. A high FTA could indicate an excessive sensitivity of the system, with a period of alarm persistence that would degrade the system's reliability (paradoxically, an FTA of 1, i.e., a system always on alert, would identify all target events with only one alert). The lower this fraction, the better the alert system.

These parameters provide a detailed assessment of the system's performance, allowing for a better understanding of its effectiveness and reliability in detecting specific events such as paroxysms. The distinctive aspect of FEEDS is the application of Monte Carlo cross-validation to an alert system, representing a significant innovation in the field of volcanic prediction.
The evaluation process of an ED system is complex and involves several phases (**Figure 1**). These include:

- Data collection and preparation. FEEDS performs an automatic signal quality analysis to determine the most suitable periods for subsequent phases.  
- Partitioning of data into training and testing sets. Once the dataset and analysis period are determined, FEEDS proceeds by randomly dividing the entire period into segments of variable length. These periods are randomly chosen to compose the training set, representing 70% of the dataset. Instead, the remaining 30% is reserved for testing the trained system. Furthermore, during this phase, FEEDS ensures that the target events are distributed differently between the two sets.   
- Implementation and optimization of the alert model. FEEDS involves the creation of an Early detection model. At this stage, the system determines the relationship between the data that allows obtaining the maximum possible number of true positives (100% success rate) and the minimum possible number of false positives (0% false alarms).  
- evaluation of model performance. During the final phase, FEEDS utilizes the ED model derived from the training process to conduct a pseudo real-time simulation on the test set, calculating how many times alarms are activated or not. Specifically, various statistics such as TPR, FDR, LT, and FTA are estimated based on these results. Using the Monte Carlo cross-validation method (Picard and Cook, 1984), the entire procedure is repeated a certain number N of times, obtaining information about the distributions of the performance parameters of the warning model for independent sets of data.

  
  ![FEEDS_framework](https://github.com/VittorioMinio93/FEEDS/assets/81485574/83b75831-0465-4576-8306-c29c7d4356f7)
**Figure 1. Schematic representation of the main processes used to evaluate the performance of a generic alert system. At the top right of the figure, Python-implemented functions are listed and used to group the main processes: data management (blue), alert model implementation (yellow), and model performance evaluation (green).** 


In conclusion, the application of Monte Carlo cross-validation to an ED system represents a significant innovation in the field of volcanic prediction. This approach provides a probabilistic estimate of possible scenarios, allowing for a more comprehensive and reliable risk assessment. 
FEEDS allows users to adapt the alert system to different types of data and models, both through a traditional threshold-based approach and the use of artificial intelligence techniques.

# Requirements
FFEDS can be run on any operation system with Python from Release 3.10.9 In addition, you need to have the following packages installed: 

-	pandas, version 1.5.3 or later.
-	numpy, version 1.24.2 or later.

 
# Citation 
If you use this code for your work, please cite the following DOI:
-	https://doi.org/10.5281/zenodo.11082459

# Contact
You can send an email to **vittorio.minio@ingv.it** to report suggestions, comments and bugs.

# References
- Cannavò et al., 2017. A multivariate probabilistic graphical model for real-time volcano monitoring on Mount Etna, Journal of Geophysical Research: Solid Earth.  https://doi.org/10.1002/2016JB013512.
- Picard and Cook, 1984. Cross-validation of regression models, Journal of the American Statistical Association. https://doi.org/10.2307/2288403.
- Ripepe et al., 2021. Ground deformation reveals the scale-invariant conduit dynamics driving explosive basaltic eruptions, Nature Communications. https://doi.org/10.1038/s41467-021-21722-2.
