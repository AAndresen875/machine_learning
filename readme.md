# This repository is examples of work related to Machine Learning

This repo is a work in progress, and is trying to clean up and put together some code examples mostly from the work I completed with the Indian Institute of Technology Roorkee's Executive Education Division in Data Science and Machine Learning, and various projects since then incorperating more real world problems and more advanced coding infrastructure.

The link with the data for the credit card fraud detection is here: https://drive.google.com/file/d/1yDluWXKfBtQvDm3aJ_1ML701EXoBCu77/view?usp=sharing
To be able to run the notebooks where it is a dependency. Access the link, download the csv and put it in the working directory of this repo. The gitignore is programmed to not commit this csv due to it's size.

### Notebooks: 
* classical_machine_learning_concatenated_examples: Concatenated examples of:
    * Principle component analysis
    * Linear Disscriminant Analysis
    * Linear Regression
    * Logistic Regression
    * Support Vector Machine
    * Decision Tree
    * K-Means Clustering
    * Hierarchical Clustering
    * Artificial Nueral Networks
    * MLP Classicication

* Capstone notebooks: a collection of different versions of the group project Kaggle competition looking to classify instances of credit card fraud. This notebook uses the data from the csv that is needed to be downloaded and added to the directory to run this notebook. I do not remember which is the final version of the project and I am working on getting all the notebooks related to this together. The notebooks include multiple ML models trained and assesed for the fraud detection as well as initial data analysis and exploration. 

### Project requirements:
* I've started a requirements.txt file, but I have not yet completed it or checked it by making a new conda environment to check for completion.

### Choosing an ML model guidemap:
```mermaid
graph TD
    Start["Define Your Problem"] --> DataType{"What type of data?"}
    
    DataType -->|Labeled Data| Supervised["Supervised Learning"]
    DataType -->|Unlabeled Data| Unsupervised["Unsupervised Learning"]
    DataType -->|High Dimensional| DimReduce["Dimensionality Reduction"]
    
    DimReduce --> Labeled{"Have labels?"}
    Labeled -->|Yes| LDA["Linear Discriminant Analysis"]
    Labeled -->|No| PCA["Principal Component Analysis"]
    
    Supervised --> Output{"Output type?"}
    Output -->|Continuous Values| Regression["Regression Problem"]
    Output -->|Categories/Classes| Classification["Classification Problem"]
    
    Regression --> LinearRel{"Linear relationship?"}
    LinearRel -->|Yes| LinReg["Linear Regression"]
    LinearRel -->|No| NonLinReg{"Complex patterns?"}
    NonLinReg -->|Yes| ANN["Artificial Neural Networks"]
    NonLinReg -->|Moderate| DT_Reg["Decision Tree"]
    
    Classification --> NumClasses{"Number of classes?"}
    NumClasses -->|Binary| Binary["Binary Classification"]
    NumClasses -->|Multiple| Multi["Multi-class Classification"]
    
    Binary --> Interpretable{"Need interpretability?"}
    Interpretable -->|Yes| LogReg["Logistic Regression"]
    Interpretable -->|No| Complex_Binary{"Complex patterns?"}
    Complex_Binary -->|Yes| MLP["MLP Classification"]
    Complex_Binary -->|No| SVM_Binary["Support Vector Machine"]
    
    Multi --> Separable{"Clear class separation?"}
    Separable -->|Yes| SVM_Multi["Support Vector Machine"]
    Separable -->|No| Complex_Multi{"Very complex?"}
    Complex_Multi -->|Yes| MLP_Multi["MLP Classification"]
    Complex_Multi -->|No| DT_Multi["Decision Tree"]
    
    Unsupervised --> Goal{"What's your goal?"}
    Goal -->|Group similar items| Clustering["Clustering"]
    Goal -->|Reduce features| UseDimReduce["Use PCA"]
    
    Clustering --> KnowK{"Know number of clusters?"}
    KnowK -->|Yes| Speed{"Need speed?"}
    KnowK -->|No| Hierarchical["Hierarchical Clustering"]
    
    Speed -->|Yes| KMeans["K-Means Clustering"]
    Speed -->|No| ConsiderHier{"Want hierarchy?"}
    ConsiderHier -->|Yes| HierAlt["Hierarchical Clustering"]
    ConsiderHier -->|No| KMeansAlt["K-Means Clustering"]
```

### How to evaluate a model guide map:
```mermaid
graph TD
    A["Start: What type of problem?"] --> B{"Problem Type"};
    B -->|"Classification"| C{"Is dataset balanced?"};
    B -->|"Regression"| D{"Are outliers important?"};
    B -->|"Generation"| E{"Do you have reference texts?"};
    B -->|"Ranking/Recommendation"| F{"Is order important?"};
    
    C -->|"Yes, balanced"| G["Use: Accuracy, F1 Score"];
    C -->|"No, imbalanced"| H{"What's more costly?"};
    
    H -->|"False Positives"| I["Use: Precision, Specificity"];
    H -->|"False Negatives"| J["Use: Recall, Sensitivity"];
    H -->|"Both equally"| K["Use: F1 Score, AUC-ROC"];
    
    D -->|"Yes, care about outliers"| L["Use: MSE, RMSE"];
    D -->|"No, robust to outliers"| M["Use: MAE, MAPE"];
    
    L --> N{"Need interpretability?"};
    M --> N;
    N -->|"Yes"| O["Also use: R², residual plots"];
    N -->|"No"| P["Proceed with chosen metric"];
    
    E -->|"Yes"| Q["Use: BLEU, ROUGE, METEOR"];
    E -->|"No"| R["Use: Perplexity, Human Eval"];
    
    F -->|"Yes, position matters"| S["Use: NDCG, MAP"];
    F -->|"No, just relevance"| T["Use: Precision@K, Recall@K"];
    
    G --> U{"Multi-class?"};
    K --> U;
    U -->|"Yes"| V["Use: Macro/Micro averaging, Confusion Matrix"];
    U -->|"No, binary"| W["Also check: AUC-ROC, PR curve"];
    
    I --> X["Also examine: Confusion Matrix"];
    J --> X;
    
    Q --> Y{"Need human assessment?"};
    R --> Y;
    Y -->|"Yes"| Z["Add: Human evaluation protocols"];
    Y -->|"No"| AA["Proceed with automated metrics"];
    
    S --> AB["Also consider: MRR"];
    T --> AB;
    
    V --> AC["Final step: Validate on holdout set"];
    W --> AC;
    X --> AC;
    O --> AC;
    P --> AC;
    Z --> AC;
    AA --> AC;
    AB --> AC;
    
    AC --> AD{"Deploying to production?"};
    AD -->|"Yes"| AE["Also monitor: Latency, Throughput, Fairness"];
    AD -->|"No"| AF["Complete evaluation"];
```