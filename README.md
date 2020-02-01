# How to read our repository
* utils: includes helper functions (objective functions used in meta-heuristic algorithms)
    * We have also built a framework for benchmark functions (such as unimodal, multimodal, composition, ... CEC 2014, ...)
    * Check it out: https://pypi.org/project/opfunu/

* script_test
    * Included each algorithm-run file (Test purpose)
    * To run it, move it to root folder (code_IQSO_MLP) like the file: run_GA.py

* models: includes all algorithms (4 folders)
    * human_based
    * physics_based
    * swarm_based
    * evolutionary_based
    * The file: root_algo.py is the root for all meta-heuristic algorithms. Because there are lots of common functions among 
algorithms. So better to have an abstract layer for all algorithms.
    
    * We are building a library for all the state-of-the-art meta-heuristic algorithms using python.
    * It haven't done yet, but you can check its development version at: https://github.com/thieunguyen5991/metaheuristics
    
* How to run?
    * 1st: run file run_multiple_algo.py (run each algorithm 15 times and save the best fitness of each run times and loss
    of the best among 15 run times into folder: convergence and stability)
    * 2nd: run file get_experiment_infor.py to read saved data from 1st step. Then transform that data into dict type 
    for making latex table then save it in foler: overall/all_algo_infor.pkl
    * 3rd: run file gen_result_tex.py to make latex table from all_algo_infor.pkl file
    * 4th: run file plot_stab_conv.py to draw convergence and stability of 30 functions.
    
    * To change the parameters of models in: run_multiple_algo.py

# Publications
* If you see our code and data useful and use it, please cites us here
    * Nguyen, T., Nguyen, T., Nguyen, B. M., & Nguyen, G. (2019). Efficient Time-Series Forecasting Using Neural Network and Opposition-Based Coral Reefs Optimization. International Journal of Computational Intelligence Systems, 12(2), 1144-1161.
    
    * Nguyen, T., Tran, N., Nguyen, B. M., & Nguyen, G. (2018, November). A Resource Usage Prediction System Using Functional-Link and Genetic Algorithm Neural Network for Multivariate Cloud Metrics. In 2018 IEEE 11th Conference on Service-Oriented Computing and Applications (SOCA) (pp. 49-56). IEEE.

    * Nguyen, T., Nguyen, B. M., & Nguyen, G. (2019, April). Building Resource Auto-scaler with Functional-Link Neural Network and Adaptive Bacterial Foraging Optimization. In International Conference on Theory and Applications of Models of Computation (pp. 501-517). Springer, Cham.

* The pre-version of our paper and code can be found at: https://github.com/chasebk/

* Don't hesitate if you have any question about our code and paper via nguyenthieu2102@gmail.com or hoangnghiabao96@gmail.com 
