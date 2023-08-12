# GLEMR
GLEMR: A **G**enerative **L**abel **E**nhancement model with Gaussian **M**ixture and partial **R**anking
 
## Environment
python=3.7.6, numpy=1.21.6, scipy=1.7.3, pytorch=1.13.0+cpu.

## Reproducing
Change the directory to this project and run the following command in terminal.
```Terminal
python demo.py
```


## Usage
Here is a simple example of using GLEMR.
```python
import numpy as np
from utils import report, binarize
from glemr import GLEMR

# load data
X, D = load_dataset('sj') # this api should be defined by users
L = binarize(D)

# train without early-stop technique
model = GLEMR(trace_step=np.inf).fit(X, L)
# show the recovery performance
Drec = model.label_distribution_
report(Drec, D)

# train with early-stop technique
model = GLEMR(trace_step=50).fit(X, L)
Drec_trace = model.trace_
# show the recovery performance
for k in Drec_trace.keys():
    print("The recovery performance at %d-th iteration:" % k)
    report(Drec_trace[k], D)
```

## Datasets
- The datasets used in our work is partially provided by [PALM](http://palm.seu.edu.cn/xgeng/LDL/index.htm)
- Emotion6: [http://chenlab.ece.cornell.edu/people/kuanchuan/index.html](http://chenlab.ece.cornell.edu/people/kuanchuan/index.html)
- Twitter-LDL and Flickr-LDL: [http://47.105.62.179:8081/sentiment/index.html](http://47.105.62.179:8081/sentiment/index.html)

## Paper
```latex
@inproceedings{Lu2023GLEMR,
    title={Generative Label Enhancement with Gaussian Mixture and Partial Ranking},
    author={Yunan Lu and Liang He and Fan Min and Weiwei Li and Xiuyi Jia},
    booktitle={AAAI Conference on Artificial Intelligence},
    year={2023},
    pages={8975-8983}
}
```
