def f1_score_max(gt, pred, thresh):
  import math
  from sklearn.metrics import precision_score, recall_score
  import numpy as np
  #P, R, thresh = precision_recall_curve(gt, pred)
  #F1 = 2*P*R/(P+R)
  #F1_ = [n for n in F1 if not math.isnan(n)]

  P=[];R=[]
  for i in thresh:
    new_pred = ((pred>=i)*1).flatten()
    P.append(precision_score(gt.flatten(), new_pred))
    R.append(recall_score(gt.flatten(), new_pred))
  P = np.array(P).flatten()
  R = np.array(R).flatten()
  F1 = 2*P*R/(P+R)
  F1_MAX = max(F1)
  if F1_MAX<0 or math.isnan(F1_MAX): 
    F1_MAX=0
    F1_THRESH=0
  else:
    idx_thresh = np.argmax(F1)
    F1_THRESH = thresh[idx_thresh]

  return F1, F1_MAX, F1_THRESH

def f1_score(gt, pred, F1_Thresh=0.5, median=False):
  import math
  import pandas
  import numpy as np
  import ipdb
  from sklearn.metrics import precision_score, recall_score
  from sklearn.metrics import f1_score as f1s
  if type(gt)==list: gt = np.array(gt)
  if type(pred)==list: pred = np.array(pred)
  # F1_Thresh = 0.5
  output = (pred>F1_Thresh)*1.0
  F1 = f1s(gt, output)
  F1_MAX=F1

  if median:
    # ipdb.set_trace()
    output_median3 = np.array(pandas.Series(output).rolling(window=3, center=True).median().bfill().ffill())
    F1_median3 = f1s(gt, output_median3)

    output_median5 = np.array(pandas.Series(output).rolling(window=5, center=True).median().bfill().ffill())
    F1_median5 = f1s(gt, output_median5)

    output_median7 = np.array(pandas.Series(output).rolling(window=7, center=True).median().bfill().ffill())
    F1_median7 = f1s(gt, output_median7)

    return [F1], F1_MAX, F1_Thresh, F1_median3, F1_median5, F1_median7
  else:
    return [F1], F1_MAX, F1_Thresh 
