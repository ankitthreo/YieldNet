import numpy as np
import pandas as pd
pred = pd.read_csv("../predict/sc/sc_1_h300_preds.csv")['Yield']
pred = list(pred)

test = pd.read_csv("../dataset/sc/test/SC_FullCV_1.csv")['Yield']
test = list(test)
pred = np.array(pred)
test = np.array(test)
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(test, pred)
print(mae)
