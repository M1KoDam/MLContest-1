from data_handler import handle_features, handle_labels, get_classes
import pandas as pd
import numpy as np

train_data = pd.read_csv("data/train.csv")
useless = {'Var127', 'Var169', 'Var150', 'Var14', 'Var89', 'Var108', 'Var139', 'Var37', 'Var91', 'Var1', 'Var158', 'Var209', 'Var186', 'Var217', 'Var80', 'Var155', 'Var157', 'Var93', 'Var10', 'Var190', 'Var97', 'Var202', 'Var16', 'Var19', 'Var90', 'Var107', 'Var182', 'Var58', 'Var99', 'Var118', 'Var204', 'Var105', 'Var188', 'Var124', 'Var187', 'Var63', 'Var167', 'Var174', 'Var95', 'Var60', 'Var148', 'Var96', 'Var4', 'Var18', 'Var40', 'Var61', 'Var34', 'Var183', 'Var128', 'Var33', 'Var50', 'Var36', 'Var141', 'Var23', 'Var88', 'Var82', 'Var192', 'Var17', 'Var129', 'Var56', 'Var175', 'Var145', 'Var180', 'Var120', 'Var66', 'Var87', 'Var5', 'Var171', 'Var142', 'Var48', 'Var159', 'Var100', 'Var136', 'Var84', 'Var98', 'Var75', 'Var20', 'Var64', 'Var184', 'Var137', 'Var79', 'Var68', 'Var15', 'Var41', 'Var122', 'Var178', 'Var43', 'Var55', 'Var161', 'Var114', 'Var179', 'Var46', 'Var138', 'Var168', 'Var131', 'Var54', 'Var121', 'Var47', 'Var29', 'Var59', 'Var42', 'Var101', 'Var176', 'Var130', 'Var151', 'Var52', 'Var111', 'Var26', 'Var53', 'Var170', 'Var102', 'Var165', 'Var70', 'Var11', 'Var116', 'Var135', 'Var147', 'Var2', 'Var8', 'Var71', 'Var69', 'Var164', 'Var177', 'Var32', 'Var9', 'Var152', 'Var45', 'Var230', 'Var104', 'Var172', 'Var27', 'Var12', 'Var115', 'Var156', 'Var62', 'Var185', 'Var3', 'Var92', 'Var146', 'Var86', 'Var31', 'Var103', 'Var106', 'Var77', 'Var154', 'Var117', 'Var162', 'Var110', 'Var199', 'Var67', 'Var166', 'Var30', 'Var49', 'Var39'}

for i in train_data.columns.tolist():
    if i not in useless:
        input("NEXT?")
        print(train_data[i].head(10))

train_labels_pd = train_data['Target'].values
train_features_pd = train_data.drop(columns=['Target', 'ID']).values

test_features_pd = pd.read_csv("data/data_predict.csv").drop(columns=['ID']).values
test_labels_pd = pd.read_csv("data/sample_submission.csv").drop(columns=['ID']).values

# x_train, x_test = handle_features(train_features_pd, test_features_pd)
# y_train = handle_labels(train_labels_pd)
