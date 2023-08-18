# Test
# test_raw =  Test_data.loc[Test_data['Label'].isin(target_labels)] # old code
test_target_1 = Test_data.loc[Test_data['Label'].isin(['ebay'])]
test_target_2 = Test_data.loc[Test_data['Label'].isin(['alibaba'])]
test_target_3 = Test_data.loc[Test_data['Label'].isin(['amazon'])]

test_VoIP_facebook = Test_data.loc[Test_data['Label'].isin(['VoIP', 'facebook'])]
test_DataFrameDict = test_case_split(test_VoIP_facebook,3)

test_target_1 = pd.concat([test_target_1,test_DataFrameDict['client_1']],axis=1)
test_target_2 = pd.concat([test_target_2,test_DataFrameDict['client_2']],axis=1)
test_target_3 = pd.concat([test_target_3,test_DataFrameDict['client_3']],axis=1)

del Test_data, test_VoIP_facebook, test_DataFrameDict