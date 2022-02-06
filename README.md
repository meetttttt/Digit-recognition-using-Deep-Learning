# Digit-recognition-using-Deep-Learning


# prerequisite for this project
1. Tensorflow above 2.0.0 

# What dataset look like:
![](https://www.kaggleusercontent.com/kf/80445693/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..cPVbnnI8YyxINN-uafB5uQ.VZmudPRwIcnA9EbsnWNoa6wjI0aj384JsYXkxYbZpD8vKllChH4A1WwKXjGlIEiUeJwCY4WcDOkSjnYL1EsGRFKAprXg4JtCMdV55aXLTQ6riDwpRxrT_lH0c-L1qUM5u80jrQNdVBYu2j-PuTBBcNUfwUb5B_u9uehSOQWm9vPc65ITK90gIi3iLHg56lO8P6gH6dQVXgWK5WeEpnD3UxE_J3vukeaSYG4Zb3Ixwne36Z8h_XzesMbk8JbqXNwPIcPQozwvEyhBlC7tptn-XlXXVP0nCXqTzGpy_ENIyewyJQnfMgYfGvjRk5aak4sTWznF1ZJW5uxpRgL4v7eG0TDSgQcTszhxQBZRZlR-AN-uhl7JMpUCRl2xn79WJxxOL0FcowiiynwiBu93sk3pLeKk7iCJrIPQ95nNSrsmvcBpIYPu2sDqvpBZ0ANkwhPE9d-PvlPvMvNOtLXjdHNlfxjT9aM-Faa7YFXfuORiXQO7ucXwY47G5HYuXxNPv1-DGN2cAag2NPhg-xRPhgxk42PC8UWeCRfSrjwpPf_UmJa2MLiugbq7Cjr133aUa-rQtY86vOCcFe_VPP9uuiBz65xCvrYptzNfeIxNMUiRvQm71byBCHNu0Mb4Zro-NrYQ3BIJ0CUG8AUdo3PXTf-KwGrwqRtTJ3yp2ce8iQSJLyZe_PqHTzGmtnCwPrLf-stY.vRg6NL_8eQv24rXo1ibwaA/__results___files/__results___10_1.png)

![](https://www.kaggleusercontent.com/kf/80445693/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..cPVbnnI8YyxINN-uafB5uQ.VZmudPRwIcnA9EbsnWNoa6wjI0aj384JsYXkxYbZpD8vKllChH4A1WwKXjGlIEiUeJwCY4WcDOkSjnYL1EsGRFKAprXg4JtCMdV55aXLTQ6riDwpRxrT_lH0c-L1qUM5u80jrQNdVBYu2j-PuTBBcNUfwUb5B_u9uehSOQWm9vPc65ITK90gIi3iLHg56lO8P6gH6dQVXgWK5WeEpnD3UxE_J3vukeaSYG4Zb3Ixwne36Z8h_XzesMbk8JbqXNwPIcPQozwvEyhBlC7tptn-XlXXVP0nCXqTzGpy_ENIyewyJQnfMgYfGvjRk5aak4sTWznF1ZJW5uxpRgL4v7eG0TDSgQcTszhxQBZRZlR-AN-uhl7JMpUCRl2xn79WJxxOL0FcowiiynwiBu93sk3pLeKk7iCJrIPQ95nNSrsmvcBpIYPu2sDqvpBZ0ANkwhPE9d-PvlPvMvNOtLXjdHNlfxjT9aM-Faa7YFXfuORiXQO7ucXwY47G5HYuXxNPv1-DGN2cAag2NPhg-xRPhgxk42PC8UWeCRfSrjwpPf_UmJa2MLiugbq7Cjr133aUa-rQtY86vOCcFe_VPP9uuiBz65xCvrYptzNfeIxNMUiRvQm71byBCHNu0Mb4Zro-NrYQ3BIJ0CUG8AUdo3PXTf-KwGrwqRtTJ3yp2ce8iQSJLyZe_PqHTzGmtnCwPrLf-stY.vRg6NL_8eQv24rXo1ibwaA/__results___files/__results___14_1.png)

# Plottting Loss of the mode
![](https://www.kaggleusercontent.com/kf/80445693/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..cPVbnnI8YyxINN-uafB5uQ.VZmudPRwIcnA9EbsnWNoa6wjI0aj384JsYXkxYbZpD8vKllChH4A1WwKXjGlIEiUeJwCY4WcDOkSjnYL1EsGRFKAprXg4JtCMdV55aXLTQ6riDwpRxrT_lH0c-L1qUM5u80jrQNdVBYu2j-PuTBBcNUfwUb5B_u9uehSOQWm9vPc65ITK90gIi3iLHg56lO8P6gH6dQVXgWK5WeEpnD3UxE_J3vukeaSYG4Zb3Ixwne36Z8h_XzesMbk8JbqXNwPIcPQozwvEyhBlC7tptn-XlXXVP0nCXqTzGpy_ENIyewyJQnfMgYfGvjRk5aak4sTWznF1ZJW5uxpRgL4v7eG0TDSgQcTszhxQBZRZlR-AN-uhl7JMpUCRl2xn79WJxxOL0FcowiiynwiBu93sk3pLeKk7iCJrIPQ95nNSrsmvcBpIYPu2sDqvpBZ0ANkwhPE9d-PvlPvMvNOtLXjdHNlfxjT9aM-Faa7YFXfuORiXQO7ucXwY47G5HYuXxNPv1-DGN2cAag2NPhg-xRPhgxk42PC8UWeCRfSrjwpPf_UmJa2MLiugbq7Cjr133aUa-rQtY86vOCcFe_VPP9uuiBz65xCvrYptzNfeIxNMUiRvQm71byBCHNu0Mb4Zro-NrYQ3BIJ0CUG8AUdo3PXTf-KwGrwqRtTJ3yp2ce8iQSJLyZe_PqHTzGmtnCwPrLf-stY.vRg6NL_8eQv24rXo1ibwaA/__results___files/__results___28_1.png)

# Plotting Accuracy of the model
![](https://www.kaggleusercontent.com/kf/80445693/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..cPVbnnI8YyxINN-uafB5uQ.VZmudPRwIcnA9EbsnWNoa6wjI0aj384JsYXkxYbZpD8vKllChH4A1WwKXjGlIEiUeJwCY4WcDOkSjnYL1EsGRFKAprXg4JtCMdV55aXLTQ6riDwpRxrT_lH0c-L1qUM5u80jrQNdVBYu2j-PuTBBcNUfwUb5B_u9uehSOQWm9vPc65ITK90gIi3iLHg56lO8P6gH6dQVXgWK5WeEpnD3UxE_J3vukeaSYG4Zb3Ixwne36Z8h_XzesMbk8JbqXNwPIcPQozwvEyhBlC7tptn-XlXXVP0nCXqTzGpy_ENIyewyJQnfMgYfGvjRk5aak4sTWznF1ZJW5uxpRgL4v7eG0TDSgQcTszhxQBZRZlR-AN-uhl7JMpUCRl2xn79WJxxOL0FcowiiynwiBu93sk3pLeKk7iCJrIPQ95nNSrsmvcBpIYPu2sDqvpBZ0ANkwhPE9d-PvlPvMvNOtLXjdHNlfxjT9aM-Faa7YFXfuORiXQO7ucXwY47G5HYuXxNPv1-DGN2cAag2NPhg-xRPhgxk42PC8UWeCRfSrjwpPf_UmJa2MLiugbq7Cjr133aUa-rQtY86vOCcFe_VPP9uuiBz65xCvrYptzNfeIxNMUiRvQm71byBCHNu0Mb4Zro-NrYQ3BIJ0CUG8AUdo3PXTf-KwGrwqRtTJ3yp2ce8iQSJLyZe_PqHTzGmtnCwPrLf-stY.vRg6NL_8eQv24rXo1ibwaA/__results___files/__results___29_1.png)



#  Confusion Matrix

[[ 975    1    1    0    0    0    1    0    1    1]
 [   0 1127    2    1    0    1    2    1    1    0]
 [   2    1 1016    3    1    0    0    4    4    1]
 [   0    0    1  997    0    1    0    2    3    6]
 [   1    1    3    0  964    0    3    1    1    8]
 [   2    0    0    8    1  874    4    0    2    1]
 [   4    2    2    1    3    4  942    0    0    0]
 [   2    3    7    0    2    0    0 1007    3    4]
 [   3    0    2    4    3    2    2    2  952    4]
 [   2    2    0    2    6    2    1    1    2  991]]
 
 # Classificaton Report
 
               precision    recall  f1-score   support

           0       0.98      0.99      0.99       980
           1       0.99      0.99      0.99      1135
           2       0.98      0.98      0.98      1032
           3       0.98      0.99      0.98      1010
           4       0.98      0.98      0.98       982
           5       0.99      0.98      0.98       892
           6       0.99      0.98      0.98       958
           7       0.99      0.98      0.98      1028
           8       0.98      0.98      0.98       974
           9       0.98      0.98      0.98      1009

    accuracy                           0.98     10000
   macro avg       0.98      0.98      0.98     10000
weighted avg       0.98      0.98      0.98     10000
