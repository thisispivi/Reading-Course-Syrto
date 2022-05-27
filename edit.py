import re
import numpy as np

benchmark = True


def foo(plo):
    l = plo.split(".")
    return l[0] + "." + l[1][:3]


def foo2(plo):
    if "e" in plo:
        l = plo.split("e")
        return plo[:5] + " e" + l[1]
    else:
        l = str(np.format_float_scientific(float(plo), precision=3)).split("e")
        return str(l[0]) + " e" + str(l[1])


string = "Benchmark MAE: 1469.4731669425926 MSE: 5337361426.769167 RMSE: 73057.24759918873 R2: 0.9176039315447099 MAPE: 3475074722341717.0 SMAPE: 0.14560095210098412 Benchmark MAE: 1193.8909819231287 MSE: 469832259.948005 RMSE: 21675.614407624184 R2: 0.9676550087582783 MAPE: 8684242120068.555 SMAPE: 0.08833238787033197 Benchmark MAE: 2245.9660622250067 MSE: 5223986199.1803255 RMSE: 72277.14852690528 R2: 0.9611034933401831 MAPE: 0.17163268437976906 SMAPE: 0.0783179441824097 Benchmark MAE: 1608.9300857905646 MSE: 4308085986.330275 RMSE: 65636.01135299337 R2: 0.8873595401879083 MAPE: 0.8489630252055266 SMAPE: 0.15697258150849805 Benchmark MAE: 1038.14148789655 MSE: 349160701.0482463 RMSE: 18685.84226221142 R2: 0.9309947868985757 MAPE: 5431784343448789.0 SMAPE: 0.13401180551441685 Benchmark MAE: 868.8444663445046 MSE: 964483014.1662521 RMSE: 31056.126837811764 R2: 0.8939521644999249 MAPE: 5.65098803931242e+16 SMAPE: 0.3037819571822138 Benchmark MAE: 1217.222141408266 MSE: 928493202.3657435 RMSE: 30471.18642858764 R2: 0.9634922743171463 MAPE: 3263863980463485.5 SMAPE: 0.10716833319357588 Benchmark MAE: 2245.9660622250067 MSE: 5223986199.1803255 RMSE: 72277.14852690528 R2: 0.9611034933401831 MAPE: 0.17163268437976906 SMAPE: 0.0783179441824097 Benchmark MAE: 2157.6721696720206 MSE: 2951950546.9096055 RMSE: 54331.85572856504 R2: 0.8327983724159599 MAPE: 1914438307176570.2 SMAPE: 0.13206942042851877 Benchmark MAE: 523.4366989872777 MSE: 135523586.4292639 RMSE: 11641.459806624936 R2: 0.6203309906448433 MAPE: 173408233108419.44 SMAPE: 0.4430236148782547 Benchmark MAE: 645.3367777529196 MSE: 217579645.99200955 RMSE: 14750.581208617155 R2: 0.11069789347971504 MAPE: 175646901033606.78 SMAPE: 0.4771403583560183 Benchmark MAE: 576.8867971572181 MSE: 218104366.05651352 RMSE: 14768.356917968686 R2: -0.09438169165642019 MAPE: 160584994382390.03 SMAPE: 0.5015260051826552"

string = string.replace("MAE:", "&")
string = string.replace("RMSE:", "&")
string = string.replace("MSE:", "&")
string = string.replace("SMAPE:", "&")
string = string.replace("MAPE:", "&")
string = string.replace("R2:", "&")
string = string.replace("Accuracy:", "&")
string = string.replace("Precision:", "&")
string = string.replace("Recall:", "&")
string = string.replace("AUC ROC:", "&")

list = re.split(' & |[0-9] [a-zA-Z]', string)

first = ""
second = ""

i = 0
while i < len(list):
    if benchmark:
        first = first + list[i + 0] + " & " + foo(list[i + 1]) + " & " + foo(list[i + 2]) + " & " + foo(
            list[i + 3]) + " & " + foo(list[i + 4]) + " & " + foo2(list[i + 5]) + " & " + foo(list[i + 6]) + "\\\\ \n"
        i = i + 7
    else:
        first = first + list[i + 0] + " & " + foo(list[i + 1]) + " & " + foo(list[i + 2]) + " & " + foo(
            list[i + 3]) + " & " + foo(list[i + 4]) + " & " + foo2(list[i + 5]) + " & " + foo(list[i + 6]) + "\\\\ \n"
        second = second + list[i + 0] + " & " + foo(list[i + 7]) + " & " + foo(list[i + 8]) + " & " + foo(
            list[i + 9]) + " & " + foo(list[i + 10]) + "\\\\ \n"
        i = i + 11

print(first)
print(second)
