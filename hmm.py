import numpy as np
from hmmlearn import hmm
import pandas as pd
import re
from sklearn.externals import joblib
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf)

outputs=np.array([])
change={}
diff={}
mini_c=0
maxi_c=0
mini_d=0
maxi_d=0
sum_change=[] #離散化に迷ったとき
sum_diff=[]
# for i in range(1,11):
#     filen='./log_hmm/train'+str(i)+'.txt'
with open('./log_hmm/c_l3.txt',"r+") as f: #teach_2はNの数が多すぎて使えない teach_3 c_l3
    line=f.readline()
    while line:
        inte,value=line.split(":")
        #print(value)
        lis=value.replace("(","").replace(")","").replace("\n","")
        n=lis.split(",") #偶数番目が予測誤差，奇数は傾きの変化量
        #print(a)
        for s in range(len(n)): #value=予測誤差diff，傾きの変化量change
            if(s%2==1): #奇数の時
                #print("change")
                sum_change.append(float(n[s]))
                change.setdefault(inte,[]).append(float(n[s]))
            else:
                #print("diff")
                sum_diff.append(float(n[s]))
                diff.setdefault(inte,[]).append(float(n[s]))
        #print(lis)
        #plt.plot([len(sum_change),len(sum_change)],[0,100],label=inte)
        line=f.readline()
    f.close()

    for k in change.keys():
        print("inte:{},shape:{}".format(k,len(change[k])))

#     #=======================================離散値をどうするか迷ったときに
#     for k in diff.keys():
#         for v in range(len(diff[k])):
#             if(change[k][v]>maxi_c): #change 最大値のとき
#                 maxi_c=change[k][v]
#             if(change[k][v]<mini_c): #change 最小値の時
#                 mini_c=change[k][v]
#             if(diff[k][v]>maxi_d): #diff 最大値のとき
#                 maxi_d=diff[k][v]
#             if(diff[k][v]<mini_d): #diff 最小値の時
#                 mini_d=diff[k][v]
#
#             #a.append(diff[k][v])
#     #print(a)
# print("max{}".format(maxi_c))
# print("mini{}".format(mini_c))
# print("ave{}".format((maxi_c-mini_c)/5))
# print("max{}".format(maxi_d))
# print("mini{}".format(mini_d))
# print("ave{}".format((maxi_c-mini_c)/5))
#
# for i in range(1,6):
#     plt.plot([0,30000],[i*13,i*13])
#
# plt.plot(sum_change,'o',label="change")
# plt.plot(sum_diff,'o',label="diff")
# plt.legend() # 凡例を表示
# plt.show()

# print("==================================")
# print("diff {}".format(pd.qcut(sum_diff,5)))
# print("change {}".format(pd.qcut(sum_change,5)))
    #print(np.shape(a))
    #print(pd.cut(a,5))
#=======================================

def Discretization(typ,n): #入力を離散値にする
    # if(n<13.6):
    #     return 0
    # if(n>4*13.6):
    #     return 5
    # for i in range(1,5):
    #     if(n>(i)*13.6 and n<=(i+1)*13.6):
    #         return i

    if(typ=="change"):
        if(n<=8.5):
            return 0
        if(n >8.5 and n<=17.0):
            return 1
        if(n >17.0 and n<=25.5):
            return 2
        if(n >25.5 and n<=34.0):
            return 3
        if(n >=34.0):
            return 4
        #Categories (5, object): [(-0.0384, 7.671] < (7.671, 15.341] < (15.341, 23.0115] < (23.0115, 30.682] <(30.682, 38.353]]
    if(typ=="diff"):
        if(n<=17):
            return 0
        if(n >17 and n<=34):
            return 1
        if(n >34 and n<=51):
            return 2
        if(n >51 and n<=68):
            return 3
        if(n >68):
            return 4
        #Categories (5, object): [(0.43, 13.779] < (13.779, 27.0621] < (27.0621, 40.345] < (40.345, 53.627] <(53.627, 66.91]]

def accuracy(estimation,true):
    count=0
    plus=0
    for n in range(len(true)):
        # if(true[n]==0): #Nの時が入ると精度もクソもないのでだめ
        #     continue
        # plus+=1
        if(estimation[n]!=0 and true[n]!=0): #予測と真の値が同じとき
            count+=1
        if(estimation[n]==0 and true[n]==0):
            count+=1
    print("accuracy {}/{}={}".format(count,len(true),(count/len(true))))
    return count/len(true)
    # print("accuracy {}/{}={}".format(count,plus,(count/plus)))
    # return count/plus


after_=[]
states_=[]
n=0
#intentions=['N','T','H','A','S']
intentions=['N','S','A','H','T']
#もしかして出てくる順番に状態の番号が振られていたら？
#intentions=['H','A','S','T','N']
# temp_diff=[]
# temp_change=[]
# for i in diff.keys():
#     print(i)
#     for s in range(len(diff[i])):
#         temp_diff.append(Discretization("diff",diff[i][s])) #離散化した状態,系列（intentionの添え字）
#         #temp_diff.append(Discretization("diff",diff[i][s]))
#         temp_change.append(Discretization("change",change[i][s]))
#         #temp.append(Discretization("change",change[i][s]))
#         #after_.append(temp_diff)
#         states_.append(i)
#     n+=1


for i in diff.keys():
    print(i)
    for s in range(len(diff[i])):
        temp=[]
        #temp_diff.append(Discretization("diff",diff[i][s])) #離散化した状態,系列（intentionの添え字）
        temp.append(Discretization("diff",diff[i][s]))
        temp.append(Discretization("change",change[i][s]))
        #temp.append(Discretization("change",change[i][s]))
        #print("temp_{}".format(temp))
        after_.append(temp)
        states_.append(i)
    n+=1
#np.random.shuffle(after_)
#print("after {}".format(after_))
#print(np.shape(after_))
#print(np.shape(intentions))
#print([states_.count(intentions[i]) for i in range(5)])

 #NHTSAの順に格納されている

#after_=np.reshape(after_,(563,2)) # 教師データの形のかきかえ
#print(after_)
# after_change=np.reshape(after_change,(50,2))
#

#================================-学習の確認用-===========================
# load_model=joblib.load('estimate_model.pkl')
# predict_=load_model.predict(after_)
# true_=[intentions.index(states_[s]) for s in range(len(states_))]
# print("predict_{}".format(predict_))
# print(states_)
# #print("true_{}".format(true_)) 予測が何が何番で保存されてるかよくわからん！
# #accuracy(predict_,true_)
# print(load_model.score(after_))
#=======================================================================


#===========================-学習用-========================================
#estimate_model_diff=hmm.GaussianHMM(n_components=5, init_params='tcm',params='stcm',covariance_type="diag", n_iter=10000, tol=0.001) #隠れ状態が5のモデルを学習
estimate_model_diff=joblib.load('hidden5_model_0203.pkl')
#estimate_model_diff=hmm.GaussianHMM(n_components=2, init_params='stcm',params='stcm',covariance_type="diag",n_iter=1000) #隠れ状態が5のモデルを学習
#estimate_model_diff.startprob_=np.array([0.3,0.2,0.2,0.2,0.1])
#intentions=['N','S','A','H','T']の順を想定
#estimate_model_diff.means_=np.array([[0,0],[2,2],[2,2],[2,2],[2,2]])
#estimate_model_diff.startprob_=np.array([1.0,0.0])
estimate_model_diff.fit(np.array(after_))
print("diff_transmat:{}".format(estimate_model_diff.transmat_))
print("diff_mean:{}".format(estimate_model_diff.means_))

score,es_state=estimate_model_diff.decode(after_)
print("after_{}".format(after_))
print("predict_{}".format(es_state))
print("true_{}".format(states_))
true_=[intentions.index(states_[s]) for s in range(len(states_))]
print(accuracy(es_state,true_))
print("モデルの対数尤度:{}".format(score))
joblib.dump(estimate_model_diff, "hidden5_model_0203.pkl")


# predict_=estimate_model_diff.predict(after_)
# true_=[intentions.index(states_[s]) for s in range(len(states_))]
# #print(states_)
# #true_=[0 if(intentions.index(states_[s])==0) else 1 for s in range(len(states_))] #2クラス分類にした場合
# print("predict_{}".format(predict_))
# print("true_{}".format(true_))
#
# accuracy(predict_,true_)
# print("model_save")

#joblib.dump(estimate_model_diff, "3_model.pkl")


'''
if(accuracy(predict_,true_)>0.6):
    print("model_save")
    #joblib.dump(estimate_model_diff, "3_model.pkl")
'''
#===================================================================


#
# estimate_model_change=hmm.GaussianHMM(n_components=5, covariance_type="full") #隠れ状態が5のモデルを学習
# emission_model_change.fit(after_change)
# print("change_start:{}".format(emission_model_change.startprob_))


# print("平均")
# for i in change.keys():
#     print(i)
#     print(np.average(change[i]))
# print("----------------")
# for j in diff.keys():
#     print(j)
#     print(np.average(diff[j]))
#
# print("中央値")
# for i in change.keys():
#     print(i)
#     print(np.median(change[i]))
# print("----------------")
# for j in diff.keys():
#     print(j)
#     print(np.median(diff[j]))


#outputs=np.reshape(outputs,(len(outputs)//2,2)) #[[x,y],,,,,,]
#print(outputs[:,0])
# start_prob=np.array([1.0,0.0]) #初期分布
# transmat=np.array([[0.6,0.4],[0.5,0.5]]) #遷移確率
# means = np.array([[1,1],[100,100]]) #近づく1 離れる0
#
# covars = 0.1 * np.tile(np.identity(2),(2,1,1))#共分散
#
# #model = hmm.GaussianHMM(n_components=2, covariance_type="full")
# #X = np.array(outputs[:,0]) #xの学習
# #X=np.reshape(X,(len(X),1))
# model.startprob_ = start_prob
# model.transmat_ = transmat
# model.means_ = means
# model.covars_ = covars
#
# X,Y=model.sample(50)
#remodel=hmm.GaussianHMM(n_components=2,covariance_type="full",init_params='tmc')
#remodel.startprob_=np.array([1.0,0.0])
#remodel.fit(X)
#print(remodel.transmat_)
#print(remodel.means_)
