import pandas as pd
import numpy as np
import math
include <data.py>

def p_ai_to_ai (ai):
    ai_ai = 0
    without_last = ai[:-1]
    sum_ = np.sum(without_last)

    if sum_ != 0:
      for i in range(len(ai)-1):
          ai_ai += (ai[i]/sum_)*ai[i+1]
    return ai_ai

def p_ai_to_aj(ai,aj):
    ai_aj = 0
    without_last = ai[:-1]
    sum_ = np.sum(without_last)

    if sum_ != 0:
      for i in range(len(ai)-1):
          ai_aj += (ai[i]/sum_)*aj[i+1]
    return ai_aj


def create_matrix(matrix):
    #create empty matrix
    len_ = (8)
    transtion_matrix = np.zeros((len_,len_))

    #convert matrix to array for each column
    matrix = matrix.transpose()

    for i in range(len_):
        for j in range(len_):
            if i==j:
                transtion_matrix[i][j] =  p_ai_to_ai(matrix[i])
            else:
                transtion_matrix[i][j] =  p_ai_to_aj(matrix[i],matrix[j])
            # if i>j:
            #     transtion_matrix[i][j] =  p_aj_to_ai(matrix[i],matrix[j])
    transtion_matrix = normalization(transtion_matrix)
    return transtion_matrix

def create_trans_matrix(arr):
  #create empty matrix
  matrix = np.zeros((7,7))
  size = len(arr)
  for i in range(size-1):
    if arr[i] == arr[i+1]:
      continue
    else:
      matrix[arr[i]][arr[i+1]] += 1
  
  norm_matrix = normalization(matrix)
  #print(norm_matrix)
  return norm_matrix

def filterSoftmax(matrix):
  filter_matrix = []
  size = len(matrix)
  for r in range(len(matrix)):
    print(matrix[r])
    new_row = np.zeros(len(matrix[0]))
    max_idx = np.argmax(matrix[r], axis = 0)
    max_val = np.amax(matrix[r], axis = 0)
    new_row[max_idx] = max_val
    mean = max_val
    tmp = matrix[r]
    for i in range(len(matrix[0])):
      tmp[max_idx] = 0
      max_idx = np.argmax(tmp, axis = 0)
      max_val2 = np.amax(tmp, axis = 0)
      if (abs(mean-max_val2) > 0.8):
        break
      else:
        mean = (mean + max_val2)/2
        new_row[max_idx] = max_val2
    filter_matrix.append(new_row)
  mark_bg = markBgAs1(filter_matrix)
  return np.array(mark_bg)

def markBgAs1(matrix):
  bg_matrix = []
  count = 0
  for i in range(len(matrix)):
    count = np.count_nonzero(matrix[i])
    if count >= 5:
      bg_row = np.zeros(len(matrix[0]))
      bg_row[0] = 1
      bg_matrix.append(bg_row)
    else:
      bg_matrix.append(matrix[i]) 
  return bg_matrix

def DiagonalToZero (matrix):
    len_ = (matrix.shape[1])
    for i in range( len_):
        for j in range( len_):
            if (i == j):
                matrix[i][j] = 0
    return matrix

def normalization (matrix):
    len_ = (matrix.shape[1])
    m_diagnoal = DiagonalToZero(matrix)
    new_m = np.zeros((len_,len_))

    for i in range(len_):
        sum_ = np.sum(m_diagnoal[i])
        if sum_ != 0:
          for j in range(len_):
            if i!=j:
                new_m [i][j] = m_diagnoal[i][j]/sum_
    return new_m

def probabilityState(matrix):
    # Compute the stationary distribution of states.
    eigvals, eigvecs = np.linalg.eig(matrix.T)
    eigvec = np.real_if_close(eigvecs[:, np.argmax(eigvals)])

    # normalization
    if eigvec.sum() != 0:
      stat_distr = eigvec / eigvec.sum()
    else:
      stat_distr = 0 
    return stat_distr

def GtwithoutBg(gt):
  gt_without_bg = []
  for i in gt:
    if i != -1:
      gt_without_bg.append(i)
  return gt_without_bg


def KLsum(first, second):
    return first + second

def KLDistance(P, Q, statesq):
    # diagnoal = DiagonalToZero(P)
    # P_nn = normalization(diagnoal)
    # statessp = probabilityState(P_nn)
    P_nn=P
    statessp = probabilityState(P)
    statesp = np.where(statessp<=0, 0, statessp)
    KLa=0
    KLb= 0
    for i in range(len(statesp)):
        if statesp[i] !=0 and statesq[i]!=0 :
            KLa += abs(statesp[i]*math.log2((statesp[i]/statesq[i])))
#     KLa = np.sum(np.where(statesp != 0, statesp * np.log(statesp / statesq), 0))

    for i in range(P_nn.shape[1]):
        sum_ =0
        for j in range(P_nn.shape[1]):
             if P_nn[i][j] !=0 and  Q[i][j] !=0:
            #     if(P_nn[i][j]*math.log2((P_nn[i][j]/Q[i][j]))<0):
            #         sum_ += (-1*P_nn[i][j]*math.log2((P_nn[i][j]/Q[i][j])))
            #     else:
                sum_ += abs(P_nn[i][j]*math.log2((P_nn[i][j]/Q[i][j])))
        KLb += statesp[i] * sum_
    return KLsum(KLa,KLb)

def MSEDistance (P, Q, statesq, prob):
    MSE_d = 0
    if prob == True :
      P_nn=P
      statessp = probabilityState(P)
      statesp = np.where(statessp<=0, 0, statessp)
      for i in range(len(statesp)):
        MSE_d += ((statesp[i] - statesq[i])) ** 2
    else:
      statesp = P
      statesq = Q
      for i in range(len(statesp)):
        for j in range(len(statesp[i])):
        #if i != 0 and j != 0:
       # if statesp[i] !=0 and statesq[i]!=0 :
          MSE_d += ((statesp[i][j] - statesq[i][j])) ** 2
    return MSE_d

def txt_to_matrix(text):
    text = ((",".join(text.split(' '))).replace(',,', ', ').replace(',', ', '))
    #matrix = ast.literal_eval(text)
    return text

def calc_dist(transtion_NN, statesq,prob):
    classes = ['coffe','tea','peanut','cheese','pealate','cofhoney','hotdog']
    for i in range(classes):
        print(classes[i], " :")
        KL_1 = MSEDistance(S1_Peanut_C1_transtion, transtion_NN[i], statesq[i], prob)
        KL_2 = MSEDistance(S2_Peanut_C1_transtion, transtion_NN, statesq, prob)
        KL_3 = MSEDistance(S4_Cheese_C1_transtion, transtion_NN, statesq, prob)
        KL_4 = MSEDistance(S1_Cheese_C1_transtion, transtion_NN, statesq, prob)
        KL_5 = MSEDistance(S4_Pealate_C1_transtion, transtion_NN, statesq, prob)
        KL_6 = MSEDistance(S1_Pealate_C1_transtion, transtion_NN, statesq, prob)
        KL_7 = MSEDistance(S1_CofHoney_C1_transtion, transtion_NN, statesq, prob)
        KL_8 = MSEDistance(S2_CofHoney_C1_transtion, transtion_NN, statesq, prob)
        KL_9 = MSEDistance(S2_Hotdog_C1_transtion, transtion_NN, statesq, prob)
        KL_10 = MSEDistance(S1_Hotdog_C1_transtion, transtion_NN, statesq, prob)
        KL_11 = MSEDistance(S2_Tea_C1_transtion, transtion_NN, statesq, prob)
        KL_12 = MSEDistance(S3_Tea_C1_transtion, transtion_NN, statesq, prob)
        KL_13= MSEDistance(S1_Coffee_C1_transtion, transtion_NN, statesq, prob)
        KL_14= MSEDistance(S4_Coffee_C1_transtion, transtion_NN, statesq, prob)
        video_names ={'peanut':KL_1,'peanut':KL_2,'cheese':KL_3,'cheese':KL_4,'pealate':KL_5,'pealate':KL_6,'cofhoney':KL_7,'cofhoney':KL_8,'hotdog':KL_9,'hotdog':KL_10,'tea':KL_11,'tea':KL_12,'coffe':KL_13,'coffe':KL_14}

        distances1 = [KL_1,KL_2,KL_3,KL_4,KL_5,KL_6,KL_7,KL_8,KL_9,KL_10,KL_11,KL_12,KL_13,KL_14]
        min_dist = min(video_names, key=video_names.get)
        print('minimum distance - ', min_dist, ' : ', min(distances1))
    return


def main():
    gt = [S1_Peanut_C1_transtion, S2_Peanut_C1_transtion, S4_Cheese_C1_transtion, S1_Cheese_C1_transtion, S4_Pealate_C1_transtion, S1_Pealate_C1_transtion, S1_CofHoney_C1_transtion, S2_CofHoney_C1_transtion, S2_Hotdog_C1_transtion, S1_Hotdog_C1_transtion, S2_Tea_C1_transtion, S3_Tea_C1_transtion, S1_Coffee_C1_transtion, S4_Coffee_C1_transtion]
    softmax = [[observation1, observation2], [observ3, obser4], [obser5, obser6], [obser7, obser8], [obser9, obser10], [obser11, obser12], [obser13, obser14]]
    
    # filter softmax matrices. comment this loop to run on original softmax
    for count,x in enumerate(softmax):
        softmax[count][0] = filterSoftmax(x[0])
        softmax[count][1] = filterSoftmax(x[1])
    
    for count,x in enumerate(softmax):
        transtion_NN[count] = create_matrix(x[0])
        transtion_NN2[count] = create_matrix(x[1])
        #probality to each state
        statesq[count] = probabilityState(transtion_NN[count])
        statesq2[count] = probabilityState(transtion_NN2[count])
    
    # False = comparing 2 matrices cell-to-cell. True = comparing eigenvectors.
    calc_dist(transtion_NN, statesq, True)
    calc_dist(transtion_NN2, statesq2, True)
