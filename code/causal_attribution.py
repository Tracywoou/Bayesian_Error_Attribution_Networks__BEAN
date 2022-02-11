import pandas as pd
import numpy as np
from GetBN import *
from Plot_dag import *
from datetime import datetime
import os
import csv

def get_evd(data_name, value, file_name, kn_list, suyin_timu):
    df = pd.read_csv(file_name)
    list_name = [data_name, 'is_right_answer']
    for kn in kn_list:
        list_name.append(kn)
    data = df[list_name]
    data = data.loc[df[data_name] == value]
    evidence_list =[]
    evidence_node_list = []
    suyin_list = []
    suyin_node_list = []
    exercise_code_list = []
    doAttributeOrnot = []
    for i in data.index:
        if df.loc[i, 'exercise_code'] not in suyin_timu:
            evidence = []
            evidence_node = []
            for kn in kn_list:
                value = df.loc[i, kn]
                if not pd.isnull(value):
                    evidence_node.append(kn)
                    evidence.append(int(value))
            evidence.append(df.loc[i, 'is_right_answer'])
            evidence_node.append("wrong")
            evidence_list.append(evidence)
            evidence_node_list.append(evidence_node)
            doAttributeOrnot.append(0)
        else:
            exercise_code_list.append(df.loc[i, 'exercise_code'])
            suyin_evidence = []
            suyin_evidence_node = []
            for kn in kn_list:
                value = df.loc[i, kn]
                if not pd.isnull(value):
                    suyin_evidence_node.append(kn)
                    suyin_evidence.append(int(value))
            suyin_evidence.append(df.loc[i, 'is_right_answer'])
            suyin_evidence_node.append("wrong")
            suyin_list.append(suyin_evidence)
            suyin_node_list.append(suyin_evidence_node)
            doAttributeOrnot.append(1)
    return evidence_list, evidence_node_list, suyin_list, suyin_node_list,exercise_code_list,doAttributeOrnot


def run(dataname, id, suyin_timu,csvFilePath):
    rate = 5
    file_path = 'output/' + dataname + "_" + str(id) + '/'
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    f_log = open(file_path + "log_.log", "a")
    f_evidence = open(file_path + "evidence.log", "a")
    f_suyin = open(file_path + "suyin.log", "a")
    file_name = "../rankData/QA_Generate.csv"
    kn_list = ["Set","Inequality","Trigonometric_function","Logarithm_versus_exponential","Plane_vector",
              "Property_of_function","Image_of_function","Spatial_imagination","Abstract_summarization",
              "Reasoning_and_demonstration","Calculation"]
    matrix = [[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0]]

    evidence_list, evidence_node_list, suyin_list, suyin_node_list,exercise_code_list,doAttributeOrnot= get_evd(dataname, id, file_name, kn_list, suyin_timu)
    index = 0

    lastSuyinIndex = -1
    for i in range(len(doAttributeOrnot)):
        if doAttributeOrnot[i] == 1:
            lastSuyinIndex = i

    masterFileName = "./studentMasterOnEachKnowledge.txt"
    studentMaster = np.loadtxt(masterFileName, dtype=np.float)
    student_id_each_knowle_master_prob = studentMaster[id]

    model = constructTheBN(matrix, kn_list)
    item_number = 500
    item_number_sub = 50
    countOfexercise_code_list = 0
    countOfEvidence = 0
    counfOfAttribute = 0
    answerItemNumber = len(doAttributeOrnot)
    suyin_evidence_number = len(suyin_list)
    if dataname == 'user_id':
        for i in range(answerItemNumber):
            if doAttributeOrnot[i] == 0:
                evidence = evidence_list[countOfEvidence]
                evidenceNode = evidence_node_list[countOfEvidence]
                model = all_causal(item_number, evidence, model, evidenceNode, False, file_path, rate)
                countOfEvidence += 1
            elif doAttributeOrnot[i] == 1:
                start_time = datetime.now()
                timu_code = exercise_code_list[countOfexercise_code_list]
                countOfexercise_code_list += 1
                suyin_evidence = suyin_list[counfOfAttribute]
                suyin_evidenceNode = suyin_node_list[counfOfAttribute]

                model = all_causal(item_number, suyin_evidence, model, suyin_evidenceNode, False, file_path, rate)
                if suyin_evidence[-1] != 1:
                    write_node = ""
                    for node in suyin_evidenceNode:
                        write_node += node
                        write_node += " "
                    f_evidence.write("exerciseCode: " + str(timu_code) + "\n")
                    f_evidence.write("evidence: " + str(suyin_evidence) + "\n")
                    f_evidence.write("evidenceNode: " + write_node + "\n")
                    f_log.write("exerciseCode: " + str(timu_code) + "\n")
                    f_log.write("evidence: " + str(suyin_evidence) + "\n")
                    f_log.write("evidenceNode: " + write_node + "\n")
                    print(suyin_evidence)
                    print(suyin_evidenceNode)
                    mat,nodelistOfMat = sub_causal(matrix, suyin_evidence, kn_list, model, suyin_evidenceNode, item_number_sub, True,
                                     file_path,rate)

                    currentProblemMaster = []



                    countOfSum = 0
                    sumOfMat = 0
                    maxWrongProbnode = []
                    for j in range(len(mat)):
                        for k in range(len(mat)):
                            if mat[j][k] != 0:
                                countOfSum += 1
                                sumOfMat += mat[j][k]
                    averageOfMat = sumOfMat/countOfSum
                    for j in range(len(mat)):
                        for k in range(len(mat)):
                            if mat[j][k] >= averageOfMat:
                                maxWrongProbnode.append(nodelistOfMat[j])


                    for eachNode in nodelistOfMat[:-1] :
                        currentProblemMaster.append(student_id_each_knowle_master_prob[kn_list.index(eachNode)])
                    min_master_prob_knowle = nodelistOfMat[currentProblemMaster.index(min(currentProblemMaster))]

                    with open(csvFilePath, 'a+',newline='') as f:
                        f_csv = csv.writer(f)
                        f_csv.writerow([id, i, maxWrongProbnode, min_master_prob_knowle, min_master_prob_knowle in maxWrongProbnode])
                        f.close()

                    mat_str = ""
                    mat_str += "[ "
                    for i in range(len(suyin_evidenceNode)):
                        for j in range(len(suyin_evidenceNode)):
                            mat_str = mat_str + str(mat[i][j]) + " "
                        if i != suyin_evidence_number - 1:
                            mat_str += "\n"
                    mat_str += " ]\n"
                    f_suyin.write(mat_str)
                    f_log.write(mat_str)
                end_time = datetime.now()
                print('Duration: {}'.format(end_time - start_time))
                counfOfAttribute += 1


csvFilePath = "./predictAndGroundtruth.csv"
headers = ['user_id', 'problem_id', 'perdict_wrong_knowledge', 'true_wrong_knowledge', 'is_right']
with open(csvFilePath, 'a+',newline='') as f:
    f_csv = csv.writer(f)
    f_csv.writerow(headers)
    f.close()

filename = "../rankData/QA_Generate.csv"
df = pd.read_csv(filename)

for user_id in range(0,4209):
    print(user_id)
    dataname = "user_id"
    isCorrectdata = df.loc[df["user_id"]==user_id]['is_right_answer']
    isCorrectdata = isCorrectdata.values
    student_suyin_timu = []
    for i in range(len(isCorrectdata)):
        if isCorrectdata[i] == 0:
            student_suyin_timu.append(i)


    run(dataname, user_id, student_suyin_timu,csvFilePath)