from pgmpy.models import BayesianNetwork
import numpy as np
import pandas as pd
import random as rd
from pgmpy.sampling import BayesianModelSampling
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianNetwork
import numpy as np
import pandas as pd
import copy
import random as rd
from pgmpy.factors.discrete import TabularCPD
import itertools




def giveSonReturnFather(matrix, nodeName, son_index):
    father_list = []
    nodeNumber = len(nodeName)
    for i in range(nodeNumber):
        if matrix[i][son_index] == 1:
            father_list.append(nodeName[i])

    return father_list


def constructTheBN(matrix, nodeName):

    model = BayesianNetwork()
    for i in nodeName:
        model.add_node(i)

    for i in range(len(nodeName)):
        for j in range(len(nodeName)):
            if matrix[i][j] == 1:
                model.add_edge(nodeName[i], nodeName[j])
    nodeNumber = len(nodeName)

    cpd_list = []
    a = [0, 1]
    for i in range(nodeNumber):
        father_list = giveSonReturnFather(matrix, nodeName, i)
        fatherNumber = len(father_list)
        DescartesArray = []
        if fatherNumber == 0:
            cpd_i = TabularCPD(nodeName[i], 2, [[0.5], [0.5]])
            cpd_list.append(cpd_i)
        else:
            if fatherNumber == 1:
                evidence_card = [2]
                cpd_i = TabularCPD(nodeName[i], 2, [[0.1, 0.9], [0.9, 0.1]], evidence=father_list,
                                   evidence_card=evidence_card)
                cpd_list.append(cpd_i)
            elif fatherNumber == 2:
                x = itertools.product(a, a)
            elif fatherNumber == 3:
                x = itertools.product(a, a, a)
            elif fatherNumber == 4:
                x = itertools.product(a, a, a, a)
            elif fatherNumber == 5:
                x = itertools.product(a, a, a, a, a)
            elif fatherNumber == 6:
                x = itertools.product(a, a, a, a, a, a)
            if fatherNumber > 1:
                for _ in x:
                    DescartesArray.append(list(_))
                probabilityForZero = []
                for j in range(pow(2, fatherNumber)):
                    sum = np.sum(DescartesArray[j])
                    probabilityForZero.append((fatherNumber - sum) * (1 / fatherNumber))
                probabilityForOne = []
                for _ in probabilityForZero:
                    probabilityForOne.append(1 - _)
                evidence_card = []
                for __ in range(fatherNumber):
                    evidence_card.append(2)
                cpd_i = TabularCPD(nodeName[i], 2, [probabilityForZero, probabilityForOne], evidence=father_list,
                                   evidence_card=evidence_card)
                cpd_list.append(cpd_i)

    for _ in cpd_list:
        model.add_cpds(_)

    return model



def constructItemForSubQuestion(model, submodel, evidence, evidenceNode, item_number,rate):

    m = item_number*rate
    nodeList = list(model.nodes())
    subNodeList = list(submodel.nodes())
    diff = []
    for _ in nodeList:
        if _ not in subNodeList:
            diff.append(_)
    samplesFromAll = BayesianModelSampling(model).forward_sample(size=m)
    samplesFromAll = samplesFromAll.drop(diff, axis=1)


    subMatr = giveModelReturnMatrix(submodel)
    leaves = []
    subNodeNumber = len(subNodeList)-1
    for i in range(subNodeNumber):
        for j in range(subNodeNumber):
            if subMatr[i][j] == 1:
                break
        leaves.append(subNodeList[i])
    list_duicuo = []

    for i in range(samplesFromAll.shape[0]):
        flag = 1
        for _ in leaves:
            if samplesFromAll.loc[i,[_]].values[0] == 0:
                list_duicuo.append(0)
                flag = 0
                break
        if flag == 1:
            list_duicuo.append(1)
    samplesFromAll['wrong'] = list_duicuo


    for i in range(item_number):
        evidenceDf = pd.DataFrame([evidence[:-1]], columns=evidenceNode[:-1])
        y_pred = model.predict(evidenceDf,n_jobs = 1)
        y_pred_values = y_pred.values[0]

        new_evidence = []
        countForPredict = 0
        countForEvidence = 0
        for _ in subNodeList[:-1]:
            if _ in evidenceNode:
                new_evidence.append(evidenceDf.loc[0, _])
                countForEvidence += 1
            else:
                new_evidence.append(y_pred_values[countForPredict])
                countForPredict += 1
        new_evidence.append(evidence[-1])
        new_evidenceDf = pd.DataFrame([new_evidence], columns=samplesFromAll.columns.values)

        samplesFromAll = samplesFromAll.append(new_evidenceDf)

    return samplesFromAll


class data_of_BayesianNetwork:
    def __init__(self, matr, nodelist):
        self.matr = matr
        self.nodelist = nodelist



def extractBN(matrix, subNodeList, nodeName):
    nodeSet = set()
    for _ in subNodeList:
        nodeSet.add(_)
    for j in range(len(nodeName)):
        for i in range(len(nodeName)):
            if(matrix[i][j] == 1 and nodeName[j] in nodeSet):
                if (nodeName[i] not in nodeSet):
                    nodeSet.add(nodeName[i])

    new_nodeName = []
    for _ in nodeName:
        if _ in nodeSet:
            new_nodeName.append(_)
    new_nodeName.append('wrong')
    number_subBN = len(new_nodeName)
    subBN = np.zeros((number_subBN,number_subBN))
    for i in range(len(nodeName)):
        for j in range(len(nodeName)):
            if matrix[i][j] == 1:
                if (nodeName[i] in new_nodeName) and (nodeName[j] in new_nodeName):
                    subBN[new_nodeName.index(nodeName[i])][new_nodeName.index(nodeName[j])] = 1


    for i in range(number_subBN-1):
        flag = 1
        for j in range(number_subBN-1):
            if subBN[i][j] == 1:
                flag = 0
                break
        if flag == 1:
            subBN[i][number_subBN-1] = 1

    extractModel = BayesianNetwork()
    for i in new_nodeName:
        extractModel.add_node(i)

    for i in range(number_subBN):
        for j in range(number_subBN):
            if subBN[i][j] == 1:
                extractModel.add_edge(new_nodeName[i], new_nodeName[j])
    return extractModel



def fitModel(model, data):
    model.fit(data)
    return model



def get_prob(model):
    nodelist = list(model.nodes())
    data = []
    for i in range(len(nodelist)):
        data.append(0)
    data = [data]
    new_data = pd.DataFrame(data, columns=nodelist)
    new_data.drop(nodelist, axis=1, inplace=True)
    prob = model.predict_probability(new_data)
    return prob


def get_prob_of_node(prob, node_name, score=False):
    if score:
        node_name = node_name + '_1'
    else:
        node_name = node_name + '_0'
    if node_name not in prob.columns:
        return 0.001
    return prob[node_name].values[0]


def getTheErorAttributionGraph(model):
    nodelist = list(model.nodes())

    nodeNumber = len(nodelist)
    matrixAll = np.zeros((nodeNumber, nodeNumber))
    for i in range(nodeNumber):
        for j in range(nodeNumber):
            if (nodelist[i], nodelist[j]) in model.edges:
                matrixAll[i][j] = 1

    inference = VariableElimination(model)
    matrixErrorAttribution = np.zeros((nodeNumber, nodeNumber))

    weight_ = []

    if "wrong" in nodelist:
        for i in range(nodeNumber):
            nameList = []
            for j in range(nodeNumber):
                if (matrixAll[j][i] != 0):
                    nameList.append(nodelist[j])
            if len(nameList) == 0:
                continue

            weightList = []
            if nodelist[i] != "wrong":
                for node in nameList:
                    phi_query = inference.query(variables  = [nodelist[i], node], evidence={'wrong':0})
                    weight = phi_query.values[0][0]
                    weight_.append(weight)
                    if weight == 0:
                        weight = -1
                    weightList.append(weight)
            else:
                for node in nameList:
                    phi_query = inference.query(variables  = [nodelist[i], node])
                    weight = phi_query.values[0][0]
                    weight_.append(weight)
                    if weight == 0:
                        weight = -1
                    weightList.append(weight)

            for j in range(nodeNumber):
                if (matrixAll[j][i] != 0):
                    matrixErrorAttribution[j][i] = weightList[0]
                    weightList.pop(0)


        sum_weight = np.sum(weight_)
        threshold = np.mean(weight_)


        data_of = data_of_BayesianNetwork(matrixErrorAttribution, nodelist)
    else:
        for i in range(nodeNumber):
            nameList = []
            for j in range(nodeNumber):
                if (matrixAll[j][i] != 0):
                    nameList.append(nodelist[j])
            if len(nameList) == 0:
                continue
            weightList = []
            for node in nameList:
                phi_query = inference.query(variables=[nodelist[i], node])
                weight = phi_query.values[0][0]
                weight_.append(weight)
                if weight == 0:
                    weight = -1
                weightList.append(weight)

            for j in range(nodeNumber):
                if (matrixAll[j][i] != 0):

                    matrixErrorAttribution[j][i] = weightList[0]
                    weightList.pop(0)


        threshold = np.mean(weight_)



        data_of = data_of_BayesianNetwork(matrixErrorAttribution, nodelist)

    return data_of, threshold



def constructOneItemWithEvidence(model, evidence, evidenceNode):
    nodeList = list(model.nodes)
    data_construct = []
    count = 0
    evidence_delete_cuo = evidence[:-1]
    evidenceNode_delete_cuo = evidenceNode[:-1]
    for _ in nodeList:
        if _ in evidenceNode_delete_cuo:
            data_construct.append(evidence_delete_cuo[count])
            count += 1
        else:

            data_construct.append(0)

    df = pd.DataFrame([data_construct], columns=nodeList, index=[0])

    nodeList_copy = nodeList.copy()
    data_copy = df.copy()

    knowledge_should_predict = list(set(nodeList) - set(evidenceNode_delete_cuo))
    data_copy.drop(knowledge_should_predict, axis=1, inplace=True)

    prob = model.predict_probability(data_copy)
    predict_length = len(knowledge_should_predict)
    each_knowledge_rate = []
    for i in range(predict_length):
        each_knowledge_rate.append(prob.iat[0, 2*i])

    countForAll = 0
    countForPredict = 0
    for _ in nodeList:
        if _ in evidenceNode_delete_cuo:
            countForAll += 1
            continue
        else:
            if rd.random() < each_knowledge_rate[countForPredict]:
                data_construct[countForAll] = 0
            else:
                data_construct[countForAll] = 1
            countForPredict += 1
            countForAll += 1
    return data_construct


def giveModelReturnMatrix(model):
    nodes = list(model.nodes)
    nodes_number = len(nodes)
    matrix = np.zeros((nodes_number, nodes_number))
    for i in range(nodes_number):
        for j in range(nodes_number):
            if (nodes[i], nodes[j]) in model.edges:
                matrix[i][j] = 1

    return matrix


def read_from_csv(filename):
    read_csv2 = pd.read_csv(filename)
    list_name = read_csv2.columns.tolist()[1:]
    samples = read_csv2[list_name]
    return (samples, list_name)

def sub_causal(matrix, evidence, nodeName, model, evidenceNode, item_number, plot, file_path,rate):
    submodel = extractBN(matrix, evidenceNode, nodeName)
    samples = constructItemForSubQuestion(model, submodel, evidence, evidenceNode, item_number,rate)
    submodel.fit(samples)
    data_of, threshold = getTheErorAttributionGraph(submodel)
    return data_of.matr, data_of.nodelist


def all_causal(item_number, evidence, model, evidenceNode, plot, file_path, rate):
    samples = []
    for i in range(item_number):
        samples.append(constructOneItemWithEvidence(model, evidence, evidenceNode))
    samples = pd.DataFrame(samples, columns=list(model.nodes))
    model.fit_update(samples, n_prev_samples=rate*item_number)
    return model




