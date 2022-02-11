import numpy as np
import random as rd
import pandas as pd
from sklearn.metrics.cluster import normalized_mutual_info_score


def calculateStudentKnowledgeOneOrZero(stu, ques, studentMaster, studentAnswerQuestion, Q_matrix, slip_stu, guess_stu):
    AnswerCorrect = studentAnswerQuestion[stu][ques]
    knowledgeOfThisQuestion = Q_matrix[ques]
    currentStudentMaster = studentMaster[stu]
    KnowledgeConceptList = []
    if AnswerCorrect == 0:
        for i in range(len(knowledgeOfThisQuestion)):
            if knowledgeOfThisQuestion[i] == 0:
                KnowledgeConceptList.append("null")
            elif knowledgeOfThisQuestion[i] == 1:
                scoreCorrect = (1-slip_stu)*currentStudentMaster[i] + guess_stu*(1-currentStudentMaster[i])
                if rd.random() < scoreCorrect:
                    KnowledgeConceptList.append(1)
                else:
                    KnowledgeConceptList.append(0)
    else:
        for i in range(len(knowledgeOfThisQuestion)):
            if knowledgeOfThisQuestion[i] == 0:
                KnowledgeConceptList.append("null")
            elif knowledgeOfThisQuestion[i] == 1:
                KnowledgeConceptList.append(1)

    return KnowledgeConceptList
