import cv2
import os
import math
from skimage import measure
import numpy as np
from matplotlib import pyplot as plt
import csv
from scipy.signal import argrelextrema

ssim = []
DOGa = []
frameindex = [0,1]
xa = []
DOGb = []
F = [0,0]
Fa = []
xc = []
DOGc = []
Fc = []
svmindexp = [0,1]
svmindexn = [0,1]
DBFa = []
DBFb = []
DBFc = []

with open('./test.csv','r') as csvfile:
  reader = csv.reader(csvfile)
  column = [row[1] for row in reader]
  column.pop(0)
  column = list(map(int,column))

index = [x for x, e in enumerate(column) if e != 0]
# print(index)

with open('./test2.csv','r') as csvfile2:
  reader2 = csv.reader(csvfile2)
  column2 = [row[1] for row in reader2]
  column2.pop(0)
  column2 = list(map(int,column2))

index2 = [x for x, e in enumerate(column2) if e != 0]

with open('./test3.csv','r') as csvfile3:
  reader3 = csv.reader(csvfile3)
  column3 = [row[1] for row in reader3]
  column3.pop(0)
  column3 = list(map(int,column3))

index3 = [x for x, e in enumerate(column3) if e != 0]
if __name__ == "__main__":
    path = '/Users/hjyyyyy/Desktop/crew/'
    fileList = os.listdir(path)
    fileList.sort()
    # print(fileList)
    for i in range(2, 599):
        frameindex.append(i)
        # print(i)
        frame0 = cv2.imread(path + os.sep + fileList[i - 2])
        frame1 = cv2.imread(path + os.sep + fileList[i - 1])
        frame2 = cv2.imread(path + os.sep + fileList[i])
        frame3 = cv2.imread(path + os.sep + fileList[i + 1])

        frame0_gray = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
        frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        frame3_gray = cv2.cvtColor(frame3, cv2.COLOR_BGR2GRAY)
        # hist0 = np.histogram(frame0_gray, bins=64)
        # hist1 = np.histogram(frame1_gray, bins=64)
        # hist2 = np.histogram(frame2_gray, bins=64)
        # hist3 = np.histogram(frame3_gray, bins=64)
        # hd1 = np.subtract(hist1, hist0)
        # hd2 = np.subtract(hist2, hist1)
        # hd3 = np.subtract(hist3, hist2)
        # n = np.linspace(0, 256, 64)
        # # plt.plot(n, hd1[0],c='blue')
        # plt.plot(n, hd2[0],c='red')
        # plt.plot(n, hd3[0],c='blue')
        # plt.show()
        # cv2.imshow('frame1_gray',frame1_gray)
        # cv2.imshow('frame2_gray',frame2_gray)
        # cv2.waitKey(0)


        DBF = np.sum(cv2.absdiff(frame2_gray, frame1_gray))/(720*1280)
        DOG = np.sum(frame2_gray.astype(np.float32) \
                     - frame1_gray.astype(np.float32))/(720*1280)
        if DOG > 0.5:
            sign = 1
        elif DOG < - 0.5:
            sign = -1
        else:
            sign = 0
        # print(DOG)
        DOGb.append(DOG)
        DBFa.append(DBF)
        score = 2/3 * np.sum(cv2.absdiff(frame2_gray, frame1_gray))/(720*1280) \
                + 1/3 * np.sum(cv2.absdiff(frame0_gray, frame3_gray))/(720*1280)
        feature = sign*DBF/score
        F.append(feature)
        # print(i, DOG)
        if feature > 0:
            svmindexp.append(i)
        # elif feature <= -0.5:
        #     svmindexn.append(i)
        if i in index2:
            xa.append(i)
            Fa.append(feature)
            DOGa.append(DOG)
            DBFb.append(DBF)
            # plt.plot(n, hist[0],c = 'red')
            # plt.show()
        elif i in index3:
            xc.append(i)
            Fc.append(feature)
            DOGc.append(DOG)
            DBFc.append(DBF)

            # plt.plot(n, hist[0],c = 'green')
            # plt.show()
        # else:
        #     plt.plot(n, hist[0])
        #     plt.show()
        # feature = DBF/score
        # i += 1
    # print(F)
    # np.r_[True, F[1:] < F[:-1]] & np.r_[F[:-1] < F[1:], True]
    local_maxima = argrelextrema(np.array(F), np.greater_equal, order=4, mode='clip')
    lmaxindex = list(local_maxima[0])
    # print(local_maxima)
    local_minima = argrelextrema(np.array(F), np.less_equal, order=4, mode='clip')
    lminindex = list(local_minima[0])

    candidatep = list(set(svmindexp).intersection(set(lmaxindex)))
    candidaten = list(set(svmindexn).intersection(set(lminindex)))
    candidate = candidaten + candidatep
    candidate.sort()
    print(candidate)
    candidateF = []
    # for m in range(599):
    #     if m in candidate:
    #         candidateF.append(F[m])
    #     m += 1
    # F = [0, 0, 1.2957989743364382, -1.297806654251789, 0.0, 0.0, -0.8260990879559856, \
    #      -0.8029620427058345, -0.7868908729121767, -0.8113989361200791, 0.0, \
    #      0.7864300300557417, 1.3029314961798515, -1.1687627885940728, 0.0, 0.0, \
    #      0.0, -0.8416399502348026, -0.8155503829093204, -0.8241722612687211, 0.0,\
    #      0.0, 0.0, 0.0, 0.0, 0.8471964103963717, 0.4605404034148713, 1.0984042452122054,\
    #      -0.41730591429950675, 1.3062611982178227, -1.1797791127620423, 0.4559820728626036,\
    #      -1.1019476481348958, 0.0, 0.0, -0.8257991630438576, -0.7901337848646223, \
    #      -0.4235145093357563, 1.2982046774956484, -1.3110917868349565, 0.0, 0.0, \
    #      1.11211413664284, -1.1253434798522326, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,\
    #      0.0, 0.0, 0.8256059802332517, 0.824555483372384, 0.834886949009723, \
    #      0.8563027203881864, 0.870396197264303, 0.0, 0.0, 0.0, 1.2549651902736578, \
    #      -0.6850066278574521, 1.212094344863361, -1.4034335448231046, 0.0, 0.0, 0.0,\
    #      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.1242466288991753, -1.134279896375857, 0.0,\
    #      0.0, 0.0, 0.0, 0.0, -0.8313738192181298, -0.8305650790170594, 0.0, 0.0, 0.0,\
    #      0.0, 0.0, 0.0, 1.1463854107926736, -1.157936670130932, 0.0, 0.0, 0.0, 0.0, \
    #      0.0, -0.8105279187743464, -0.8121466519847967, -0.8228692833278927, \
    #      -0.8347758636906537, -0.8399210155753173, -0.8308454292966057, -0.8217260225244117,\
    #      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,\
    #      0.0, 0.3133007381889132, 1.3503383927459702, -1.343975929169963, 0.0, 0.0,\
    #      0.9809296402906897, -1.0150652313611683, -0.6372517327162229, -0.8134758953325935,\
    #      -0.8202885371901661, -0.8337960324875926, 0.0, 0.0, 0.0, 0.0, 0.8235003923009883, \
    #      0.7944363817995458, 0.7928853897406799, 0.801590962823714, 0.8031183052454925, \
    #      0.8089412946027087, 0.8110400675406281, 0.8150568859749685, 0.0, 0.0, 0.0, 0.0,\
    #      0.0, 0.9876098369499117, -0.9843519377551146, 0.0, 0.0, 0.0, 0.0, 1.3342918904039585,\
    #      -1.3380934984145594, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, \
    #      1.281485713520684, -1.2841577750578381, 0.0, 0.0, 1.3400568763132172, \
    #      -1.3395972785435584, 0.0, 0.0, 0.0, 1.2542980897522065, -1.2571938750767297, \
    #      0.0, 0.5267386108211236, 1.203333470016166, -1.1921157575240677, -0.5121359512189421,\
    #      -0.7694821558131013, 0.8768756588102695, -0.9255200682015456, -0.6886848890906847,\
    #      -0.7510561308541378, 0.9751430327954156, -0.9961700908157209, 0.0, 0.0, 0.0, 0.0, \
    #      0.8024444036377275, 0.7900078162956556, 0.7910446842608787, 0.7945889044921738,\
    #      0.29662164446348216, 1.2319639814119534, -1.3026742543501253, -0.5930501671289383,\
    #      0.0, 1.0340527265744652, -1.032328992710567, 0.0, 0.9713108926379316, \
    #      -0.8858264700745376, 0.7020639208189687, 0.9122383159653576, -0.9227544822699055, \
    #      -0.9232424150157098, 0.0, -0.8091028417509623, -0.7984876449759449, 0.8445388262740623,\
    #      -0.8993043813595477, -0.5189907187956838, 0.8094768202707051, 1.2032652296321449, \
    #      -1.1282008744679173, 0.3181927866653282, 0.0, 0.0, 0.0, 0.0, 0.7987286923617709,\
    #      0.7829061069182544, 0.7883353814652021, 0.7928246508068831, 0.7987300163796514, \
    #      0.8087196971212152, 0.0, 0.0, 0.0, 0.9866641349820611, -0.9836177119738463, 0.0,\
    #      0.0, 0.0, 0.0, 0.9215277218332919, -0.9217200613669287, 0.0, 0.9689941874355436, \
    #      -0.9606458845540229, -0.7628669946637561, -0.8002432452477539, 0.0, 1.1717124872616096,\
    #      -1.1695631518318046, -0.5384008707482993, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, \
    #      1.3999514343201986, -1.384790907731426, 0.2918712721308657, -0.9248519969627598, 0.0, \
    #      0.0, 0.0, 0.0, 0.0, 0.0, 1.0463377840863204, -1.0507169542399402, 0.0, 0.0, 0.0, \
    #      1.3128922551854083, -1.253533704110511, 0.5382342268298261, -1.0112046704625515, \
    #      0.0, 0.0, -0.5419108773102043, 1.1578798551517735, -1.1840980363000009, \
    #      -0.4699071492981563, 0.0, 0.0, 0.0, 0.0, 1.2363776568964857, -1.2459224846049524,\
    #      -0.4302999895610309, -0.7692247698915964, 0.865954080643027, -0.9143942817750325, \
    #      0.8141467122847891, 0.9033891186537165, -0.9574731836802903, 0.0, 0.0, 0.0, \
    #      1.3039661000592018, -1.3040442654813893, 0.0, 0.8116989483466963, 0.0, 0.0, 0.0,\
    #      0.0, 0.7272104624491597, 0.9090148145151791, -0.8899517342816945, 0.0, 0.0, 0.0, \
    #      1.2909677373648942, -1.17920321601726, 0.7683493184488479, 0.9784499691654591, \
    #      -0.9449004682603456, -0.6622741359894397, -0.7202358161437163, -0.7867457252164722,\
    #      -0.786385853249506, -0.7968917943977853, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, \
    #      0.8778409010011891, -0.8878752662273627, 0.0, 0.0, 0.8982978295126767, \
    #      -0.8897683163512858, -0.7539520636004752, -0.7907626332570642, -0.7862644226984156, -0.7822122520498197, -0.7810505393424754, -0.7654502300266985, 0.8193480570129659, 0.6596757996887461, 1.0291724997645462, -0.969596761620905, 0.7539117428915172, -0.9943239247546334, 0.0, 1.3163860096265294, -1.3217829281440812, 0.0, 0.9134748044481468, -0.9334518991227658, 0.0, -0.4351548465765744, 1.2747430798393882, -1.2761941142898257, 0.4546938869750048, -0.8596429913032084, 0.8579315842105852, -0.9056338108694872, 0.0, -0.8334321060918334, 0.0, -0.824822020680318, 0.0, -0.8215536477991552, 0.0, 0.8357502906741885, 0.0, 0.0, 0.9085742196769159, -0.9630905904405053, 0.0, 1.050903614737271, -1.0346400282962263, 0.7187429727208604, -0.849938990437969, 0.8522827043127166, -0.835520705597745, 0.0, 0.678785453107668, 1.0435027331158855, -1.0529714287178857, 0.7069234967364358, 0.8664028652554275, -0.8508889162037357, 0.0, 0.0, 0.0, 0.669259953550997, 1.1008586610897502, -1.1226353280062265, 0.6745718471580855, 1.1593840449620976, -1.1786329914725198, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9231955303779559, -0.9127595000728902, 0.8121102813736921, -0.8353430323928722, 0.8503524991963074, -0.8656958558019604, -0.7937141284123899, -0.8161232087736568, 0.8332180113399569, -0.810931819468594, 0.7264259882387949, 1.0240558386945493, -1.0267795349543583, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.835884098012719, -0.8337922635995288, 0.0, 0.832659798054725, 0.0, 0.0, 0.833312870052899, -0.8020547250193695, 0.8726595313219635, -0.8709453315203468, 0.0, 0.8441357659224019, -0.7795275194201812, 0.8825290507878081, -0.4331285849567186, 1.301683203680262, -1.2788530982070936, 0.427136441530931, -0.9050878051399951, 0.0, 0.0, 0.0, 0.9051623534113198, -0.9105436343917189, 0.0, 0.0, 0.0, 0.9960284533521858, -0.9701650380865643, 0.7905972761888067, 0.855565432421271, -0.8181066889680392, 0.8128960706299112, 0.8124548555463528, 0.8196379587477265, 0.8171728378402454, 0.8221105225301508, 0.0, 0.0, 0.8965973518331691, -0.8941834243766089, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9983597386778568, -0.9842957326737779, 0.0, -0.6188929095302896, 1.1197729880772496, -1.1143432504292685, 0.0, 0.7454751017111853, 0.9447393326135262, -0.8669190440315205, 0.9849247167007027, -1.0897223296291718, 0.0, 0.0, 0.8116919941983848, 0.0, 0.0, 0.7874922453146782, 0.8690126988079189, -0.8833663140686948, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8102415645474645, -0.8113304844503372, 0.0, 0.8352461146957313, -0.8354383109804928, 0.7871365345680524, 0.0, 0.0, -0.7948153118168999, 0.0, 0.0, 0.8373839465484025, -0.8409289960299698, -0.5783089169260338, 1.1498808160519551, -1.0508476450034254, 0.7719699016772941, -0.9774282358388123, -0.6826035657592511, 0.0, 0.8179993388283243, -0.8263956817177489, 0.0, 0.0, 0.8573164069396337, -0.8628195795619655, 0.0, 0.0, 0.8852727984731117, -0.8982808112686971, -0.736274797029305, -0.7938967330438224, 0.0, 0.0, 0.0, 0.0, 0.8577403913331735, -0.8450595740214994, 0.76701854766474, -0.8040421959228914, 0.7838755865169764, 0.0, 0.7808425437511121, 0.7640827108945487, 0.8811621050204153, -0.7651535198372064, 0.9536549783696272, -0.9885980421624387, 0.0, 0.0, 0.780801283522074, 0.0, 0.0, 0.8299436795840905, 0.9224841485645159, -0.8652208538569258, 0.7360514202709754, 0.7852351220018255, 0.0, -0.8227077003165834, -0.6423623153535009, 1.0685606619648502, -1.0949180606253759, -0.5731614515951251, -0.7810489337063843, -0.7812806206239239, -0.7334691297081768, 0.8011107744786727, 0.8453316718192121, -0.8829642721807789, -0.6941389939444286, -0.7906065521722841, -0.7909295246052989, -0.7934217223831472, 0.0, 0.0, 0.0, 0.9158429604503403, -0.9239778265496994, 0.0, 0.0, 0.74623763253685, 0.9509072451588003, -0.9766846979302684, -0.722847383962212, 0.0, 0.9194008659390445, -0.932997929266853, -0.7675546853441196, 0.0, 0.0, 0.8014164023971886, -0.7976613508786264, 0.0, -0.7964638802487964, 0.0, 0.0, 0.0, 0.812668287087336, -0.8135329019474535, 0.0, 0.0]

    # candidate = [2, 12, 29, 38, 61, 74, 89, 120, 134, 141, 147, 153, 169, 174, 186, 192, 197, 217, 233, 242, 247, 259, 269, 274, 281, 288, 295, 300, 309, 314, 336, 346, 351, 358, 377, 385, 397, 406, 417, 425, 433, 440, 447, 452, 464, 475, 481, 489, 500, 511, 525, 533, 543, 551, 558, 565, 574, 579, 588, 595]
    svmfeature = []#np.zeros((len(svmindex),9))
    r = 0
    # for j in candidate:
    #     print(j)
    #     n = np.linspace(j, j + 1 + len(F[j-4:j+5]), len(F[j-4:j+5]))
    #     svmfeature.append(F[j-4:j+5])
    #     plt.plot(n, svmfeature[r])
    #     plt.axhline(y=0, linestyle='-')
    #     plt.show()
    #     r += 1
    # print(svmfeature)
    # plt.figure(1)
    # plt.xlabel('Frame index')
    # plt.ylabel('DOG')
    # plt.grid(axis = 'y', ls = '--')
    # plt.plot(frameindex,DOGb)
    # plta = plt.scatter(xa, DOGa, c = 'red')
    # pltb = plt.scatter(xc, DOGc, marker = '^',c = 'green')
    # plt.legend([plta, pltb], ['Single flash frame', 'Consecutive flash frames'], loc = 'upper right')
    plt.figure(2)
    plt.xlabel('Frame index')
    plt.ylabel('Feature')
    plt.grid(axis='y', ls='--')
    plt.plot(frameindex,F)
    # plt.scatter(candidate,candidateF,c = 'red')
    pltfa = plt.scatter(xa, Fa, c = 'red')
    pltfc = plt.scatter(xc, Fc, marker = '^', c = 'green')
    plt.legend([pltfa, pltfc], ['Single flash frame', 'Consecutive flash frames'], loc = 'upper right')
    plt.axhline(y = 0.5, linestyle = '-',c = 'orange')
    plt.axhline(y = -0.5, linestyle = '-',c = 'orange')
    plt.show()
    # plt.figure(3)
    # plt.xlabel('Frame index')
    # plt.ylabel('DBF')
    # plt.grid(axis = 'y', ls = '--')
    # plt.plot(frameindex,DBFa)
    # plta = plt.scatter(xa, DBFb, c = 'red')
    # pltb = plt.scatter(xc, DBFc, marker = '^',c = 'green')
    # plt.legend([plta, pltb], ['Single flash frame', 'Consecutive flash frames'], loc = 'upper right')
    # plt.show()
    #             current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    #             previous_frame_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
    #             DBF = cv2.absdiff(current_frame_gray, previous_frame_gray)
    #             DOG = current_frame_gray - previous_frame_gray
    #             score =
    #             # d = psnr(original, contrast)
    #             e = measure.compare_ssim(previous_frame_gray, current_frame_gray)
    #             ssim.append(e)
    #             previous_frame = current_frame.copy()
    #             count += 1
    #             frameindex.append(count)
    #             if count in index:
    #                 xa.append(count)
    #                 ssima.append(e)
    #                 # print(xa)
    #                 # print(ssima)
    #     else:
    #         break
    #     plt.figure()
    #     plt.plot(frameindex,ssim)
    #     plt.scatter(xa, ssima)
    #     plt.show()
    # cap.release()
    # cv2.destroyAllWindows()



