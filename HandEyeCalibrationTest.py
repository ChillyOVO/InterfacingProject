import cv2
import numpy as np
import PDDTVisionToolBox as pd

# 手眼标定测试
# 根据TOB = THB @ TCH @ TOC, 即是物体在基坐标系下的位姿 = 手在基坐标系下的位姿 * 相机在末端坐标系下的位姿 * 物体在相机中的位姿
# 构造多组 THB @ TCH @ TOC, 其中THB由正向运动学计算,TOC由标定板真是坐标系进行solvepnp反算得到,因此需要明确尺度单位进行统一

# TOC求解 方法1：solvepnp反向求解
# 确定相机内参
Intrinsic = np.array([[4295.82123518699, 0, 1947.32510469332],
                      [0, 4297.26536461942, 1050.71392654110],
                      [0, 0, 1]])
Distortion = np.array([-0.109780715530354, 0.133569098996419, 0, 0])
# 构建目标实际坐标系
# solve pnp
# 分解R/T

# TOC求解 方法2：利用标定函数,直接计算旋转矢量,平移适量
# 旋转矢量
RotVec = np.array([[0.257474791207814, -0.0155190998515539, -0.0580872749056459],
                   [0.257118789795498, -0.0157061498795581, -0.0583222386633372],
                   [0.257103287406290, -0.0152289599063214, -0.0583995687201808],
                   [0.257117810315047, -0.0154471206440839, -0.0580295079121719],
                   [0.256855864817970, -0.0154090529241123, -0.0583138503271804],
                   [0.256779066762576, -0.0155616588212847, -0.0583799292787064],
                   [0.256879677446261, -0.0152888010230252, -0.0581932221988976],
                   [0.256920921625924, -0.0157336236870655, -0.0583520257817011],
                   [0.256996033331680, -0.0156661232088012, -0.0583946356656039],
                   [0.316052302471475, 0.0475295905590182, -0.0671687043657850],
                   [0.198079564993741, -0.0784174394020308, -0.0495956444326105],
                   [0.320381633673388, -0.0744680672786082, -0.0501949474871335],
                   [0.193984979806343, 0.0424982262302839, -0.0662628406332123],
                   [0.257843645142121, -0.0158754618579552, -0.0579786060798066],
                   [0.257189167986448, -0.0161631912538368, -0.0582449835256856],
                   [0.257132808648898, -0.0156865900153470, -0.0583389908899666],
                   [0.257765495290678, -0.0155033429478493, -0.0578548848723808],
                   [0.257645697787194, -0.0155165938382747, -0.0580713126801928],
                   [0.257299790059185, -0.0155297648387842, -0.0581617762645225],
                   [0.257392719726007, -0.0158330865793150, -0.0578715232428141],
                   [0.257296908474909, -0.0157777648637893, -0.0581118251123337],
                   [0.257301184630482, -0.0157479220550327, -0.0581244342914568],
                   [0.316495454572797, 0.0476142616448935, -0.0667751814343254],
                   [0.198126447593838, -0.0784161197080974, -0.0493203541760869],
                   [0.320758231792013, -0.0745301223706457, -0.0500028495566539],
                   [0.194208205616671, 0.0426099623072870, -0.0660545031692118],
                   [0.257748815952828, -0.0156430370417015, -0.0580556540480855],
                   [0.257811276233391, -0.0154940493367910, -0.0583150768193594],
                   [0.256896996377246, -0.0157287114222502, -0.0584226467305629],
                   [0.257856898336148, -0.0155764266543757, -0.0579666954059502],
                   [0.257701928712380, -0.0157121278661607, -0.0580950671228085],
                   [0.257094475870071, -0.0157673755595358, -0.0583245357155497],
                   [0.257831393021855, -0.0155198477882311, -0.0579367581950719],
                   [0.257662690432388, -0.0157653468177120, -0.0580968177158021],
                   [0.257426958846401, -0.0156585720817072, -0.0581476434152160],
                   [0.316654608403850, 0.0475238381327264, -0.0668489760714177],
                   [0.198224973169626, -0.0785135305115029, -0.0494012757777630],
                   [0.320948768620887, -0.0744231841285401, -0.0500186647292842],
                   [0.194333762322521, 0.0426830726727041, -0.0661782129239805]])
# 平移向量
TransVec = np.array([[-42.0388975942281, -29.1282996401180, 270.676774024013],
                     [-27.0847755210788, -29.0233924284279, 270.857129044421],
                     [-12.0926787369000, -28.9401546752487, 270.973438539908],
                     [-42.0429109960685, -14.0705047749475, 270.727232692080],
                     [-27.0928908252659, -14.0022711858371, 270.811786228461],
                     [-12.0801196872657, -13.9195864990493, 270.899948499241],
                     [-42.1982658704413, 0.947637610183456, 270.563881921151],
                     [-27.2127322334896, 1.03532796060365, 270.697255401257],
                     [-12.1826220739898, 1.05389770632844, 270.783997635847],
                     [11.0185119710987, -22.0085357896941, 266.880380338795],
                     [-35.4953052765226, 24.1507408556957, 271.788999638854],
                     [-34.8307687769498, -22.0529616965917, 260.959996826867],
                     [11.1212353931002, 24.5755438674065, 277.659895085417],
                     [-42.0766424618718, -29.0701109087736, 280.588050221224],
                     [-27.1106926885857, -29.0067141800861, 280.736026522628],
                     [-12.1472063368512, -28.9294233424782, 280.857058495947],
                     [-42.0493696510356, -14.1236904070589, 280.546742467831],
                     [-27.0808811174178, -14.0442366850251, 280.634518413255],
                     [-12.0778877116896, -13.9589224977768, 280.750298717321],
                     [-42.2159909219614, 0.962539396860107, 280.439043273176],
                     [-27.2532360681132, 1.04057652873591, 280.594564294699],
                     [-12.2147952019718, 1.08984068762583, 280.656954649968],
                     [11.6373441540234, -22.5885791732075, 276.628526507196],
                     [-36.1086061730672, 24.7727552009467, 281.645141978700],
                     [-35.4330850321671, -22.6539082105477, 270.732105377281],
                     [11.7006187923657, 25.2502146492247, 287.533380623409],
                     [-42.1170538800584, -29.0424742795311, 290.561419684038],
                     [-27.1601250801592, -28.9755148069163, 290.657409069993],
                     [-12.1889306961722, -28.8862275727477, 290.839533860735],
                     [-42.1341888782232, -14.0289565924610, 290.513984862924],
                     [-27.1400151680421, -13.9746318442873, 290.600105398847],
                     [-12.1815558583738, -13.8655736416367, 290.732298188765],
                     [-42.2381664380841, 0.895765830235100, 290.380108134961],
                     [-27.2577942012604, 0.968653221080034, 290.523310214891],
                     [-12.2588128642948, 1.04053273173669, 290.630777952975],
                     [12.1581897157603, -23.1790280795714, 286.569275818191],
                     [-36.8240775035031, 25.4074429376005, 291.606843343907],
                     [-36.1646519422796, -23.2885988957842, 280.630768291410],
                     [12.2295661045684, 25.8666862417433, 297.502611194278]])
# 取旋转矩阵行数作为循环操作总数
row, col = RotVec.shape
# 循环获得旋转矩阵元胞与平移矩阵元胞
RotMatObjToCam = []
TransMatObjToCam = []
for i in range(row):
    # 旋转矩阵转换
    RotMatTemp = pd.getRotVec2RotMAT(RotVec[i, :])
    RotMatObjToCam.append(RotMatTemp)
    # 平移向量转换
    TransMatObjToCam.append(TransVec[i, :].reshape((-1, 1)))
# TOC变形形成增广矩阵
RTObjToCam = []
for i in range(row):
    RTTemp = np.column_stack((RotMatObjToCam[i], TransMatObjToCam[i]))
    RTTemp = np.row_stack((RTTemp, np.array([0, 0, 0, 1])))
    RTObjToCam.append(RTTemp)
# print(RTObjToCam)

# THB求解
# 机械臂记录坐标值
PoseData = np.array(
    [[3.88824793e-01, 8.31221434e-02, 7.10427891e-01, -2.90211318e+00, -1.20056335e+00, 8.76360570e-05],
     [3.88787833e-01, 9.81021822e-02, 7.10502525e-01, -2.90234032e+00, -1.20021632e+00, -8.2286500e-05],
     [3.88810889e-01, 1.13095034e-01, 7.10491798e-01, -2.90220613e+00, -1.20027328e+00, -9.3155891e-05],
     [4.03747826e-01, 8.31560534e-02, 7.10502217e-01, -2.90217832e+00, -1.20049942e+00, -2.4177500e-04],
     [4.03723419e-01, 9.81357134e-02, 7.10530169e-01, -2.90225249e+00, -1.20028792e+00, -2.5119999e-04],
     [4.03708031e-01, 1.13152023e-01, 7.10549600e-01, -2.90223849e+00, -1.20030695e+00, -3.4016709e-04],
     [4.18762419e-01, 8.31560043e-02, 7.10458931e-01, -2.90218877e+00, -1.20050774e+00, -6.8323336e-05],
     [4.18721905e-01, 9.81347749e-02, 7.10544739e-01, -2.90236430e+00, -1.20021024e+00, -2.9073751e-04],
     [4.18703610e-01, 1.13139619e-01, 7.10547007e-01, -2.90228323e+00, -1.20033882e+00, -2.9363187e-04],
     [0.41865656, 0.11313568, 0.71055114, -2.82737795, -1.16924925, 0.0507615],
     [0.41877404, 0.11310379, 0.71048455, 2.82910752, 1.16986536, 0.05125059],
     [0.41868233, 0.11313826, 0.71050552, 2.87084527, 1.18730681, -0.12516696],
     [0.41873715, 0.11312944, 0.71049653, -2.86903578, -1.18664297, -0.12538876],
     [3.88808939e-01, 8.31567847e-02, 7.20400555e-01, -2.90215213e+00, -1.20058405e+00, 2.5433736e-04],
     [3.88806860e-01, 9.81453746e-02, 7.20395257e-01, -2.90230422e+00, -1.20028615e+00, 2.5048486e-04],
     [3.88785220e-01, 1.13136201e-01, 7.20426524e-01, -2.90228017e+00, -1.20036700e+00, 1.1799263e-04],
     [4.03764553e-01, 8.32207891e-02, 7.20404511e-01, -2.90191152e+00, -1.20083751e+00, 2.2888723e-04],
     [4.03769801e-01, 9.81814422e-02, 7.20405540e-01, -2.90198446e+00, -1.20062409e+00, 2.5961931e-04],
     [4.03759937e-01, 1.13189076e-01, 7.20429045e-01, -2.90203461e+00, -1.20050652e+00, 1.9317729e-04],
     [4.18771749e-01, 8.31929757e-02, 7.20370277e-01, -2.90207490e+00, -1.20082531e+00, 2.6999381e-04],
     [4.18739039e-01, 9.81405902e-02, 7.20440734e-01, -2.90231029e+00, -1.20047038e+00, 1.6114720e-04],
     [4.18745303e-01, 1.13160119e-01, 7.20414859e-01, -2.90218621e+00, -1.20060116e+00, 1.7602455e-04],
     [0.41872553, 0.11320052, 0.7204343, -2.82716543, -1.16975628, 0.05115379],
     [0.41876191, 0.11316653, 0.72039417, 2.82894129, 1.17022677, 0.05095686],
     [0.41870431, 0.11318417, 0.72040248, 2.87070607, 1.18756638, -0.12547735],
     [0.41874747, 0.1131986, 0.72040693, -2.86892219, -1.18708335, -0.12520315],
     [3.88822625e-01, 8.31533235e-02, 7.30343241e-01, -2.90205684e+00, -1.20057010e+00, 4.30407147e-04],
     [3.88813885e-01, 9.81297964e-02, 7.30346878e-01, -2.90221075e+00, -1.20021130e+00, 4.96670779e-04],
     [3.88800071e-01, 1.13125650e-01, 7.30376184e-01, -2.90222655e+00, -1.20026784e+00, 3.89063981e-04],
     [4.03760962e-01, 8.31803815e-02, 7.30361246e-01, -2.90196534e+00, -1.20071994e+00, 3.51320390e-04],
     [4.03741385e-01, 9.81806756e-02, 7.30433779e-01, -2.90204533e+00, -1.20057066e+00, 2.08390517e-04],
     [4.03732958e-01, 1.13158354e-01, 7.30425501e-01, -2.90214436e+00, -1.20033043e+00, 1.41976332e-04],
     [4.18725794e-01, 8.31927621e-02, 7.30393917e-01, -2.90197935e+00, -1.20084656e+00, 3.92678266e-04],
     [4.18694097e-01, 9.81688535e-02, 7.30428264e-01, -2.90214067e+00, -1.20054151e+00, 3.48390675e-04],
     [4.18697308e-01, 1.13157418e-01, 7.30428863e-01, -2.90207056e+00, -1.20063302e+00, 2.90047122e-04],
     [0.41868467, 0.11314758, 0.7304499, -2.82718039, -1.16964457, 0.05137802],
     [0.418711, 0.11313188, 0.73044755, 2.82894346, 1.17018143, 0.0508217],
     [0.41867512, 0.11313589, 0.73046391, 2.87061487, 1.18753855, -0.12563076],
     [0.41872196, 0.113148, 0.73045789, -2.86898264, -1.18697689, -0.12505635]])

# 转换成增广矩阵
row, col = PoseData.shape
RotMatHandToBase = []
TransMatHandToBase = []
RTHandToBase = []
for i in range(row):
    # 旋转
    RotVec = np.array([PoseData[i, 3], PoseData[i, 4], PoseData[i, 5]])
    RotMatTemp = pd.getRotVec2RotMAT(RotVec)
    RotMatHandToBase.append(RotMatTemp)
    # 平移
    TransVec = np.array([[PoseData[i, 0]], [PoseData[i, 1]], [PoseData[i, 2]]])
    TransVec = 1000 * TransVec
    TransMatHandToBase.append(TransVec)
    # 增广
    RTTemp = np.column_stack((RotMatTemp, TransVec))
    RTTemp = np.row_stack((RTTemp, np.array([0, 0, 0, 1])))
    RTHandToBase.append(RTTemp)

# TCH求解
# 利用Opencv 手眼求解
RotMatCamToHand, TransMatCamToHand = cv2.calibrateHandEye(RotMatHandToBase, TransMatHandToBase, RotMatObjToCam,
                                                          TransMatObjToCam)
RTCamToHand = np.column_stack((RotMatCamToHand, TransMatCamToHand))
RTCamToHand = np.row_stack((RTCamToHand, np.array([0, 0, 0, 1])))

# TOB验证
# 通过TOB的变化量验证TCH的正确性
Rx = np.zeros((row, 1))
Ry = np.zeros((row, 1))
Rz = np.zeros((row, 1))
X = np.zeros((row, 1))
Y = np.zeros((row, 1))
Z = np.zeros((row, 1))
for i in range(row):
    RTObjToBase = RTHandToBase[i] @ RTCamToHand @ RTObjToCam[i]
    # 累计旋转角度
    RotObjToBase = RTObjToBase[0:3, 0:3]
    Rx[i], Ry[i], Rz[i] = pd.getRotationMatrixToAngles(RotObjToBase)
    # 位移
    X[i] = RTObjToBase[0, 3]
    Y[i] = RTObjToBase[1, 3]
    Z[i] = RTObjToBase[2, 3]
    # print(RTObjToBase)

print('Rx重复误差：', np.std(Rx, ddof=1), ' rad')
print('Ry重复误差：', np.std(Ry, ddof=1), ' rad')
print('Rz重复误差：', np.std(Rz, ddof=1), ' rad')
print('X重复误差：', np.std(X, ddof=1), ' mm')
print('Y重复误差：', np.std(Y, ddof=1), ' mm')
print('Z重复误差：', np.std(Z, ddof=1), ' mm')

print(RTCamToHand)

# [[-7.03041567e-01 -7.11139516e-01  3.62544955e-03  8.90227245e+01]
#  [ 7.11132519e-01 -7.02981928e-01  1.03416157e-02  3.92668263e+01]
#  [-4.80570609e-03  9.84876080e-03  9.99939952e-01  1.06567273e+02]
#  [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00]]
