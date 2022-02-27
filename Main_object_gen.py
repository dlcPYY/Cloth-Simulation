# Ref: httpq
import datetime
import sys
import taichi as ti
import trimesh
import numpy
import _thread
import time
import vtk
import plyfile
from plyfile import PlyData

ti.init(arch=ti.cuda)

EPS = 1e-10

Mu = 0.2


# Air resistance
A: ti.f64
A = 0  # 0.0001
# 密度， 单位kg/m2
# Density= 0.276

Density: ti.f64
Density = (0.276)
# 弹性系数，单位：N/m
massK: ti.f64
massK = (0.0001750)
# 布料边长 单位mm
length: ti.f64
length = (38)
# 布料初始高度
initialHeight: ti.f64
initialHeight = (15)
# 布料分辨率
widthSize, heightSize = 100, 100
# 三根弹簧,单位：mm
massLengthA = length/(widthSize+1)
massLengthB = ti.sqrt(2 * massLengthA * massLengthA)
massLengthC = massLengthA*2


starttime = datetime.datetime.now()
endtime = datetime.datetime.now()

pointMass = Density*(length/1e3)*(length/1e3)/(widthSize*heightSize)
print(pointMass)


#######################################################################################

faceSize = widthSize * heightSize * 2

pointSize = (widthSize + 1) * (heightSize + 1)

oriPointLocation = ti.Vector.field(3, dtype=ti.f64, shape=pointSize)
pointLocation = ti.Vector.field(3, dtype=ti.f64, shape=pointSize)
lastpointLocation = ti.Vector.field(3, dtype=ti.f64, shape=pointSize)
pointVelocity = ti.Vector.field(3, dtype=ti.f64, shape=pointSize)
pointForce = ti.Vector.field(3, dtype=ti.f64, shape=pointSize)
temppointForce = ti.Vector.field(3, dtype=ti.f64, shape=pointSize)
temppointLocation = ti.Vector.field(3, dtype=ti.f64, shape=pointSize)
temppointVelocity = ti.Vector.field(3, dtype=ti.f64, shape=pointSize)


lossLunction = ti.field(dtype=ti.f64, shape=pointSize)

Idx = ti.field(dtype=ti.i32, shape=faceSize * 3)


# Y Forward
G = ti.Vector([0.0, 0.0, -9.8], dt=ti.f64)

Wind = ti.Vector([0.000000, 0.0, 0.0], dt=ti.f64)

sphereCenter = ti.Vector([6.0, 6.0, 4.0], dt=ti.f64)
sphereRadious = 3.0

# 整体的圆心
cylinderCenter = ti.Vector([length/2, length/2, 15.0], dt=ti.f64)
# 圆台上半径9.8
circularTR = ti.f64
circularTR = 3.0

# 圆台下半径
circularBR = ti.f64
circularBR = 6.0

# 圆台高度
circularHeight = ti.f64
circularHeight = 4.0

# 圆台上底部
circularTH = ti.f64
circularTH = 15.0


# y圆柱的圆心
cylinderCenter2 = ti.Vector([length/2, length/2, 11.0], dt=ti.f64)
# 圆台上半径9.8
circularTR2 = ti.f64
circularTR2 = 0.5
# 圆台高度
circularHeight2 = ti.f64
circularHeight2 = 9.0

# 圆台上底部
circularTH2 = ti.f64
circularTH2 = 11.0

# y圆柱的圆心
cylinderCenter3 = ti.Vector([length/2, length/2, 2.0], dt=ti.f64)
# 圆台上半径9.8
circularTR3 = ti.f64
circularTR3 = 8.0
# 圆台高度
circularHeight3 = ti.f64
circularHeight3 = 1.0

# 圆台上底部
circularTH3 = ti.f64
circularTH3 = 2.0



cylinderRadious = 1.5
cylinderHight = 1.0
cylinderNormal = ti.Vector([0.0, 0.0, -1.0], dt=ti.f64)

 
@ti.func
def pointID(x, y):
    R = -1
    if 0 <= x and x <= widthSize and 0 <= y and y <= heightSize:
        R = y * (widthSize + 1) + x
    return R


def pointIDPy(x, y):
    R = -1
    if 0 <= x and x <= widthSize and 0 <= y and y <= heightSize:
        R = y * (widthSize + 1) + x
    return R


@ti.func
def pointCoord(ID):
    return (ID % (widthSize + 1), ID // (widthSize + 1))


@ti.func
def massID(ID):
    R = ti.Vector([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], dt=ti.i32)
    x, y = pointCoord(ID)
    R[0], R[1] = pointID(x-1, y), pointID(x+1, y)
    R[2], R[3] = pointID(x, y-1), pointID(x, y+1)
    R[4], R[5] = pointID(x-1, y-1), pointID(x+1, y+1)
    R[6], R[7] = pointID(x-1, y+1), pointID(x+1, y-1)
    R[8], R[9] = pointID(x, y-2), pointID(x, y+2)
    R[10], R[11] = pointID(x-2, y), pointID(x+2, y)
    return R


@ti.kernel
def InitTi():
    for i in range(pointSize):
        x, y = pointCoord(i)
        pointLocation[i] = (x * massLengthA, y * massLengthA, initialHeight)
        oriPointLocation[i] = (x * massLengthA, y * massLengthA, initialHeight)
        pointVelocity[i] = (0, 0, 0)


@ti.kernel
def Smooth():
    for i in pointForce:
        temppointForce[i] = pointForce[i]
        temppointLocation[i] = pointLocation[i]
        temppointVelocity[i] = pointVelocity[i]
        effectiveNum = 1
        Dirs = massID(i)
        # 结构弹簧
        for j in ti.static(range(0, 8)):
            if not Dirs[j] == -1:
                effectiveNum += 1
                temppointForce[i] += pointForce[Dirs[j]]
                temppointLocation[i] += pointLocation[Dirs[j]]
                temppointVelocity[i] += pointVelocity[Dirs[j]]
        temppointForce[i] /= effectiveNum
        temppointLocation[i] /= effectiveNum
        temppointVelocity[i] /= effectiveNum
    for i in pointForce:
        Dirs = massID(i)
        judge = True
        for j in ti.static(range(0, 8)):
            if not Dirs[j] == -1:
                judge = False
        if judge:
            pointLocation[i] = temppointLocation[i]
        # pointForce[i]=temppointForce[i]
        # pointVelocity[i]=temppointVelocity[i]


@ti.kernel
def ComputeForce():
    for i in pointForce:
        pointForce[i] = (0, 0, 0)
        Dirs = massID(i)
        # 结构弹簧
        for j in ti.static(range(0, 4)):
            if not Dirs[j] == -1:
                Dir = pointLocation[Dirs[j]] - pointLocation[i]
                pointForce[i] += (Dir.norm() - massLengthA) * \
                    massK * Dir / Dir.norm()
        # 扭曲弹簧
        for j in ti.static(range(4, 8)):
            if not Dirs[j] == -1:
                Dir = pointLocation[Dirs[j]] - pointLocation[i]
                pointForce[i] += (Dir.norm() - massLengthB) * \
                    massK * Dir / Dir.norm()
        # 拉伸弹簧
        for j in ti.static(range(8, 12)):
            if not Dirs[j] <= -1:
                Dir = pointLocation[Dirs[j]] - pointLocation[i]
                pointForce[i] += (Dir.norm() - massLengthC) * \
                    massK * Dir / Dir.norm()
        # 重力+风
        pointForce[i] += G * pointMass + Wind
        # 模型吸引力
        # if pointLocation[i][2]<circularTH2 and pointLocation[i][2]>circularTH2-circularHeight2:
        #     Dir = ti.Vector([pointLocation[i][0]-cylinderCenter[0], pointLocation[i][1]-cylinderCenter[1], 0.0], dt=ti.f64)
        #     nD = Dir/Dir.norm()
        #     pointForce[i]-=(nD*0.00003)
        # 空气阻力
        # pointVelocity[i] -= A * pointVelocity[i]


@ti.kernel
def Forward(T: ti.f64):
    for i in range(pointSize):
        pointVelocity[i] += T * pointForce[i] / pointMass
        pointLocation[i] = pointLocation[i]+pointVelocity[i]*T
        # if pointLocation[i][2]<0:
        #   pointLocation[i][2]=0
        #   pointVelocity[i]=(0,0,0)


@ti.kernel
def RecordLocation():
    for i in pointLocation:
        lastpointLocation[i] = pointLocation[i]


@ti.kernel
def ComputeCollsion():
    for i in pointLocation:
        # pointLocation[i][2] = max(0, pointLocation[i][2])
        # 最上方圆台
        if pointLocation[i][2] < circularTH and pointLocation[i][2] > circularTH-circularHeight:
            Dir = pointLocation[i] - sphereCenter
            nowRadious = circularTR + (circularBR-circularTR)*(circularTH - pointLocation[i][2])/(circularHeight)
            Dir2 = pointLocation[i] - cylinderCenter
            fi = (Dir2.norm()**2-Dir2[2]**2)**0.5-nowRadious
            if fi < 0 :
                pointVelocity[i] = (0, 0, 0)
        # 侧面
                if fi > -1.0*1e-4:
                    tempcenter = ti.Vector(
                        [cylinderCenter[0], cylinderCenter[1], pointLocation[i][2]], dt=ti.f64)
                    Dir3 = pointLocation[i] - tempcenter
                    nD = Dir3/Dir3.norm()
                    pointLocation[i] = tempcenter+(nowRadious+1e-4)*nD
                # 顶面
                elif circularTH-pointLocation[i][2] < 1e-4:
                    pointLocation[i][2] = circularTH+1e-4
                # 底面
                elif pointLocation[i][2] > circularTH-circularHeight and pointLocation[i][2]-(circularTH-circularHeight) < 1e-4:
                    pointLocation[i][2] = circularTH-circularHeight-1e-4
        # 中间圆柱
        if pointLocation[i][2] < circularTH2 and pointLocation[i][2] > circularTH2-circularHeight2:
            Dir = pointLocation[i] - sphereCenter
            Dir2 = pointLocation[i] - cylinderCenter2
            fi = (Dir2.norm()**2-Dir2[2]**2)**0.5-circularTR2
            if fi < 0 :
                pointVelocity[i] = (0, 0, 0)
        # 侧面
                if fi > -1.0*1e-4:
                    tempcenter = ti.Vector(
                        [cylinderCenter2[0], cylinderCenter2[1], pointLocation[i][2]], dt=ti.f64)
                    Dir3 = pointLocation[i] - tempcenter
                    nD = Dir3/Dir3.norm()
                    pointLocation[i] = tempcenter+(circularTR2+1e-4)*nD
                # 顶面
                elif circularTH2-pointLocation[i][2] < 1e-4:
                    pointLocation[i][2] = circularTH2+1e-4
                # 底面
                elif pointLocation[i][2] > circularTH2-circularHeight2 and pointLocation[i][2]-(circularTH2-circularHeight2) < 1e-4:
                    pointLocation[i][2] = circularTH2-circularHeight2-1e-4
        # 底部圆柱
        if pointLocation[i][2] < circularTH3 and pointLocation[i][2] > circularTH3-circularHeight3:
            Dir2 = pointLocation[i] - cylinderCenter3
            fi = (Dir2.norm()**2-Dir2[2]**2)**0.5-circularTR3
            if fi < 0 :
                pointVelocity[i] = (0, 0, 0)
        # 侧面
                if fi > -1.0*1e-4:
                    tempcenter = ti.Vector(
                        [cylinderCenter3[0], cylinderCenter3[1], pointLocation[i][2]], dt=ti.f64)
                    Dir3 = pointLocation[i] - tempcenter
                    nD = Dir3/Dir3.norm()
                    pointLocation[i] = tempcenter+(circularTR3+1e-4)*nD
                # 顶面
                elif circularTH3-pointLocation[i][2] < 1e-4:
                    pointLocation[i][2] = circularTH3+1e-4
                # 底面
                elif pointLocation[i][2] > circularTH3-circularHeight3 and pointLocation[i][2]-(circularTH3-circularHeight3) < 1e-4:
                    pointLocation[i][2] = circularTH3-circularHeight3-1e-4
        # Dir2 = pointLocation[i] - cylinderCenter
        # fi=(Dir2.norm()**2-Dir2[2]**2)**0.5-cylinderRadious
        # if fi<0 and pointLocation[i][2]>cylinderCenter[2]-cylinderHight and pointLocation[i][2]<cylinderCenter[2]:
        #   pointVelocity[i] = (0, 0, 0)
        # # 侧面
        #   if fi>-1.0*1e-4:
        #     tempcenter=ti.Vector([cylinderCenter[0], cylinderCenter[1], pointLocation[i][2]], dt=ti.f64)
        #     Dir3=pointLocation[i] - tempcenter
        #     nD=Dir3/Dir3.norm()
        #     pointLocation[i]=tempcenter+(cylinderRadious+1e-4)*nD
        #   elif cylinderCenter[2]-pointLocation[i][2]<1e-4:
        #     pointLocation[i][2]=cylinderCenter[2]+1e-4
        #   elif pointLocation[i][2]>cylinderCenter[2]-cylinderHight and pointLocation[i][2]+cylinderHight-cylinderCenter[2]<1e-4:
        #     pointLocation[i][2]=cylinderCenter[2]-cylinderHight-1e-4


@ti.kernel
def SelfCollsion():
    tips = True
    if tips:
        for i in range(pointSize):
            for j in range(i+1, pointSize):
                if (i != widthSize and j == i+1) or j == i+widthSize+1 or j == i+widthSize+2:
                    continue
                # print("ifLink")
                Dir = pointLocation[j] - pointLocation[i]
                if Dir.norm() < massLengthA:
                    pointLocation[j] = lastpointLocation[j]
                    pointLocation[i] = lastpointLocation[i]
                    pointVelocity[i] = (0, 0, 0)
                    pointVelocity[j] = (0, 0, 0)
                    # print(pointLocation[j],pointLocation[i],i,j,Dir.norm(),massLengthA)
                    # print(i,j)

def read_ply(filename):
    plydata = PlyData.read(filename)
    pc = plydata['vertex'].data
    pc_array = numpy.array([[x, y, z] for x,y,z in pc])
    return pc_array

def InitTi_2(ply_array):
    for i in range(pointSize):
        pointLocation[i] = (ply_array[i][0],ply_array[i][1],ply_array[i][2])
        oriPointLocation[i] = pointLocation[i]
        pointVelocity[i] = (0, 0, 0)

def Init():
    InitTi()
    Index = 0
    for i in range(widthSize):
        for j in range(heightSize):
            ID_1 = pointIDPy(i, j)
            ID_2 = pointIDPy(i+1, j)
            ID_3 = pointIDPy(i, j+1)
            ID_4 = pointIDPy(i+1, j+1)

            Idx[Index + 0] = ID_1
            Idx[Index + 1] = ID_2
            Idx[Index + 2] = ID_3
            Idx[Index + 3] = ID_2
            Idx[Index + 4] = ID_4
            Idx[Index + 5] = ID_3
            Index += 6


def Step():

    for i in range(150):
        RecordLocation()
        ComputeForce()
        Forward(2e-6)
        ComputeCollsion()
        Smooth()
        # SelfCollsion()
    # print("testA")

    # Smooth()
    # Smooth()


def Export(i: int):
    npL = pointLocation.to_numpy()
    npI = Idx.to_numpy()

    mesh_writer = ti.PLYWriter(
        num_vertices=pointSize, num_faces=faceSize, face_type="tri")

    mesh_writer.add_vertex_pos(npL[:, 0], npL[:, 1], npL[:, 2])
    mesh_writer.add_faces(npI)
    # mesh_writer.show()
    mesh_writer.export_frame_ascii(i, 'S.ply')
    if i == 50000:
        f = open('log.txt', 'w')
        for i in range(5000):
            f.write(str(lossLunction[i])+'\n')
        sys.exit()
    print('Frame >> %03d' % (i))


def ReadinMesh():
    PlyPath = "D:/library/model/circle_vase.stl"
    mesh = trimesh.load(PlyPath)
    nv = mesh.vertices
    nf = mesh.faces
    v = numpy.array(nv)
    f = numpy.array(nf)

    return (v, f)


def Calculate(name, tempPointLocation):
    for i in range(pointSize):
        Dir = tempPointLocation[i] - oriPointLocation[i]
        lossLunction[name] += Dir.norm()*Dir.norm()
    print(name)


def main():
    # gui =ti.GUI('ClothSimulation System',background_color=0xDDDDDD)

    Init()

    #v,f = ReadinMesh()
    Frame = 0
    try:
        while True:
            # gui.show()
            Step()
            Frame += 1

            if not Frame % 100:
                # tempPointLocation=pointLocation
                # try:
                #   _thread.start_new_thread( Calculate, (Frame // 100, tempPointLocation, ) )
                # except:
                #   print ("Error: 无法启动线程")

                #   tempid=Frame // 100
                #   lossLunction[tempid]+=Dir.norm()*Dir.norm()
                # for i in range(pointSize):
                #   pointLocation[i]*=10
                # for i in range(pointSize):
                #   Dir = pointLocation[i] - oriPointLocation[i]
                #   tempid=Frame // 100
                #   lossLunction[tempid]+=Dir.norm()*Dir.norm()
                Export(Frame // 100)
                # print(pointForce[0])
                # print(pointVelocity[0])
                # pointLocation=tempPointLocation
    except Exception as Error:
        print(Error)


if __name__ == '__main__':
    main()
