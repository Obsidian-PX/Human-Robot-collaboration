# This example illustrates how to execute complex commands from
# a remote API client. You can also use a similar construct for
# commands that are not directly supported by the remote API.  此示例说明如何从远程API客户端执行复杂命令。对于远程API不直接支持的命令，也可以使用类似的构造。
#
# Load the demo scene 'remoteApiCommandServerExample.ttt' in CoppeliaSim, then 
# start the simulation and run this program.  在CoppeliaSim中加载演示场景“remoteApiCommandServerExample.ttt”，然后启动模拟并运行此程序。
#
# IMPORTANT: for each successful call to simxStart, there
# should be a corresponding call to simxFinish at the end!  重要提示：对于每次成功调用simxStart，末尾都应该有一个对应的simxFinish调用！

try:
    import sim
except:
    print ('--------------------------------------------------------------')
    print ('"sim.py" could not be imported. This means very probably that')
    print ('either "sim.py" or the remoteApi library could not be found.')
    print ('Make sure both are in the same folder as this file,')
    print ('or appropriately adjust the file "sim.py"')
    print ('--------------------------------------------------------------')
    print ('')
#sim.py“无法导入。这很可能意味着 （'找不到“sim.py”或remoteApi库。'）（'确保两者与此文件位于同一文件夹中，'）（'或适当调整文件“sim.py“）
import sys
import ctypes
print ('Program started')
sim.simxFinish(-1) # just in case, close all opened connections
clientID=sim.simxStart('127.0.0.1',19999,True,True,5000,5) # Connect to CoppeliaSim
if clientID!=-1:
    print ('Connected to remote API server')

    # 1. First send a command to display a specific message in a dialog box:  首先发送命令以在对话框中显示特定消息：
    emptyBuff = bytearray()
    res,retInts,retFloats,retStrings,retBuffer=sim.simxCallScriptFunction(clientID,'remoteApiCommandServer',sim.sim_scripttype_childscript,'displayText_function',[],[],['Hello world!'],emptyBuff,sim.simx_opmode_blocking)
    if res==sim.simx_return_ok:
        print ('Return string: ',retStrings[0]) # display the reply from CoppeliaSim (in this case, just a string)显示CoppeliaSim的回复（在本例中，只是一个字符串）
    else:
        print ('Remote function call failed')

    # 2. Now create a dummy object at coordinate 0.1,0.2,0.3 with name 'MyDummyName': 现在，在坐标0.1,0.2,0.3处创建一个名为“MyDummyName”的虚拟对象：
    res,retInts,retFloats,retStrings,retBuffer=sim.simxCallScriptFunction(clientID,'remoteApiCommandServer',sim.sim_scripttype_childscript,'createDummy_function',[],[0.1,0.2,0.3],['MyDummyName'],emptyBuff,sim.simx_opmode_blocking)
    if res==sim.simx_return_ok:
        print ('Dummy handle: ',retInts[0]) # display the reply from CoppeliaSim (in this case, the handle of the created dummy)显示来自CoppeliaSim的回复（在本例中，是创建的虚拟对象的句柄）
    else:
        print ('Remote function call failed')

    # 3. Now send a code string to execute some random functions: 现在发送一个代码字符串来执行一些随机函数：
    code="local octreeHandle=simCreateOctree(0.5,0,1)\n" \
    "simInsertVoxelsIntoOctree(octreeHandle,0,{0.1,0.1,0.1},{255,0,255})\n" \
    "return 'done'"
    res,retInts,retFloats,retStrings,retBuffer=sim.simxCallScriptFunction(clientID,"remoteApiCommandServer",sim.sim_scripttype_childscript,'executeCode_function',[],[],[code],emptyBuff,sim.simx_opmode_blocking)
    if res==sim.simx_return_ok:
        print ('Code execution returned: ',retStrings[0])
    else:
        print ('Remote function call failed')

    # Now close the connection to CoppeliaSim: 现在关闭与CoppeliaSim的连接：
    sim.simxFinish(clientID)
else:
    print ('Failed connecting to remote API server')
print ('Program ended')
