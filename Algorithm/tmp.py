import time
import random
import threading
####################################################################################################################################
def increaseY():
    global Y_t,fps_
    Y_t+=fps_             #virtual queue fps:30
    threading.Timer(1, increaseY).start()

def inferenceFunction(latency):
    time.sleep(latency)
    return 1

def resourceIdxChange(idx):
    global resorce_list
    max_idx=len(resorce_list)-1
    increment=random.randint(-15,15)
    if(0<=(idx+increment)<=max_idx):
        idx+=increment
    return idx

def makeDPP(exit,Y_t,p_th):
    global mAP_list
    DPP=(2*Y_t*(p_th-mAP_list[exit]))+((p_th-mAP_list[exit])**2)
    return DPP

def makeLatency(exit,current_resource):
    global computation_exit
    latency=computation_exit[exit]/current_resource
    return latency

####################################################################################################################################
#constant variables 
infinity=2**32
total_exit=4            #exit number is 4
####################################################################################################################################
#hyperparameters
V=10                    # Latency weight constants
p_th=75                 # random value
image_num=5000          # image number
fps_=30                 # fps:30
min_resource=1000000    # min:1Mhz
max_resource=4000000000 # max:4Ghz
resorce_step=1000000    # step:1Mhz
resorce_list=[i for i in range(min_resource,max_resource+1,resorce_step)] #length: 4000000
resource_idx=(((max_resource-min_resource)//resorce_step)+1)//2 # initial value of resource
####################################################################################################################################
#layer class
layer_calculation={     # each layer's calculation value
    'init_layer':1,
    'layer1':7,
    'layer2':10,
    'layer3':50,
    'layer4':16,
    'RPN':4,
    'Detector':16,
    'EE':3}
computation_exit=[0]*total_exit 
computation_exit[0]=(layer_calculation['init_layer']+layer_calculation['EE']+layer_calculation['RPN']+layer_calculation['Detector'])
computation_exit[1]=(computation_exit[0]+layer_calculation['layer1']+layer_calculation['EE']+layer_calculation['RPN']+layer_calculation['Detector'])
computation_exit[2]=(computation_exit[1]+layer_calculation['layer2']+layer_calculation['EE']+layer_calculation['RPN']+layer_calculation['Detector'])
computation_exit[3]=(computation_exit[2]+layer_calculation['layer3']+layer_calculation['RPN']+layer_calculation['Detector'])
#computation_exit=[24, 54, 87, 157]
####################################################################################################################################
#expectation accuracy
mAP_list=[40,60,80,90]  # 4-exit exist and each has expectation mAP
####################################################################################################################################
#variables initialize
target_exit=3           # initializing target exit as last exit
Y_t=0                   # initialize virtual queue
objective=0             # objective function value
####################################################################################################################################
#main function
def main():
    increaseY()             # virtual queue fps:30 !!!!!!!!!!!!!!!!!!START!!!!!!!!!!!!!!
    for image in range(image_num): #image만큼 iteration process
        current_resource=resorce_list[resourceIdxChange(resource_idx)] #dynamic resource state
        #calculate objective function and get target exit number
        objective=infinity
        target_exit=3       #initializing target_exit as final exit
        for exit in range(total_exit):
            latency=computation_exit[exit]/current_resource
            DPP=(2*Y_t*(p_th-mAP_list[exit]))+((p_th-mAP_list[exit])**2)
            if(DPP + (V*latency)<objective):
                objective = DPP + (V*latency)
                target_exit=exit

        # inference step
        inferenceFunction(latency)
        # after inference 
        inference_result_mAP=random.randint(mAP_list[target_exit]-10,mAP_list[target_exit]+10)
        Y_t=max(Y_t+p_th-inference_result_mAP,0)
        print(f'{image}번째 이미지의 exit: {target_exit}, resource값: {current_resource}')
####################################################################################################################################
#compare DPP and Latency
def compareDPPnLatency():
    for p_th in range(30,101):
        for Y_t in range(30,101):
            for exit in range(4):
                DPP=makeDPP(exit,Y_t,p_th)
                
    for current_resource in range(min_resource,max_resource+1,resorce_step):
        for exit in range(4):
            latency=makeLatency(exit,current_resource)
            

####################################################################################################################################
#run
if __name__ == "__main__":
    main()
    
'''TODO
1. dpp와 latency의 비율을 계산한다. 각각 그래프를 그려본다. 그리고 내가 원하는 V의 범위를 구한다.

2. 현재 dynamic resource 상황이 조금 인위적이고 자연스럽지가 않다. 조금 class로 만들어서 minimum spike를 지속한다든가 아니면 하나의 spike를 커스텀해서 만들고 이를 여러개 만들고 하나의 타임 range로 만든다. 

3. 매 image마다 finial-exit, latency, Y_t의 값은(V_q상태), 정확도(inference_result_mAP), # DPP & latency 비교 plotly lib를 통해 시각화
(정확도는?(정확도는 시뮬레이션이여서 알 수 가 없다. exit에 비례한다.),)
'''