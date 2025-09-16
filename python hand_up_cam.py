import cv2, numpy as np, time, torch
from ultralytics import YOLO

MODEL="yolo11n-pose.pt"
CAM_INDEX=0
CAP_W,CAP_H,REQ_FPS=640,360,60
IMG=384
KPT_TH=0.35
THRESH_PCT=60
MIN_CONSEC=2
DRAW_SKELETON=False
L_SH,R_SH,L_EL,R_EL,L_WR,R_WR,L_HIP,R_HIP=5,6,7,8,9,10,11,12

def clamp01(x): return 0.0 if x<0 else 1.0 if x>1 else x

def arm_score(kxy,kcf,sh,el,wr,hip):
    if kcf[sh]<KPT_TH or kcf[wr]<KPT_TH: return 0.0
    sh_y,wr_y=kxy[sh,1],kxy[wr,1]
    dy=sh_y-wr_y
    if dy<=0: return 0.0
    hip_ok=kcf[hip]>=KPT_TH
    norm=abs(kxy[sh,1]-kxy[hip,1]) if hip_ok else 0.15
    norm=max(norm,1e-4)
    base=clamp01(dy/(norm*1.1))
    order=1.0
    if kcf[el]>=KPT_TH:
        el_y=kxy[el,1]
        order=1.0 if (wr_y<el_y<sh_y) else 0.7
    conf=(kcf[sh]*kcf[wr]*(kcf[el] if kcf[el]>0 else kcf[sh]))**(1/3)
    return clamp01(base*order*conf)

def iou(a,b):
    ax1,ay1,ax2,ay2=a; bx1,by1,bx2,by2=b
    ix1,iy1=max(ax1,bx1),max(ay1,by1); ix2,iy2=min(ax2,bx2),min(ay2,by2)
    iw,ih=max(0,ix2-ix1),max(0,iy2-iy1)
    inter=iw*ih
    if inter==0: return 0.0
    area=(ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter
    return inter/area

def main():
    device="cuda" if torch.cuda.is_available() else "cpu"
    try: torch.set_float32_matmul_precision("high")
    except: pass
    torch.backends.cudnn.benchmark=True
    model=YOLO(MODEL)
    cap=cv2.VideoCapture(CAM_INDEX,cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,CAP_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,CAP_H)
    cap.set(cv2.CAP_PROP_FPS,REQ_FPS)
    prev=time.time()
    prev_boxes=[]; up_streak=[]; prev_pct=[]
    while True:
        ok,frame=cap.read()
        if not ok: break
        res=model(frame,imgsz=IMG,conf=0.35,verbose=False,device=device,half=(device=="cuda"))[0]
        im=frame if DRAW_SKELETON else frame.copy()
        curr_boxes=res.boxes.xyxy.cpu().numpy().astype(int) if res.boxes is not None else np.zeros((0,4),dtype=int)
        kpts=res.keypoints.xyn.cpu().numpy() if res.keypoints is not None else np.zeros((0,17,2),dtype=float)
        kconf=res.keypoints.conf.cpu().numpy() if (res.keypoints is not None and res.keypoints.conf is not None) else np.ones((len(kpts),17),dtype=float)
        new_streak=[0]*len(curr_boxes); new_pct=[0]*len(curr_boxes)
        for i,box in enumerate(curr_boxes):
            kxy=kpts[i]; kcf=kconf[i]
            left=arm_score(kxy,kcf,L_SH,L_EL,L_WR,L_HIP)
            right=arm_score(kxy,kcf,R_SH,R_EL,R_WR,R_HIP)
            pct=int(round(max(left,right)*100))
            new_pct[i]=pct
            is_up=pct>=THRESH_PCT
            if len(prev_boxes):
                j=int(np.argmax([iou(box,pb) for pb in prev_boxes]))
                s=up_streak[j] if (len(up_streak)>j and iou(box,prev_boxes[j])>0.1) else 0
                new_streak[i]=max(0,min(10,s+(1 if is_up else -1)))
            else:
                new_streak[i]=1 if is_up else 0
            if new_streak[i]>=MIN_CONSEC and is_up:
                x1,y1,x2,y2=box
                cv2.rectangle(im,(x1,y1),(x2,y2),(0,200,0),2)
                cv2.putText(im,f"HAND UP {pct}%",(x1,max(20,y1-8)),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,200,0),2,cv2.LINE_AA)
            elif DRAW_SKELETON and res.keypoints is not None:
                im=res.plot()
        now=time.time(); fps=1.0/(now-prev) if now>prev else 0; prev=now
        cv2.putText(im,f"{fps:.1f} FPS",(10,28),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2,cv2.LINE_AA)
        cv2.imshow("Raise-Hand %",im)
        prev_boxes,up_streak,prev_pct=curr_boxes,new_streak,new_pct
        k=cv2.waitKey(1)&0xFF
        if k in (27,ord('q')): break
    cap.release(); cv2.destroyAllWindows()

if __name__=="__main__":
    main()