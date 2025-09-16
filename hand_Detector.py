import os,glob,cv2,numpy as np,torch
from ultralytics import YOLO

def nms(boxes,thr=0.5):
    if not boxes: return []
    b=np.array(boxes,dtype=float)
    x1,y1,x2,y2=b[:,0],b[:,1],b[:,2],b[:,3]
    areas=(x2-x1+1)*(y2-y1+1)
    order=areas.argsort()[::-1]
    keep=[]
    while order.size>0:
        i=order[0]
        keep.append((int(x1[i]),int(y1[i]),int(x2[i]),int(y2[i])))
        xx1=np.maximum(x1[i],x1[order[1:]])
        yy1=np.maximum(y1[i],y1[order[1:]])
        xx2=np.minimum(x2[i],x2[order[1:]])
        yy2=np.minimum(y2[i],y2[order[1:]])
        w=np.maximum(0,xx2-xx1+1); h=np.maximum(0,yy2-yy1+1)
        inter=w*h
        ovr=inter/(areas[i]+areas[order[1:]]-inter+1e-6)
        inds=np.where(ovr<=thr)[0]
        order=order[inds+1]
    return keep

def to_yolo(x1,y1,x2,y2,w,h):
    xc=(x1+x2)/2.0/w; yc=(y1+y2)/2.0/h
    bw=(x2-x1)/w; bh=(y2-y1)/h
    return xc,yc,bw,bh

src=r"C:\Users\User\Desktop\labled image\COCO_labels"  # ← 학습할 이미지 폴더 경로

base="datasets/handup"
img_train=os.path.join(base,"images/train")
lbl_train=os.path.join(base,"labels/train")
os.makedirs(img_train,exist_ok=True)
os.makedirs(lbl_train,exist_ok=True)

exts=("*.jpg","*.jpeg","*.png","*.bmp","*.JPG","*.PNG")
paths=[]
for e in exts: paths+=glob.glob(os.path.join(src,"**",e),recursive=True)
paths=sorted(list(set(paths)))

for p in paths:
    im=cv2.imread(p)
    if im is None: continue
    h,w=im.shape[:2]
    b,g,r=cv2.split(im)
    mask=((g>200)&(r<80)&(b<80)).astype(np.uint8)*255
    mask=cv2.dilate(mask,None,iterations=1)
    cnts,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cand=[]
    for c in cnts:
        x,y,wc,hc=cv2.boundingRect(c)
        if wc*hc<800: continue
        cand.append((x,y,x+wc,y+hc))
    boxes=nms(cand,0.5)
    name=os.path.splitext(os.path.basename(p))[0]
    out_img=os.path.join(img_train,name+".jpg")
    cv2.imwrite(out_img,im)
    with open(os.path.join(lbl_train,name+".txt"),"w",encoding="utf-8") as f:
        for (x1,y1,x2,y2) in boxes:
            xc,yc,bw,bh=to_yolo(x1,y1,x2,y2,w,h)
            f.write(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

yaml_path=os.path.join(base,"handup.yaml")
with open(yaml_path,"w",encoding="utf-8") as f:
    f.write("path: datasets/handup\ntrain: images/train\nval: images/train\nnames:\n  0: hand_up\n")

dev = 0 if (torch.cuda.is_available() and torch.cuda.device_count()>0) else "cpu"
is_cpu = (dev=="cpu")
batch = 4 if is_cpu else 16
workers = 0 if is_cpu else 4

model=YOLO("yolo11n.pt")
model.train(data=yaml_path,imgsz=416,epochs=60,batch=batch,device=dev,cos_lr=True,patience=20,cache=True,workers=workers)

best="runs/detect/train/weights/best.pt"
if os.path.exists(best):
    y=YOLO(best)
    try:
        if not is_cpu:
            y.export(format="engine",half=True)
        else:
            y.export(format="onnx",opset=12)
    except Exception:
        pass
