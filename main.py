import cv2
import ultralytics
from ultralytics import YOLO


model=YOLO("best.pt")


cap=cv2.VideoCapture(0)

while True:
    ret,frame=cap.read()

    results=model(frame)
    if ret is False:
        break

    if len(results)>0:
        result=results[0]

        if result.boxes is not None:
            boxes=result.boxes
            for box in boxes:
                cls=int(box.cls[0])
                x1,y1,x2,y2=box.xyxy[0]
                x1=int(x1)
                y1=int(y1)
                x2=int(x2)
                y2=int(y2)

                conf=float(box.conf)

                cls_name=model.names[cls]


                cv2.rectangle(frame,
                              (x1,y1),
                              (x2,y2),
                              (255,0,255),
                              2)
                cv2.putText(frame,
                            f"class :{cls_name} conf :{conf:.2f}",
                            (x1,y1-10),
                            cv2.FONT_HERSHEY_COMPLEX,
                            1,
                            (255,0,0),
                            2
                            )    
                
        cv2.imshow("Hand signs detection ",frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
          break

cap.release()
cv2.destroyAllWindows        