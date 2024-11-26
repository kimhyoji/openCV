import cv2
import mediapipe as mp #pip install mediapipe

# Mediapipe 초기화
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Mediapipe 얼굴 검출 모델 로드
with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
    # 이미지 읽기
    image = cv2.imread('kalina.jpg')  # 얼굴 이미지 경로로 교체
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Mediapipe는 RGB 이미지를 처리

    # Mediapipe로 얼굴 검출
    results = face_detection.process(rgb_image)

    if results.detections:
        for detection in results.detections:
            # 검출된 얼굴의 바운딩 박스 정보 추출
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = image.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                int(bboxC.width * iw), int(bboxC.height * ih)

            # 얼굴 영역 추출 및 모자이크 처리
            face = image[y:y + h, x:x + w]
            face = cv2.resize(face, (max(w // 10, 1), max(h // 10, 1)))  # 축소
            face = cv2.resize(face, (w, h), interpolation=cv2.INTER_NEAREST)  # 확대
            image[y:y + h, x:x + w] = face

    # 결과 출력
    cv2.imshow('Mosaic Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 결과 저장
    cv2.imwrite('../test/mosaic_image.jpg', image)
