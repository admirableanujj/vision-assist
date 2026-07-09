# Project Links — VisionAssist

> Reference doc for all external resources. Update this as new links are added.

---

## Project Management

| Resource | Link | Notes |
|---|---|---|
| Task Tracker | https://docs.google.com/spreadsheets/d/1cYx6bHmgmjRESfTtepiw54KWg2WP80_3qUlFLqhfMlQ/edit?gid=1822494463#gid=1822494463 | Milestone 3 tab — Shubham owns rows 10, 14, 18 |
| GitHub Repo | https://github.com/admirableanujj/vision-assist | Main branch |
| Capstone Projects 1–6 (all links) | https://docs.google.com/document/d/1g2ROBYo1lp1IwRBy5uoMAlS-OtMAGbUxkkoql2aT1RQ/edit?tab=t.0 | Master reference doc — Project 3 section has all VisionAssist links |

---

## Grading & Rubrics

| Resource | Link | Notes |
|---|---|---|
| Problem Statement | https://docs.google.com/document/d/1sEk7Vph1l9zb-g8Pvl6bnc0NnV0JoJFThJ9Un_7cFIE/edit?tab=t.0 | Full spec, deliverables, eval criteria |
| Detailed Rubrics | https://docs.google.com/spreadsheets/d/1hYQSjnmM_lZixqvhC00vu4VN_bOwfwIs8b9bBbWHGnM/edit?gid=1932046554#gid=1932046554 | 6 criteria, Excellent=10 / Satisfactory=7 / Poor=4 |

---

## Architecture & Design (Meeting Minutes)

| Resource | Location | Notes |
|---|---|---|
| Architecture UML Deck | `meeting-minutes/VisionAssist_Architecture_UML.pdf` | 12 diagrams: overview, deployment, use case, class, 5 sequence, activity, state |
| Capstone Problem Doc | `meeting-minutes/Capstone - VisionAssist - Lost & Found AI.pdf` | Meeting notes + project spec |

---

## Technical References

### YOLO / Computer Vision
| Resource | Link | Notes |
|---|---|---|
| YOLOv8 Predict | https://docs.ultralytics.com/modes/predict/ | Inference parameters |
| YOLOv8 Train | https://docs.ultralytics.com/modes/train/ | Fine-tuning docs |
| YOLOv8 Train (YouTube) | https://www.youtube.com/watch?v=ZN3nRZT7b24 | Video walkthrough of fine-tuning |
| COCO 80 Classes | https://gist.github.com/AruniRC/7b3dadd004da04c80198557db5da4bda | Full list of classes YOLO recognises out-of-the-box |
| Object Detection Models Comparison | https://medium.com/@amitkharche/object-detection-models-explained-r-cnn-yolo-ssd-49b607c9ef6d | R-CNN vs YOLO vs SSD explained |
| SSD vs YOLO | https://www.baeldung.com/cs/object-detection-ssd-yolo | Side-by-side comparison |
| SSD Paper (arXiv) | https://arxiv.org/pdf/1512.02325 | Original SSD paper |

### Evaluation Metrics
| Resource | Link | Notes |
|---|---|---|
| mAP Explained (YouTube) | https://www.youtube.com/watch?v=QdWidmqLwbw | C23–C25 in the course playlist |
| Precision/Recall (Wikipedia) | https://en.wikipedia.org/wiki/Precision_and_recall | Conceptual reference |
| Precision/Recall Image | https://en.wikipedia.org/wiki/Precision_and_recall#/media/File:Precisionrecall.svg | Diagram for documentation/slides |

### Speech
| Resource | Notes |
|---|---|
| `pyttsx3` | Text-to-speech (offline, cross-platform) — recommended for TTS |
| `gTTS` | Google Text-to-Speech (online) — alternative TTS |
| `SpeechRecognition` | Python STT library (wraps Google, Sphinx, etc.) |

### MLOps / Infrastructure
| Resource | Link | Notes |
|---|---|---|
| Kaggle GPU Setup | https://www.kaggle.com/code/dskagglemt/enabling-and-testing-the-gpu | Free GPU for YOLO training/fine-tuning |
| MLFlow + Airflow | https://www.youtube.com/watch?v=GCY5VX0na6M&t=5s | Experiment tracking pipeline |
| MLFlow with AWS | https://www.youtube.com/watch?v=XEZ7Hx2NrO8 | Cloud MLFlow deployment |

---

## Rubric Summary (target: Excellent on all 6)

| Criterion | Weight | Excellent requires |
|---|---|---|
| Implementation | 35% | All features work flawlessly — voice, time/date, locate, register. No delays. Modular, clean code. |
| Design & Architecture | 10% | Decoupled components, readable code, scalable structure |
| Integration | 15% | Seamless voice + vision + backend working end-to-end |
| Documentation | 15% | Full system architecture doc, user manuals, code documentation |
| Innovation & Creativity | 15% | Features beyond core requirements; unique approach |
| Presentation | 10% | Live demo to review committee |
