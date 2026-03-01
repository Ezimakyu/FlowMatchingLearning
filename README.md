# FlowMatchingLearning


(Note to later that for project setup, will need to run this)
modal deploy backend/modal/transcription_service.py
modal deploy backend/modal/vision_extraction_service.py
modal deploy backend/modal/embedding_service.py

Understanding some params:
--doc-id -> document id (drives chunk ids)
--source-file-id -> id for file (to trace)
--model-profile (test, demo) 