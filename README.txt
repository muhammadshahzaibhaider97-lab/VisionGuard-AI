VisionGuard AI — expanded multi-mode prototype

Included modes:
1. Workplace & Warehouse Safety: flags motion in a configured review zone.
2. Restricted-Area Intrusion: flags motion in a configured restricted zone.
3. Road & Traffic Monitoring: summarizes moving regions/scene activity (not a validated vehicle counter).
4. General-Purpose Video Inspection: summarizes motion throughout the video.

Current technical limits:
- Uses OpenCV background subtraction and contour-based motion detection.
- Does not identify people or vehicles, recognize PPE, determine intent, detect collisions, or prove an intrusion.
- All events require human review.
- This package does not yet include an AWS integration or a trained object detector.
- Verify the installed OpenCV version against the competition's OpenCV 5 requirement before claiming compliance.

Install/run in Windows Command Prompt:
1. Open the project folder.
2. Activate your existing environment:
   .venv\Scripts\activate
3. Install packages if needed:
   python -m pip install -r requirements.txt
4. Run:
   python -m streamlit run app.py
