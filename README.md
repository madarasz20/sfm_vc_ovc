Mobile 3D Reconstruction App (Structure-from-Motion)

  This project is a mobile Structure-from-Motion (SfM) pipeline implemented for Android using OpenCV.
  The goal is to allow a user to capture a sequence of photos on their phone and generate a 3D point cloud locally on the device without any server-side computation.

OBJECTIVE

  The original goal of the project was to reconstruct indoor room geometry from photos.
  During experimentation it became clear that classical SfM pipelines struggle in indoor conditions on mobile hardware (low texture, motion blur, parallax issues, lighting instability).

HOW IT WORKS

  1. CAMERA CALIBRATION
     Estimates the camera intrinsic matrix using a set of 9x6 calibration board images
  
  2. FEATURE DETECTION
     ORB features are extracted from each captured image
     A FLANN-based matcher pairs these features between consecutive frames
  
  3. POSE ESTIMATION
     Essential matrix estimation with RANSAC
     Recovering the relative camera rotation and translation
     PnP refinement (Optional)
  
  4. TRIANGULATION
     3D points are computed from 2D correspondences across image pairs
  
  5. POINT CLOUD EXPORT & VISUALIZATION
     The reconstructed 3D points are saved as a ".ply" file
     An integrated OpenGL viewer allows the user to inspect the point cloud
  
  6. DEBUG VISUALIZATION
     The app can save and display match overlays to show which image regions contribute to the reconstruction.


LIMITATIONS

During testing, several well-known SfM failure cases were observed:
        1. Low texture surfaces (e.g. white walls) has very few to no features to detect by ORB
        2. Artiffical lighting causes inconsistent camera exposure, causing the calibration matrix to be not accurate
        3. Even minimal camera rotations cause a weak parallax and an unstable essential matrix
        4. Picture depth causes mismatched and noisy triangulation
        5. Mobil sensor noise degrades the descriptor quality

RECOMMENDED USAGE (MVP)

   For the best result calpture slow, stable motion in a straight line. Ensure good natural lighting. 
   Choose objects with deep texture. Avoid glossy sufraces and high-contrast enviroments.



        

    
   
