from __future__ import annotations
import cv2
import numpy as np

class ORBAligner:
    """
    Aligns a scanned page image to a template using ORB feature matching + homography.
    """
    def __init__(self, template_bgr: np.ndarray):
        self.template = template_bgr
        self.tgray = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)

        self.orb = cv2.ORB_create(nfeatures=4000)
        self.kp_t, self.des_t = self.orb.detectAndCompute(self.tgray, None)

        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def align(self, scan_bgr: np.ndarray) -> np.ndarray:
        sgray = cv2.cvtColor(scan_bgr, cv2.COLOR_BGR2GRAY)
        kp_s, des_s = self.orb.detectAndCompute(sgray, None)

        if des_s is None or self.des_t is None or len(kp_s) < 20:
            raise RuntimeError("Alignment failed: not enough features.")

        knn = self.matcher.knnMatch(des_s, self.des_t, k=2)

        good = []
        for m, n in knn:
            if m.distance < 0.75 * n.distance:
                good.append(m)

        if len(good) < 25:
            raise RuntimeError("Alignment failed: insufficient good matches.")

        src_pts = np.float32([kp_s[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([self.kp_t[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        H, _mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H is None:
            raise RuntimeError("Alignment failed: homography not found.")

        h, w = self.template.shape[:2]
        aligned = cv2.warpPerspective(scan_bgr, H, (w, h))
        return aligned
