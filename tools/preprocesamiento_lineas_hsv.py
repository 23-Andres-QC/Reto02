import cv2
import numpy as np
import math
from pathlib import Path

# USO:
# python3 preprocesamiento_lineas_hsv.py imagen.png
# Salidas: prueba_1..4_*.png + mascara_final_amarillo.png + mascara_final_blanco.png
# Herramienta OFFLINE (no es nodo ROS) para calibrar visualmente los rangos
# HSV de amarillo/blanco a partir de una captura de pantalla o frame guardado.

INPUT_PATH = Path(__import__('sys').argv[1]) if len(__import__('sys').argv) > 1 else Path('captura.png')
OUT_DIR = INPUT_PATH.parent

# Si tu imagen es captura de rqt_image_view, recorta la barra superior.
# Para imagen directa del tópico /image_raw, pon CROP_Y0 = 0.
CROP_Y0 = 120

# Rangos calibrados para la captura que enviaste.
# Blanco real: baja saturación, alto brillo. Amarillo: tono 15-45, saturación media/alta.
WHITE_LO  = np.array([0, 0, 170])
WHITE_HI  = np.array([180, 65, 255])
YELLOW_LO = np.array([15, 45, 80])
YELLOW_HI = np.array([45, 255, 255])


def make_roi(h, w):
    """ROI para ignorar fondo, estantes y parte superior de la imagen."""
    roi = np.zeros((h, w), dtype=np.uint8)
    poly = np.array([[(0, 95), (w, 95), (w, h), (0, h)]], dtype=np.int32)
    cv2.fillPoly(roi, poly, 255)
    return roi


def hsv_masks(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask_y = cv2.inRange(hsv, YELLOW_LO, YELLOW_HI)
    mask_w = cv2.inRange(hsv, WHITE_LO, WHITE_HI)
    return mask_y, mask_w


def morph_clean(mask):
    """Quita puntos pequeños y une cortes en la cinta."""
    m = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=1)
    return m


def component_filter(mask, color_name):
    """Conserva solo componentes grandes y alargados: las cintas, no reflejos."""
    H, W = mask.shape[:2]
    num, labels, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask)

    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        cx, cy = cents[i]
        pts = np.column_stack(np.where(labels == i))

        elong = 0.0
        if len(pts) > 10:
            xy = pts[:, ::-1].astype(np.float32)
            _, _, eigval = cv2.PCACompute2(xy, mean=None)
            elong = float(eigval[0, 0] / (eigval[1, 0] + 1e-6))

        if color_name == 'yellow':
            keep = area > 500 and elong > 8 and cy > 100
        else:
            keep = area > 1000 and elong > 5 and cx > W * 0.45 and cy > 100

        if keep:
            out[labels == i] = 255

    return out


def overlay_masks(bgr, mask_y, mask_w, alpha=0.60):
    """Amarillo detectado en amarillo; blanco detectado en celeste para verlo claro."""
    color = np.zeros_like(bgr)
    color[mask_y > 0] = (0, 255, 255)      # amarillo en BGR
    color[mask_w > 0] = (255, 180, 0)      # celeste en BGR
    blended = cv2.addWeighted(bgr, 1 - alpha, color, alpha, 0)
    no_mask = (mask_y == 0) & (mask_w == 0)
    blended[no_mask] = bgr[no_mask]
    return blended


def draw_final(bgr, mask_y, mask_w):
    out = overlay_masks(bgr, mask_y, mask_w, alpha=0.50)
    h, w = bgr.shape[:2]

    # Hough sobre las máscaras filtradas: solo líneas largas.
    for mask, color in [(mask_y, (0, 255, 255)), (mask_w, (255, 120, 0))]:
        edges = cv2.Canny(mask, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30,
                                minLineLength=70, maxLineGap=25)
        if lines is not None:
            segs = []
            for x1, y1, x2, y2 in lines.reshape(-1, 4):
                segs.append((math.hypot(x2 - x1, y2 - y1), x1, y1, x2, y2))
            segs.sort(reverse=True)
            for _, x1, y1, x2, y2 in segs[:2]:
                cv2.line(out, (x1, y1), (x2, y2), color, 5)

    # Fila de muestra para calcular Y, W y C.
    row = int(0.72 * h)
    band = slice(max(0, row - 8), min(h, row + 8))

    def centroid_x(mask_band):
        m = cv2.moments(mask_band, binaryImage=True)
        return None if m['m00'] < 1e-3 else m['m10'] / m['m00']

    xy = centroid_x(mask_y[band, :])
    xw = centroid_x(mask_w[band, :])

    cv2.line(out, (0, row), (w, row), (0, 255, 0), 1)
    if xy is not None:
        cv2.circle(out, (int(xy), row), 8, (0, 255, 255), -1)
        cv2.putText(out, 'Y', (int(xy) + 8, row - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    if xw is not None:
        cv2.circle(out, (int(xw), row), 8, (255, 120, 0), -1)
        cv2.putText(out, 'W', (int(xw) - 35, row - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 120, 0), 2)
    if xy is not None and xw is not None:
        xc = int((xy + xw) / 2)
        cv2.circle(out, (xc, row), 8, (0, 0, 255), -1)
        cv2.putText(out, 'C', (xc + 8, row - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(out, 'C=(Y+W)/2', (max(5, xc - 80), row + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

    return out


def main():
    raw = cv2.imread(str(INPUT_PATH))
    if raw is None:
        raise FileNotFoundError(INPUT_PATH)

    img = raw[CROP_Y0:].copy()
    h, w = img.shape[:2]

    # Prueba 1: HSV simple
    mask_y1, mask_w1 = hsv_masks(img)
    out1 = overlay_masks(img, mask_y1, mask_w1)

    # Prueba 2: ROI + morfología
    roi = make_roi(h, w)
    mask_y2 = morph_clean(cv2.bitwise_and(mask_y1, roi))
    mask_w2 = morph_clean(cv2.bitwise_and(mask_w1, roi))
    out2 = overlay_masks(img, mask_y2, mask_w2)

    # Prueba 3: componentes grandes/elongados
    mask_y3 = component_filter(mask_y2, 'yellow')
    mask_w3 = component_filter(mask_w2, 'white')
    out3 = overlay_masks(img, mask_y3, mask_w3)

    # Prueba 4: resultado final con Hough + centro
    out4 = draw_final(img, mask_y3, mask_w3)

    cv2.imwrite(str(OUT_DIR / 'prueba_1_hsv_simple_color.png'), out1)
    cv2.imwrite(str(OUT_DIR / 'prueba_2_roi_morfologia_color.png'), out2)
    cv2.imwrite(str(OUT_DIR / 'prueba_3_componentes_grandes_color.png'), out3)
    cv2.imwrite(str(OUT_DIR / 'prueba_4_hough_final_color.png'), out4)
    cv2.imwrite(str(OUT_DIR / 'mascara_final_amarillo.png'), mask_y3)
    cv2.imwrite(str(OUT_DIR / 'mascara_final_blanco.png'), mask_w3)

    print('OK. Archivos generados en:', OUT_DIR)


if __name__ == '__main__':
    main()
