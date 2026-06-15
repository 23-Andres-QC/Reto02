#!/usr/bin/env python3
"""CapyTown lane_detector - Semana 11 (RC-2).

Amarillo (izquierda): HSV
Blanco  (derecha):    LAB — más robusto ante reflejos que HSV

El robot va al CENTRO entre las dos líneas.

Convención de signo:
  error > 0  →  centro a la DERECHA  →  girar derecha (ω < 0)
  error < 0  →  centro a la IZQUIERDA →  girar izquierda (ω > 0)
"""

import math
import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from cv_bridge import CvBridge


class LaneDetector(Node):
    def __init__(self):
        super().__init__('lane_detector')
        self.bridge = CvBridge()

        self.declare_parameters('', [
            # Blanco - LAB
            ('white_l_min',      130),
            ('white_l_max',      255),
            ('white_a_min',      100),
            ('white_a_max',      155),
            ('white_b_min',      100),
            ('white_b_max',      155),
            ('white_min_aspect', 2.5),   # altura/ancho mínimo para forma de línea
            ('white_max_area',   2500),  # rechazar manchas grandes (reflejos)
            # Amarillo - HSV
            ('yellow_h_min', 15),
            ('yellow_h_max', 40),
            ('yellow_s_min', 60),
            ('yellow_s_max', 255),
            ('yellow_v_min', 80),
            ('yellow_v_max', 255),
            # Geometría
            ('min_area',        150),
            ('lane_width_m',    0.21),
            ('px_per_meter',    600.0),
            ('look_ahead_row',  0.88),
            ('publish_debug',   True),
        ])

        gp = self.get_parameter

        self.white_lo_lab     = np.array([gp('white_l_min').value,
                                           gp('white_a_min').value,
                                           gp('white_b_min').value], dtype=np.uint8)
        self.white_hi_lab     = np.array([gp('white_l_max').value,
                                           gp('white_a_max').value,
                                           gp('white_b_max').value], dtype=np.uint8)
        self.white_min_aspect = float(gp('white_min_aspect').value)
        self.white_max_area   = float(gp('white_max_area').value)

        self.yellow_lo = np.array([gp('yellow_h_min').value,
                                    gp('yellow_s_min').value,
                                    gp('yellow_v_min').value], dtype=np.uint8)
        self.yellow_hi = np.array([gp('yellow_h_max').value,
                                    gp('yellow_s_max').value,
                                    gp('yellow_v_max').value], dtype=np.uint8)

        self.min_area       = float(gp('min_area').value)
        self.lane_width_m   = float(gp('lane_width_m').value)
        self.px_per_meter   = float(gp('px_per_meter').value)
        self.look_ahead_row = float(gp('look_ahead_row').value)
        self.publish_debug  = bool(gp('publish_debug').value)

        self.M         = None
        self.warp_size = None

        self.sub     = self.create_subscription(
            Image, '/image_raw', self.on_image, 10)
        self.pub_err = self.create_publisher(Float32, '/lane_error', 10)
        self.pub_dbg = self.create_publisher(Image, '/lane/debug_image', 10)

        self.get_logger().info('lane_detector listo.')
        self.get_logger().info(
            f'yellow HSV [{self.yellow_lo}] - [{self.yellow_hi}]  '
            f'white LAB [{self.white_lo_lab}] - [{self.white_hi_lab}]')

    # ------------------------------------------------------------------
    def build_ipm(self, w, h):
        src = np.float32([
            [0.20 * w, 0.55 * h],
            [0.80 * w, 0.55 * h],
            [1.00 * w, 0.97 * h],
            [0.00 * w, 0.97 * h],
        ])
        dst = np.float32([
            [0.25 * w, 0.0],
            [0.75 * w, 0.0],
            [0.75 * w,  h],
            [0.25 * w,  h],
        ])
        self.M         = cv2.getPerspectiveTransform(src, dst)
        self.warp_size = (w, h)

    # ------------------------------------------------------------------
    def on_image(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge: {e}')
            return

        h, w = frame.shape[:2]
        if self.M is None:
            self.build_ipm(w, h)

        warp = cv2.warpPerspective(frame, self.M, self.warp_size)

        # Amarillo: HSV
        hsv         = cv2.cvtColor(warp, cv2.COLOR_BGR2HSV)
        mask_yellow = cv2.inRange(hsv, self.yellow_lo, self.yellow_hi)

        # Blanco: LAB
        lab             = cv2.cvtColor(warp, cv2.COLOR_BGR2LAB)
        mask_white_raw  = cv2.inRange(lab, self.white_lo_lab, self.white_hi_lab)

        kernel = np.ones((3, 3), np.uint8)
        mask_yellow    = cv2.morphologyEx(mask_yellow,   cv2.MORPH_OPEN,  kernel)
        mask_yellow    = cv2.morphologyEx(mask_yellow,   cv2.MORPH_CLOSE, kernel)
        mask_white_raw = cv2.morphologyEx(mask_white_raw, cv2.MORPH_OPEN,  kernel)
        mask_white_raw = cv2.morphologyEx(mask_white_raw, cv2.MORPH_CLOSE, kernel)

        # Excluir píxeles amarillos del blanco
        mask_white_raw = cv2.bitwise_and(mask_white_raw, cv2.bitwise_not(mask_yellow))

        # Filtrar blanco: solo líneas consecutivas, no manchas grandes
        mask_white = self._filter_line_shape(mask_white_raw)

        row  = int(self.look_ahead_row * h)
        band = slice(max(0, row - 8), min(h, row + 8))
        x_yellow = self._centroid_x(mask_yellow[band, :])
        x_white  = self._centroid_x(mask_white[band, :])

        lane_width_px = self.lane_width_m * self.px_per_meter

        # Rechazar blanco que esté a la izquierda del amarillo (imposible físicamente)
        # o que esté fuera del rango 60-130% del ancho de carril esperado
        if x_yellow is not None and x_white is not None:
            dist_px = x_white - x_yellow
            if dist_px <= 0 or dist_px < lane_width_px * 0.6 or dist_px > lane_width_px * 1.3:
                x_white = None

        # Centro del carril
        if x_yellow is not None and x_white is not None:
            center_px = (x_yellow + x_white) / 2.0
        elif x_yellow is not None:
            # Solo amarillo: asumimos que el blanco está 11cm a la derecha
            center_px = x_yellow + lane_width_px / 2.0
        else:
            center_px = None  # sin amarillo → publicar NaN

        error_m = (center_px - w / 2.0) / self.px_per_meter if center_px is not None else float('nan')

        # Si el blanco detectado empujaría el error más allá del umbral del carril
        # (center_px lejos del expected) ya fue descartado arriba.
        # Aquí: corrección suave si el amarillo se acerca al centro (robot derivando izquierda)
        if x_yellow is not None and not math.isnan(error_m):
            yellow_warn_px = w * 0.38   # 38% desde la izquierda
            if x_yellow > yellow_warn_px:
                # Empuje suave a la derecha proporcional a la intrusión
                proximity = (x_yellow - yellow_warn_px) / self.px_per_meter * 0.8
                error_m += proximity

        out      = Float32()
        out.data = float(error_m)
        self.pub_err.publish(out)

        if self.publish_debug:
            self._publish_debug(warp, mask_white, mask_yellow, row,
                                x_white, x_yellow, center_px, msg)

    # ------------------------------------------------------------------
    def _filter_line_shape(self, mask):
        """Mantiene contornos con forma de línea (alto/ancho >= min_aspect).
        Rechaza manchas grandes (reflejos) y pequeñas (ruido)."""
        result   = np.zeros_like(mask)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area or area > self.white_max_area:
                continue
            _, _, cw, ch = cv2.boundingRect(cnt)
            if ch / (cw + 1e-5) >= self.white_min_aspect:
                cv2.drawContours(result, [cnt], -1, 255, -1)
        return result

    @staticmethod
    def _centroid_x(mask):
        m = cv2.moments(mask, binaryImage=True)
        if m['m00'] < 1e-3:
            return None
        return m['m10'] / m['m00']

    def _publish_debug(self, warp, mask_white, mask_yellow, row,
                       xw, xy, xc, header_msg):
        h, w = warp.shape[:2]
        dbg = np.zeros((h, w, 3), dtype=np.uint8)

        dbg[mask_white  > 0] = (255, 255, 255) # blanco   → blanco
        dbg[mask_yellow > 0] = (0, 255, 255)   # amarillo → cyan (encima del blanco)

        cv2.line(dbg, (0, row), (w, row), (0, 255, 0), 1)
        cv2.line(dbg, (w // 2, 0), (w // 2, h), (128, 128, 128), 1)

        for x, color, label in (
            (xw, (200, 200, 200), 'W'),
            (xy, (0, 255, 255),   'Y'),
            (xc, (0, 0, 255),     'C'),
        ):
            if x is not None:
                cv2.circle(dbg, (int(x), row), 6, color, -1)
                cv2.putText(dbg, label, (int(x) + 8, row - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        out        = self.bridge.cv2_to_imgmsg(dbg, 'bgr8')
        out.header = header_msg.header
        self.pub_dbg.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = LaneDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
