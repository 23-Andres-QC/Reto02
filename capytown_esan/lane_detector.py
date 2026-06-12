#!/usr/bin/env python3
"""CapyTown lane_detector - Semana 11 (RC-2).

Segmenta el borde blanco (derecha) y el eje amarillo (izquierda) por HSV,
aplica una vista de pájaro (IPM) y publica el error lateral en metros
sobre /lane_error.

Convención de signo:
  error > 0  →  centro del carril a la DERECHA del robot  →  girar derecha (ω < 0)
  error < 0  →  centro del carril a la IZQUIERDA del robot →  girar izquierda (ω > 0)
"""

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
            ('white_h_min',  0),
            ('white_h_max',  180),
            ('white_s_min',  0),
            ('white_s_max',  30),
            ('white_v_min',  180),
            ('white_v_max',  255),
            ('yellow_h_min', 15),
            ('yellow_h_max', 40),
            ('yellow_s_min', 80),
            ('yellow_s_max', 255),
            ('yellow_v_min', 80),
            ('yellow_v_max', 255),
            ('min_area',        150),
            ('lane_width_m',    0.21),
            ('px_per_meter',    600.0),
            ('look_ahead_row',  0.6),
            ('publish_debug',   True),
        ])

        gp = self.get_parameter
        self.white_lo = np.array([gp('white_h_min').value,
                                   gp('white_s_min').value,
                                   gp('white_v_min').value], dtype=np.uint8)
        self.white_hi = np.array([gp('white_h_max').value,
                                   gp('white_s_max').value,
                                   gp('white_v_max').value], dtype=np.uint8)
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
            f'white HSV [{self.white_lo}] - [{self.white_hi}]  '
            f'yellow HSV [{self.yellow_lo}] - [{self.yellow_hi}]')

    # ------------------------------------------------------------------
    # IPM: transforma imagen de cámara a vista de pájaro (bird's-eye)
    # ------------------------------------------------------------------
    def build_ipm(self, w, h):
        """Calcula la homografía para la vista de pájaro.

        Cámara mirando hacia adelante y ligeramente hacia abajo.
        El trapecio src captura el suelo visible desde ~55% de la imagen.

        Para ajustar en el lab:
          - Si las líneas se ven muy lejos: subir 0.55 → 0.50
          - Si las líneas se cortan abajo: bajar 0.97 → 0.99
          - Si el trapecio es muy estrecho: ampliar 0.20/0.80 → 0.10/0.90
        """
        src = np.float32([
            [0.20 * w, 0.55 * h],   # arriba-izquierda  (horizonte del suelo)
            [0.80 * w, 0.55 * h],   # arriba-derecha
            [1.00 * w, 0.97 * h],   # abajo-derecha     (frente del robot)
            [0.00 * w, 0.97 * h],   # abajo-izquierda
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
    # Callback principal
    # ------------------------------------------------------------------
    def on_image(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge: {e}')
            return

        # Rotar imagen 90° horario para corregir cámara girada
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        h, w = frame.shape[:2]

        if self.M is None:
            self.build_ipm(w, h)

        # 1) Vista de pájaro (IPM)
        warp = cv2.warpPerspective(frame, self.M, self.warp_size)
        hsv  = cv2.cvtColor(warp, cv2.COLOR_BGR2HSV)

        # 2) Máscaras de color
        mask_white  = cv2.inRange(hsv, self.white_lo,  self.white_hi)
        mask_yellow = cv2.inRange(hsv, self.yellow_lo, self.yellow_hi)

        kernel = np.ones((3, 3), np.uint8)
        mask_white  = cv2.morphologyEx(mask_white,  cv2.MORPH_OPEN, kernel)
        mask_white  = cv2.morphologyEx(mask_white,  cv2.MORPH_CLOSE, kernel)
        mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, kernel)
        mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_CLOSE, kernel)

        # 3) Fila de muestreo (look-ahead): banda horizontal de ±8 px
        row  = int(self.look_ahead_row * h)
        band = slice(max(0, row - 8), min(h, row + 8))
        x_white  = self._centroid_x(mask_white[band, :])
        x_yellow = self._centroid_x(mask_yellow[band, :])

        # 4) Centro del carril (px) → error lateral (m)
        #
        # Carril interno: amarillo a la IZQUIERDA, blanco a la DERECHA.
        # Si solo se ve una línea, se estima la otra usando lane_width_px.
        # error_m > 0  →  centro del carril a la derecha del robot.
        lane_width_px = self.lane_width_m * self.px_per_meter

        if x_white is not None and x_yellow is not None:
            center_px = (x_white + x_yellow) / 2.0
        elif x_white is not None:
            # Solo línea blanca (derecha): estima amarilla a lane_width_px a su izquierda
            center_px = x_white - lane_width_px / 2.0
        elif x_yellow is not None:
            # Solo línea amarilla (izquierda): estima blanca a lane_width_px a su derecha
            center_px = x_yellow + lane_width_px / 2.0
        else:
            center_px = None

        if center_px is not None:
            error_m = (center_px - w / 2.0) / self.px_per_meter
        else:
            error_m = float('nan')

        out      = Float32()
        out.data = float(error_m)
        self.pub_err.publish(out)

        if self.publish_debug:
            self._publish_debug(warp, mask_white, mask_yellow, row, x_white, x_yellow, center_px, msg)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _centroid_x(mask):
        """Centroide horizontal de la máscara (None si muy pequeño)."""
        m = cv2.moments(mask, binaryImage=True)
        if m['m00'] < 1e-3:
            return None
        return m['m10'] / m['m00']

    def _publish_debug(self, warp, mask_white, mask_yellow, row, xw, xy, xc, header_msg):
        h, w = warp.shape[:2]
        dbg = np.zeros((h, w, 3), dtype=np.uint8)

        # Colorear píxeles detectados: blanco y amarillo sobre negro
        dbg[mask_white  > 0] = (255, 255, 255)
        dbg[mask_yellow > 0] = (0, 255, 255)

        # Línea de look-ahead
        cv2.line(dbg, (0, row), (w, row), (0, 255, 0), 1)

        # Puntos de centroide y centro
        for x, color, label in (
            (xw, (255, 255, 255), 'W'),
            (xy, (0, 255, 255),   'Y'),
            (xc, (0, 0, 255),     'C'),
        ):
            if x is not None:
                cv2.circle(dbg, (int(x), row), 6, color, -1)
                cv2.putText(dbg, label, (int(x) + 8, row - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Centro de imagen
        cv2.line(dbg, (w // 2, 0), (w // 2, h), (128, 128, 128), 1)

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
