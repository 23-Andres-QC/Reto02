#!/usr/bin/env python3
"""CapyTown lane_detector - Semana 11 (RC-2).

Amarillo (izquierda): HSV — ÚNICA referencia para calcular el error/centro.
Blanco  (derecha):    LAB — se detecta y se muestra en el debug, pero NO
                       interviene en el cálculo (mete demasiado ruido/error).

El robot va siempre a "amarillo + 11cm" (centro real del carril de 22cm).

3 bandas horizontales (superior, central, inferior) sobre la imagen: se mide
el amarillo en cada una y se promedia — usa toda la línea visible, no un
solo punto. Esos 3 puntos también trazan la línea de recorrido (guía) que
se recalcula en cada frame, mostrada en magenta en el debug.

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
from std_msgs.msg import Float32, Int32
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
            ('white_min_aspect', 1.8),   # altura/ancho mínimo para forma de línea
            ('white_max_area',   4500),  # rechazar manchas grandes (reflejos)
            # Amarillo - HSV
            ('yellow_h_min', 15),
            ('yellow_h_max', 40),
            ('yellow_s_min', 60),
            ('yellow_s_max', 255),
            ('yellow_v_min', 80),
            ('yellow_v_max', 255),
            # Geometría
            ('min_area',        150),
            ('lane_width_m',    0.22),
            ('px_per_meter',    600.0),
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
        self.publish_debug  = bool(gp('publish_debug').value)

        self.M         = None
        self.warp_size = None

        # Filtro EMA sobre los centroides — reduce jitter frame-a-frame
        # que de otro modo se amplifica en el término D del PID
        self.x_yellow_f = None
        self.x_white_f  = None
        self.ema_alpha  = 0.5

        self.sub     = self.create_subscription(
            Image, '/image_raw', self.on_image, 10)
        self.pub_err = self.create_publisher(Float32, '/lane_error', 10)
        self.pub_dbg = self.create_publisher(Image, '/lane/debug_image', 10)
        self.pub_servo = self.create_publisher(Int32, '/servo_s2', 10)

        # Posición inicial de cámara — publica una vez tras 0.5s
        self._servo_sent = False
        self._servo_timer = self.create_timer(0.5, self._init_servo)

        self.get_logger().info('lane_detector listo.')
        self.get_logger().info(
            f'yellow HSV [{self.yellow_lo}] - [{self.yellow_hi}]  '
            f'white LAB [{self.white_lo_lab}] - [{self.white_hi_lab}]')

    # ------------------------------------------------------------------
    def _init_servo(self):
        if self._servo_sent:
            return
        msg = Int32()
        msg.data = -45
        self.pub_servo.publish(msg)
        self.get_logger().info('Servo s2 → -45°')
        self._servo_sent = True
        self._servo_timer.cancel()

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

        # SOLO AMARILLO para calibración/error — el blanco se sigue detectando
        # y mostrando en el debug, pero no se usa en el cálculo (mete demasiado
        # error/ruido). 3 bandas (superior, central, inferior) sobre la imagen:
        # se calcula amarillo+11cm en cada una y se traza la línea de recorrido
        # (guía) que minimiza el error, recalculada en cada frame.
        band_rows  = [h // 6, h // 2, (5 * h) // 6]   # superior, central, inferior
        band_slices = [
            slice(0, h // 3),
            slice(h // 3, (2 * h) // 3),
            slice((2 * h) // 3, h),
        ]

        lane_width_px = self.lane_width_m * self.px_per_meter

        def _band_yellow_center(sl):
            """Centro del carril (amarillo+11cm) en una banda, usando solo amarillo."""
            xy = self._centroid_x(mask_yellow[sl, :])
            if xy is None:
                return None, None
            return xy, xy + lane_width_px / 2.0

        band_points    = [_band_yellow_center(sl) for sl in band_slices]  # [(xy,xc), ...]
        trajectory_pts = [(c, r) for (_, c), r in zip(band_points, band_rows) if c is not None]

        # Promedio de los centroides de amarillo válidos en las 3 bandas — usa
        # toda la línea visible, no solo un punto, para el cálculo del error.
        yellow_vals  = [xy for xy, _ in band_points if xy is not None]
        x_yellow_raw = sum(yellow_vals) / len(yellow_vals) if yellow_vals else None

        # Blanco: solo se calcula para mostrarlo en el debug, NO se usa en el error
        x_white_raw = self._centroid_x(mask_white)

        # Filtro EMA — suaviza el centroide antes de usarlo en el cálculo de error
        x_yellow = self._ema_update('x_yellow_f', x_yellow_raw)
        x_white  = self._ema_update('x_white_f',  x_white_raw)   # solo informativo

        # Centro del carril — siempre amarillo + 11cm (el blanco no interviene)
        center_px = x_yellow + lane_width_px / 2.0 if x_yellow is not None else None

        error_m = (center_px - w / 2.0) / self.px_per_meter if center_px is not None else float('nan')

        # Log de diagnóstico: posición robot, posición amarillo, separación real vs la mitad
        # del carril esperada (target_cm, derivado de lane_width_m — nunca un número fijo).
        # error_sep_cm = separacion_cm - target_cm → 0 = separación correcta
        #                                             negativo = se ACERCA al amarillo
        #                                             positivo = se ALEJA del amarillo
        if x_yellow is not None:
            target_cm     = self.lane_width_m * 100.0 / 2.0   # mitad del carril, en cm
            separacion_cm = ((w / 2.0) - x_yellow) / self.px_per_meter * 100.0
            error_sep_cm  = separacion_cm - target_cm
            if error_sep_cm < -0.3:
                estado = 'se ACERCA al amarillo'
            elif error_sep_cm > 0.3:
                estado = 'se ALEJA del amarillo'
            else:
                estado = f'separación correcta ({target_cm:.1f}cm)'
            self.get_logger().info(
                f'Robot(centro)={w/2:.0f}px  Amarillo={x_yellow:.0f}px  '
                f'separación={separacion_cm:.1f}cm  error={error_sep_cm:+.1f}cm  → {estado}',
                throttle_duration_sec=0.5)
        else:
            self.get_logger().info('Sin amarillo detectado', throttle_duration_sec=0.5)

        out      = Float32()
        out.data = float(error_m)
        self.pub_err.publish(out)

        if self.publish_debug:
            self._publish_debug(warp, mask_white, mask_yellow, band_rows,
                                x_white, x_yellow, center_px, msg, trajectory_pts)

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

    def _ema_update(self, attr, value):
        """Filtro exponencial: suaviza la lectura cruda, resetea si se pierde detección."""
        prev = getattr(self, attr)
        if value is None:
            setattr(self, attr, None)
            return None
        if prev is None:
            setattr(self, attr, value)
            return value
        filtered = (1.0 - self.ema_alpha) * prev + self.ema_alpha * value
        setattr(self, attr, filtered)
        return filtered

    @staticmethod
    def _centroid_x(mask):
        m = cv2.moments(mask, binaryImage=True)
        if m['m00'] < 1e-3:
            return None
        return m['m10'] / m['m00']

    def _publish_debug(self, warp, mask_white, mask_yellow, band_rows,
                       xw, xy, xc, header_msg, trajectory_pts=None):
        h, w = warp.shape[:2]

        # Overlay translúcido sobre la cámara real (bird's-eye), no fondo negro:
        # se ve la pista tal cual la cámara la capta, con las detecciones resaltadas.
        overlay = warp.copy()
        overlay[mask_white  > 0] = (255, 255, 255)  # blanco detectado (solo informativo)
        overlay[mask_yellow > 0] = (0, 255, 255)    # amarillo detectado (cyan, encima) — el que se usa
        dbg = cv2.addWeighted(overlay, 0.55, warp, 0.45, 0)

        # 3 líneas verdes = las 3 bandas (superior, central, inferior) donde
        # se mide el amarillo para trazar la línea de recorrido
        for r in band_rows:
            cv2.line(dbg, (0, r), (w, r), (0, 255, 0), 1)
        cv2.line(dbg, (w // 2, 0), (w // 2, h), (128, 128, 128), 1)

        # Línea de recorrido planeada: une los centros calculados en las
        # bandas superior/central/inferior — magenta, bien distinguible.
        if trajectory_pts and len(trajectory_pts) >= 2:
            pts = np.array([[int(x), int(y)] for x, y in trajectory_pts], dtype=np.int32)
            cv2.polylines(dbg, [pts], False, (255, 0, 255), 2)
            for x, y in trajectory_pts:
                cv2.circle(dbg, (int(x), int(y)), 4, (255, 0, 255), -1)

        mid_row = band_rows[1]   # banda central, para los marcadores W/Y/C
        for x, color, label in (
            (xw, (200, 200, 200), 'W'),
            (xy, (0, 255, 255),   'Y'),
            (xc, (0, 0, 255),     'C'),
        ):
            if x is not None:
                cv2.circle(dbg, (int(x), mid_row), 6, color, -1)
                cv2.putText(dbg, label, (int(x) + 8, mid_row - 4),
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
