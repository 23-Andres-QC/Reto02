#!/usr/bin/env python3
"""CapyTown lane_detector - Semana 11 (RC-2).

Amarillo (izquierda) y Blanco (derecha): ambos en HSV.
Centro del carril: C=(Y+W)/2 si se detectan los dos colores (y la distancia
entre ellos es razonable); si solo hay amarillo, centro = amarillo + 11cm
(mitad del carril de 22cm).

Filtro de forma: elongación por PCA sobre componentes conectados (en vez de
bounding-box alto/ancho) — más robusto para distinguir cintas largas de
reflejos/manchas redondeadas. Se aplica a ambos colores.

3 bandas horizontales (superior, central, inferior) sobre la imagen: se mide
el centro en cada una y se promedia — usa toda la línea visible, no un
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
            # Blanco - HSV (baja saturación, alto brillo)
            ('white_h_min', 0),   ('white_h_max', 180),
            ('white_s_min', 0),   ('white_s_max', 65),
            ('white_v_min', 170), ('white_v_max', 255),
            ('white_min_elong',  5.0),   # elongación PCA mínima (línea, no mancha)
            ('white_min_area',   1000),  # área mínima
            ('white_max_area',   8000),  # rechazar manchas grandes (reflejos)
            # Amarillo - HSV
            ('yellow_h_min', 15),
            ('yellow_h_max', 45),
            ('yellow_s_min', 45),
            ('yellow_s_max', 255),
            ('yellow_v_min', 80),
            ('yellow_v_max', 255),
            ('yellow_min_elong', 8.0),   # elongación PCA mínima
            ('yellow_min_area',  500),   # área mínima
            ('yellow_max_area',  20000), # rechazar manchas grandes
            # Geometría
            ('lane_width_m',    0.22),
            ('px_per_meter',    600.0),
            ('publish_debug',   True),
            ('white_bias_m',    0.02),  # m — desplaza el centro objetivo hacia el amarillo (se apegaba al blanco)
        ])

        gp = self.get_parameter

        self.white_lo = np.array([gp('white_h_min').value,
                                   gp('white_s_min').value,
                                   gp('white_v_min').value], dtype=np.uint8)
        self.white_hi = np.array([gp('white_h_max').value,
                                   gp('white_s_max').value,
                                   gp('white_v_max').value], dtype=np.uint8)
        self.white_min_elong = float(gp('white_min_elong').value)
        self.white_min_area  = float(gp('white_min_area').value)
        self.white_max_area  = float(gp('white_max_area').value)

        self.yellow_lo = np.array([gp('yellow_h_min').value,
                                    gp('yellow_s_min').value,
                                    gp('yellow_v_min').value], dtype=np.uint8)
        self.yellow_hi = np.array([gp('yellow_h_max').value,
                                    gp('yellow_s_max').value,
                                    gp('yellow_v_max').value], dtype=np.uint8)
        self.yellow_min_elong = float(gp('yellow_min_elong').value)
        self.yellow_min_area  = float(gp('yellow_min_area').value)
        self.yellow_max_area  = float(gp('yellow_max_area').value)

        self.lane_width_m   = float(gp('lane_width_m').value)
        self.px_per_meter   = float(gp('px_per_meter').value)
        self.publish_debug  = bool(gp('publish_debug').value)
        self.white_bias_m   = float(gp('white_bias_m').value)

        self.M         = None
        self.warp_size = None

        # Filtro EMA sobre los centroides — reduce jitter frame-a-frame
        # que de otro modo se amplifica en el término D del PID
        self.x_yellow_f = None
        self.x_white_f  = None
        self.x_center_f = None
        self.ema_alpha  = 0.5

        self.sub     = self.create_subscription(
            Image, '/image_raw', self.on_image, 10)
        self.pub_err   = self.create_publisher(Float32, '/lane_error', 10)
        self.pub_slope = self.create_publisher(Float32, '/lane_slope', 10)
        self.pub_dbg = self.create_publisher(Image, '/lane/debug_image', 10)
        self.pub_servo = self.create_publisher(Int32, '/servo_s2', 10)

        # Posición inicial de cámara — publica una vez tras 0.5s
        self._servo_sent = False
        self._servo_timer = self.create_timer(0.5, self._init_servo)

        self.get_logger().info('lane_detector listo.')
        self.get_logger().info(
            f'yellow HSV [{self.yellow_lo}] - [{self.yellow_hi}]  '
            f'white HSV [{self.white_lo}] - [{self.white_hi}]')

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

        # Amarillo y blanco: ambos en HSV
        hsv         = cv2.cvtColor(warp, cv2.COLOR_BGR2HSV)
        mask_yellow_raw = cv2.inRange(hsv, self.yellow_lo, self.yellow_hi)
        mask_white_raw  = cv2.inRange(hsv, self.white_lo, self.white_hi)

        open_k  = np.ones((3, 3), np.uint8)
        close_k = np.ones((7, 7), np.uint8)   # cierre más grande: une cortes en la cinta
        mask_yellow_raw = cv2.morphologyEx(mask_yellow_raw, cv2.MORPH_OPEN,  open_k)
        mask_yellow_raw = cv2.morphologyEx(mask_yellow_raw, cv2.MORPH_CLOSE, close_k)
        mask_white_raw  = cv2.morphologyEx(mask_white_raw,  cv2.MORPH_OPEN,  open_k)
        mask_white_raw  = cv2.morphologyEx(mask_white_raw,  cv2.MORPH_CLOSE, close_k)

        # Excluir píxeles amarillos del blanco
        mask_white_raw = cv2.bitwise_and(mask_white_raw, cv2.bitwise_not(mask_yellow_raw))

        # Filtro de elongación PCA: conserva solo componentes largas y delgadas
        # (cintas), rechaza manchas/reflejos redondeados — aplicado a ambos colores
        mask_yellow = self._component_filter(mask_yellow_raw, self.yellow_min_area,
                                              self.yellow_max_area, self.yellow_min_elong)
        mask_white  = self._component_filter(mask_white_raw, self.white_min_area,
                                              self.white_max_area, self.white_min_elong)

        # Usa AMBOS colores para calibración/error, con centroide C=(Y+W)/2
        # cuando hay ambos — igual que en la herramienta de calibración offline.
        # Si solo hay amarillo, centro = amarillo + 11cm. 3 bandas (superior,
        # central, inferior): se traza la línea de recorrido (guía) con el
        # centro de cada banda, recalculada en cada frame.
        band_rows  = [h // 6, h // 2, (5 * h) // 6]   # superior, central, inferior
        band_slices = [
            slice(0, h // 3),
            slice(h // 3, (2 * h) // 3),
            slice((2 * h) // 3, h),
        ]

        lane_width_px = self.lane_width_m * self.px_per_meter
        white_bias_px = self.white_bias_m * self.px_per_meter

        def _band_center(sl):
            """Centro del carril en una banda: C=(Y+W)/2 si hay ambos colores
            y la distancia es razonable; si solo hay uno de los dos, ese
            color + 11cm (amarillo) o - 11cm (blanco) hacia el otro lado.
            El carrito puede avanzar con CUALQUIERA de los dos colores.
            Cuando el blanco participa del cálculo, el objetivo se desplaza
            white_bias_px hacia el amarillo — el robot se apegaba demasiado
            al blanco con el punto medio exacto."""
            xy = self._centroid_x(mask_yellow[sl, :])
            xw = self._centroid_x(mask_white[sl, :])
            if xy is not None and xw is not None:
                dist = xw - xy
                if dist <= 0 or dist < lane_width_px * 0.6 or dist > lane_width_px * 1.3:
                    xw = None   # blanco fuera de rango esperado → descartar
            if xy is not None and xw is not None:
                return xy, xw, (xy + xw) / 2.0 - white_bias_px
            elif xy is not None:
                return xy, None, xy + lane_width_px / 2.0
            elif xw is not None:
                return None, xw, xw - lane_width_px / 2.0 - white_bias_px
            return None, None, None

        band_points    = [_band_center(sl) for sl in band_slices]  # [(xy,xw,xc), ...]
        trajectory_pts = [(c, r) for (_, _, c), r in zip(band_points, band_rows) if c is not None]

        # Pendiente para los GIROS: usa solo el AMARILLO, y SOLO dentro de
        # la banda INFERIOR (ajustando una línea a sus píxeles) — no la
        # banda central. La banda central mira más adelante en la pista,
        # lo que hacía que la pendiente reaccionara a una curva que el
        # robot todavía no alcanza, anticipando el giro demasiado pronto.
        # Calculando la pendiente solo con lo que hay DENTRO de la banda
        # inferior (la más cercana al robot), la decisión de girar depende
        # únicamente de lo que está pasando justo donde está el robot, no
        # de lo que se ve más lejos.
        slope_m = self._inferior_slope(mask_yellow, band_slices[2])

        # La posición real del robot la marca la banda INFERIOR (la más
        # cercana al robot). Las bandas superior y central no son "dónde
        # está el robot" — son la guía de la pista más adelante, y solo
        # se usan para la pendiente/anticipación de curva (arriba).
        x_yellow_raw, x_white_raw, center_raw = band_points[2]

        # Filtro EMA — suaviza el centroide antes de usarlo en el cálculo de error
        x_yellow  = self._ema_update('x_yellow_f', x_yellow_raw)
        x_white   = self._ema_update('x_white_f',  x_white_raw)
        center_px = self._ema_update('x_center_f', center_raw)

        error_m = (center_px - w / 2.0) / self.px_per_meter if center_px is not None else float('nan')

        # Zona de seguridad ANTICIPADA: no solo reacciona cuando ya está cerca
        # de una línea — el margen de alerta se AGRANDA según el ángulo actual
        # (slope_m, centro vs inferior). Si la línea guía muestra que el carro
        # se está angulando hacia el amarillo o el blanco, esa tendencia indica
        # que va a salirse aunque todavía no esté dentro del margen estático —
        # se corrige antes, no cuando ya esté encima de la línea.
        # NOTA: ganancias bajadas (antes 1.8/1.2) — se detectó que en curvas
        # este empujón se sumaba al PID normal + FF de anticipación y
        # componía una corrección demasiado fuerte, causando que el error
        # rebotara de un extremo a otro en vez de asentarse suave.
        if center_px is not None:
            safety_margin_px = lane_width_px * 0.30
            angle_px = 0.0 if math.isnan(slope_m) else slope_m * self.px_per_meter
            look_ahead_gain = 0.7   # antes 1.2 — agranda menos el margen por ángulo
            boost_gain = 1.2        # antes 1.8 — empujón más suave
            max_error_m = 0.20      # tope: evita que el error se dispare sin límite

            if x_yellow is not None:
                dist_to_yellow_px = (w / 2.0) - x_yellow   # > margen = seguro
                # angle_px < 0 → x_mid < x_bot → el carril se cierra hacia la izquierda
                # (se acerca al amarillo) → agranda el margen de alerta de ese lado
                approaching_yellow = max(0.0, -angle_px)
                margin_yellow = safety_margin_px + approaching_yellow * look_ahead_gain
                if dist_to_yellow_px < margin_yellow:
                    intrusion = (margin_yellow - dist_to_yellow_px) / self.px_per_meter
                    error_m += intrusion * boost_gain

            if x_white is not None:
                dist_to_white_px = x_white - (w / 2.0)     # > margen = seguro
                approaching_white = max(0.0, angle_px)
                margin_white = safety_margin_px + approaching_white * look_ahead_gain
                if dist_to_white_px < margin_white:
                    intrusion = (margin_white - dist_to_white_px) / self.px_per_meter
                    error_m -= intrusion * boost_gain

            error_m = max(-max_error_m, min(max_error_m, error_m))

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
        elif x_white is not None:
            self.get_logger().info(
                f'Sin amarillo, usando BLANCO={x_white:.0f}px como referencia',
                throttle_duration_sec=0.5)
        else:
            self.get_logger().info('Sin amarillo ni blanco detectado — frena', throttle_duration_sec=0.5)

        out      = Float32()
        out.data = float(error_m)
        self.pub_err.publish(out)

        slope_out      = Float32()
        slope_out.data = float(slope_m)
        self.pub_slope.publish(slope_out)

        if self.publish_debug:
            self._publish_debug(warp, mask_white, mask_yellow, band_rows,
                                x_white, x_yellow, center_px, msg, trajectory_pts)

    # ------------------------------------------------------------------
    @staticmethod
    def _component_filter(mask, min_area, max_area, min_elong):
        """Conserva solo componentes conectados grandes y alargados (cintas),
        usando elongación por PCA (relación entre el autovalor mayor y el menor
        de la nube de puntos del componente) — más robusto que un bounding-box
        para distinguir líneas largas de manchas/reflejos redondeados."""
        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        out = np.zeros_like(mask)

        for i in range(1, num):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < min_area or area > max_area:
                continue

            ys, xs = np.where(labels == i)
            if len(xs) < 10:
                continue

            pts = np.column_stack((xs, ys)).astype(np.float32)
            _, _, eigval = cv2.PCACompute2(pts, mean=None)
            elong = float(eigval[0, 0] / (eigval[1, 0] + 1e-6))

            if elong >= min_elong:
                out[labels == i] = 255

        return out

    def _inferior_slope(self, mask_yellow, sl):
        """Pendiente del amarillo calculada SOLO con los píxeles dentro de
        la banda dada (la inferior) — ajusta una línea (cv2.fitLine) y
        evalúa esa línea en el borde superior e inferior de la banda. No
        usa ningún punto de fuera de la banda (ni central ni superior),
        así la pendiente refleja únicamente lo que pasa justo donde está
        el robot, no la curvatura de la pista más adelante."""
        ys, xs = np.where(mask_yellow[sl, :] > 0)
        if len(xs) < 20:
            return float('nan')
        pts = np.column_stack((xs, ys)).astype(np.float32)
        vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
        if abs(vy) < 1e-6:
            return float('nan')   # línea horizontal dentro de la banda — sin pendiente útil
        y_top = 0
        y_bot = (sl.stop - sl.start) - 1
        x_top = x0 + (y_top - y0) * (vx / vy)
        x_bot = x0 + (y_bot - y0) * (vx / vy)
        return (x_top - x_bot) / self.px_per_meter

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
        overlay[mask_white  > 0] = (255, 255, 255)  # blanco detectado
        overlay[mask_yellow > 0] = (0, 255, 255)    # amarillo detectado (cyan, encima)
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
