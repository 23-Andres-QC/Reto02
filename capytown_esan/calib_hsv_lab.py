#!/usr/bin/env python3
"""Calibración visual HSV — RC-2.

Hace EXACTAMENTE los mismos cálculos que lane_detector.py (IPM, detección
de amarillo/blanco en HSV, filtro de elongación PCA, 3 bandas, línea guía,
error, separación, yaw/posición vía IMU/odometría) pero:

  - NUNCA publica en /cmd_vel (no mueve el robot, ni falta que lo intente)
  - NO depende de lane_controller — se corre solo
  - Imprime todo en consola para que muevas el robot A MANO y veas qué
    calcula en cada posición/ángulo distinto

Usar para calibrar HSV y verificar visualmente la línea guía sin
arriesgar que el robot se mueva.
"""

import math
import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, Imu
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32
from cv_bridge import CvBridge


def quat_to_yaw(q):
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


class CalibHsvLab(Node):
    def __init__(self):
        super().__init__('calib_hsv_lab')
        self.bridge = CvBridge()

        self.declare_parameters('', [
            ('white_h_min', 0),   ('white_h_max', 180),
            ('white_s_min', 0),   ('white_s_max', 65),
            ('white_v_min', 170), ('white_v_max', 255),
            ('white_min_elong', 5.0), ('white_min_area', 1000), ('white_max_area', 8000),
            ('yellow_h_min', 15), ('yellow_h_max', 45),
            ('yellow_s_min', 45), ('yellow_s_max', 255),
            ('yellow_v_min', 80), ('yellow_v_max', 255),
            ('yellow_min_elong', 8.0), ('yellow_min_area', 500), ('yellow_max_area', 20000),
            ('lane_width_m', 0.22),
            ('px_per_meter', 600.0),
            ('white_bias_m', 0.02),
            ('slope_lookahead_m', 0.03),
            ('slope_scale_m', 0.20),
            ('imu_topic', '/imu'),
            ('odom_topic', '/odom_raw'),
        ])

        gp = self.get_parameter
        self.white_lo = np.array([gp('white_h_min').value, gp('white_s_min').value,
                                   gp('white_v_min').value], dtype=np.uint8)
        self.white_hi = np.array([gp('white_h_max').value, gp('white_s_max').value,
                                   gp('white_v_max').value], dtype=np.uint8)
        self.white_min_elong = float(gp('white_min_elong').value)
        self.white_min_area  = float(gp('white_min_area').value)
        self.white_max_area  = float(gp('white_max_area').value)

        self.yellow_lo = np.array([gp('yellow_h_min').value, gp('yellow_s_min').value,
                                    gp('yellow_v_min').value], dtype=np.uint8)
        self.yellow_hi = np.array([gp('yellow_h_max').value, gp('yellow_s_max').value,
                                    gp('yellow_v_max').value], dtype=np.uint8)
        self.yellow_min_elong = float(gp('yellow_min_elong').value)
        self.yellow_min_area  = float(gp('yellow_min_area').value)
        self.yellow_max_area  = float(gp('yellow_max_area').value)

        self.lane_width_m = float(gp('lane_width_m').value)
        self.px_per_meter = float(gp('px_per_meter').value)
        self.white_bias_m = float(gp('white_bias_m').value)
        self.slope_lookahead_m = float(gp('slope_lookahead_m').value)
        self.slope_scale_m     = float(gp('slope_scale_m').value)

        self.M, self.warp_size = None, None
        self.x_yellow_f = None
        self.x_white_f  = None
        self.x_center_f = None
        self.ema_alpha  = 0.5

        self.yaw = None
        self.pos_x = None
        self.pos_y = None

        self.sub_img  = self.create_subscription(Image, '/image_raw', self.on_image, 10)
        self.sub_imu  = self.create_subscription(Imu, str(gp('imu_topic').value), self.on_imu, 10)
        self.sub_odom = self.create_subscription(Odometry, str(gp('odom_topic').value), self.on_odom, 10)
        self.pub_dbg  = self.create_publisher(Image, '/calib/debug_image', 10)
        self.pub_err  = self.create_publisher(Float32, '/calib/error', 10)
        self.pub_err_yellow = self.create_publisher(Float32, '/calib/error_yellow', 10)

        self.get_logger().info(
            'calib_hsv_lab listo — NO mueve el robot. '
            'Mueve el robot a mano y observa /calib/debug_image y esta consola.')
        self.get_logger().info(
            f'yellow HSV [{self.yellow_lo}]-[{self.yellow_hi}]  '
            f'white HSV [{self.white_lo}]-[{self.white_hi}]')

    # ------------------------------------------------------------------
    def on_imu(self, msg):
        self.yaw = quat_to_yaw(msg.orientation)

    def on_odom(self, msg):
        self.pos_x = msg.pose.pose.position.x
        self.pos_y = msg.pose.pose.position.y

    # ------------------------------------------------------------------
    def build_ipm(self, w, h):
        src = np.float32([
            [0.20 * w, 0.55 * h], [0.80 * w, 0.55 * h],
            [1.00 * w, 0.97 * h], [0.00 * w, 0.97 * h],
        ])
        dst = np.float32([
            [0.25 * w, 0.0], [0.75 * w, 0.0],
            [0.75 * w, h],   [0.25 * w, h],
        ])
        self.M = cv2.getPerspectiveTransform(src, dst)
        self.warp_size = (w, h)

    @staticmethod
    def _component_filter(mask, min_area, max_area, min_elong):
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
        """Pendiente del amarillo usando SOLO los píxeles de la banda dada
        (la inferior) — igual que lane_detector.py, no usa la central.
        Tangente real (dx/dy) por slope_scale_m fijo — ver comentario en
        lane_detector.py sobre por qué no se usa directamente el
        desplazamiento dentro de la banda."""
        ys, xs = np.where(mask_yellow[sl, :] > 0)
        if len(xs) < 20:
            return float('nan')
        pts = np.column_stack((xs, ys)).astype(np.float32)
        vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
        # Piso más alto (no 1e-6) — ver lane_detector.py: línea casi horizontal
        # dentro de la franja hace que tangent explote con poco ruido.
        if abs(vy) < 0.05:
            return float('nan')
        tangent = vx / vy
        # Signo invertido respecto a tangent puro — ver lane_detector.py
        # (convención histórica: x_LEJOS - x_CERCA, no x_CERCA - x_LEJOS)
        slope_m = -tangent * self.slope_scale_m
        max_slope_m = 0.35
        return max(-max_slope_m, min(max_slope_m, slope_m))

    def _ema(self, attr, value):
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

        hsv = cv2.cvtColor(warp, cv2.COLOR_BGR2HSV)
        mask_yellow_raw = cv2.inRange(hsv, self.yellow_lo, self.yellow_hi)
        mask_white_raw  = cv2.inRange(hsv, self.white_lo, self.white_hi)

        open_k  = np.ones((3, 3), np.uint8)
        close_k = np.ones((7, 7), np.uint8)
        mask_yellow_raw = cv2.morphologyEx(mask_yellow_raw, cv2.MORPH_OPEN,  open_k)
        mask_yellow_raw = cv2.morphologyEx(mask_yellow_raw, cv2.MORPH_CLOSE, close_k)
        mask_white_raw  = cv2.morphologyEx(mask_white_raw,  cv2.MORPH_OPEN,  open_k)
        mask_white_raw  = cv2.morphologyEx(mask_white_raw,  cv2.MORPH_CLOSE, close_k)
        mask_white_raw  = cv2.bitwise_and(mask_white_raw, cv2.bitwise_not(mask_yellow_raw))

        mask_yellow = self._component_filter(mask_yellow_raw, self.yellow_min_area,
                                              self.yellow_max_area, self.yellow_min_elong)
        mask_white  = self._component_filter(mask_white_raw, self.white_min_area,
                                              self.white_max_area, self.white_min_elong)

        # 3 bandas — igual que lane_detector.py
        band_rows   = [h // 6, h // 2, (5 * h) // 6]
        band_slices = [slice(0, h // 3), slice(h // 3, (2 * h) // 3), slice((2 * h) // 3, h)]
        lane_width_px = self.lane_width_m * self.px_per_meter
        white_bias_px = self.white_bias_m * self.px_per_meter

        def band_center(sl):
            xy = self._centroid_x(mask_yellow[sl, :])
            xw = self._centroid_x(mask_white[sl, :])
            if xy is not None and xw is not None:
                dist = xw - xy
                if dist <= 0 or dist < lane_width_px * 0.6 or dist > lane_width_px * 1.3:
                    xw = None
            if xy is not None and xw is not None:
                return xy, xw, (xy + xw) / 2.0 - white_bias_px
            elif xy is not None:
                return xy, None, xy + lane_width_px / 2.0
            elif xw is not None:
                return None, xw, xw - lane_width_px / 2.0 - white_bias_px
            return None, None, None

        band_points    = [band_center(sl) for sl in band_slices]
        trajectory_pts = [(c, r) for (_, _, c), r in zip(band_points, band_rows) if c is not None]

        # Amarillo, SOLO dentro de una franja angosta al fondo de la imagen,
        # de tamaño en metros reales (slope_lookahead_m) — igual que
        # lane_detector.py. Más angosta que la banda inferior de error.
        slope_band_px = max(5, int(self.slope_lookahead_m * self.px_per_meter))
        slope_slice = slice(max(0, h - slope_band_px), h)
        slope_m = self._inferior_slope(mask_yellow, slope_slice)

        # La posición real del robot la marca la banda INFERIOR (la más
        # cercana al robot). Superior/central solo se usan para la pendiente.
        x_yellow_raw, x_white_raw, center_raw = band_points[2]

        x_yellow  = self._ema('x_yellow_f', x_yellow_raw)
        x_white   = self._ema('x_white_f',  x_white_raw)
        center_px = self._ema('x_center_f', center_raw)

        error_m = (center_px - w / 2.0) / self.px_per_meter if center_px is not None else float('nan')

        # Error solo-amarillo (ignora blanco) — igual que lane_detector.py.
        error_yellow_m = ((x_yellow + lane_width_px / 2.0) - w / 2.0) / self.px_per_meter \
            if x_yellow is not None else float('nan')

        # Zona de seguridad ANTICIPADA: el margen se agranda según el ángulo
        # (slope_m) — igual que lane_detector.py. Ganancias bajadas (antes
        # 1.8/1.2) y tope de error — evita componer demasiado con PID+FF.
        if center_px is not None:
            safety_margin_px = lane_width_px * 0.30
            angle_px = 0.0 if math.isnan(slope_m) else slope_m * self.px_per_meter
            look_ahead_gain = 0.7
            boost_gain = 1.2
            max_error_m = 0.20

            if x_yellow is not None:
                dist_to_yellow_px = (w / 2.0) - x_yellow
                approaching_yellow = max(0.0, -angle_px)
                margin_yellow = safety_margin_px + approaching_yellow * look_ahead_gain
                if dist_to_yellow_px < margin_yellow:
                    error_m += (margin_yellow - dist_to_yellow_px) / self.px_per_meter * boost_gain
            if x_white is not None:
                dist_to_white_px = x_white - (w / 2.0)
                approaching_white = max(0.0, angle_px)
                margin_white = safety_margin_px + approaching_white * look_ahead_gain
                if dist_to_white_px < margin_white:
                    error_m -= (margin_white - dist_to_white_px) / self.px_per_meter * boost_gain

            error_m = max(-max_error_m, min(max_error_m, error_m))

        out = Float32()
        out.data = float(error_m)
        self.pub_err.publish(out)

        out_y = Float32()
        out_y.data = float(error_yellow_m)
        self.pub_err_yellow.publish(out_y)

        # ---- IMPRESIÓN EN CONSOLA: posición, ángulo, detección, línea guía ----
        yaw_deg = math.degrees(self.yaw) if self.yaw is not None else float('nan')
        pos_txt = (f'x={self.pos_x:+.3f}m y={self.pos_y:+.3f}m'
                   if self.pos_x is not None else 'sin odometría')

        if x_yellow is not None:
            target_cm     = self.lane_width_m * 100.0 / 2.0
            separacion_cm = ((w / 2.0) - x_yellow) / self.px_per_meter * 100.0
            error_sep_cm  = separacion_cm - target_cm
            estado = ('se ACERCA al amarillo' if error_sep_cm < -0.3 else
                      'se ALEJA del amarillo' if error_sep_cm > 0.3 else
                      f'separación correcta ({target_cm:.1f}cm)')
            recto = 'RECTA' if (not math.isnan(slope_m) and abs(slope_m) < 0.015) else 'INCLINADA'
            self.get_logger().info(
                f'[CALIB] Amarillo={x_yellow:.0f}px  Blanco={"sin detectar" if x_white is None else f"{x_white:.0f}px"}  '
                f'separación={separacion_cm:.1f}cm  error={error_sep_cm:+.1f}cm → {estado}  |  '
                f'pendiente={slope_m*100:+.1f}cm ({recto})  |  yaw={yaw_deg:.1f}°  {pos_txt}  '
                f'|  línea_guía_pts={[f"({int(x)},{int(y)})" for x, y in trajectory_pts]}',
                throttle_duration_sec=0.3)
        else:
            self.get_logger().info(
                f'[CALIB] Sin amarillo detectado  |  yaw={yaw_deg:.1f}°  {pos_txt}',
                throttle_duration_sec=0.3)

        self._publish_debug(warp, mask_white, mask_yellow, band_rows,
                            x_white, x_yellow, center_px, msg, trajectory_pts)

    # ------------------------------------------------------------------
    def _publish_debug(self, warp, mask_white, mask_yellow, band_rows,
                       xw, xy, xc, header_msg, trajectory_pts):
        h, w = warp.shape[:2]
        overlay = warp.copy()
        overlay[mask_white  > 0] = (255, 255, 255)
        overlay[mask_yellow > 0] = (0, 255, 255)
        dbg = cv2.addWeighted(overlay, 0.55, warp, 0.45, 0)

        for r in band_rows:
            cv2.line(dbg, (0, r), (w, r), (0, 255, 0), 1)
        cv2.line(dbg, (w // 2, 0), (w // 2, h), (128, 128, 128), 1)

        if trajectory_pts and len(trajectory_pts) >= 2:
            pts = np.array([[int(x), int(y)] for x, y in trajectory_pts], dtype=np.int32)
            cv2.polylines(dbg, [pts], False, (255, 0, 255), 2)
            for x, y in trajectory_pts:
                cv2.circle(dbg, (int(x), int(y)), 4, (255, 0, 255), -1)

        mid_row = band_rows[1]
        for x, color, label in ((xw, (200, 200, 200), 'W'), (xy, (0, 255, 255), 'Y'), (xc, (0, 0, 255), 'C')):
            if x is not None:
                cv2.circle(dbg, (int(x), mid_row), 6, color, -1)
                cv2.putText(dbg, label, (int(x) + 8, mid_row - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        out = self.bridge.cv2_to_imgmsg(dbg, 'bgr8')
        out.header = header_msg.header
        self.pub_dbg.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = CalibHsvLab()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
