#!/usr/bin/env python3
"""Calibración visual HSV/LAB — RC-2.

Hace EXACTAMENTE los mismos cálculos que lane_detector.py (IPM, detección
de amarillo HSV / blanco LAB, 3 bandas, línea guía, error, separación,
yaw/posición vía IMU/odometría) pero:

  - NUNCA publica en /cmd_vel (no mueve el robot, ni falta que lo intente)
  - NO depende de lane_controller — se corre solo
  - Imprime todo en consola para que muevas el robot A MANO y veas qué
    calcula en cada posición/ángulo distinto

Usar para calibrar HSV/LAB y verificar visualmente la línea guía sin
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
            ('white_l_min', 80), ('white_l_max', 255),
            ('white_a_min', 85), ('white_a_max', 170),
            ('white_b_min', 85), ('white_b_max', 170),
            ('white_min_aspect', 1.8), ('white_max_area', 4500),
            ('yellow_h_min', 15), ('yellow_h_max', 40),
            ('yellow_s_min', 60), ('yellow_s_max', 255),
            ('yellow_v_min', 80), ('yellow_v_max', 255),
            ('min_area', 150),
            ('lane_width_m', 0.22),
            ('px_per_meter', 600.0),
            ('imu_topic', '/imu'),
            ('odom_topic', '/odom_raw'),
        ])

        gp = self.get_parameter
        self.white_lo_lab = np.array([gp('white_l_min').value, gp('white_a_min').value,
                                       gp('white_b_min').value], dtype=np.uint8)
        self.white_hi_lab = np.array([gp('white_l_max').value, gp('white_a_max').value,
                                       gp('white_b_max').value], dtype=np.uint8)
        self.white_min_aspect = float(gp('white_min_aspect').value)
        self.white_max_area   = float(gp('white_max_area').value)

        self.yellow_lo = np.array([gp('yellow_h_min').value, gp('yellow_s_min').value,
                                    gp('yellow_v_min').value], dtype=np.uint8)
        self.yellow_hi = np.array([gp('yellow_h_max').value, gp('yellow_s_max').value,
                                    gp('yellow_v_max').value], dtype=np.uint8)

        self.min_area     = float(gp('min_area').value)
        self.lane_width_m = float(gp('lane_width_m').value)
        self.px_per_meter = float(gp('px_per_meter').value)

        self.M, self.warp_size = None, None
        self.x_yellow_f = None
        self.ema_alpha  = 0.5

        self.yaw = None
        self.pos_x = None
        self.pos_y = None

        self.sub_img  = self.create_subscription(Image, '/image_raw', self.on_image, 10)
        self.sub_imu  = self.create_subscription(Imu, str(gp('imu_topic').value), self.on_imu, 10)
        self.sub_odom = self.create_subscription(Odometry, str(gp('odom_topic').value), self.on_odom, 10)
        self.pub_dbg  = self.create_publisher(Image, '/calib/debug_image', 10)
        self.pub_err  = self.create_publisher(Float32, '/calib/error', 10)

        self.get_logger().info(
            'calib_hsv_lab listo — NO mueve el robot. '
            'Mueve el robot a mano y observa /calib/debug_image y esta consola.')
        self.get_logger().info(
            f'yellow HSV [{self.yellow_lo}]-[{self.yellow_hi}]  '
            f'white LAB [{self.white_lo_lab}]-[{self.white_hi_lab}]')

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

    def _filter_line_shape(self, mask):
        result = np.zeros_like(mask)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area or area > self.white_max_area:
                continue
            _, _, cw, ch = cv2.boundingRect(cnt)
            if ch / (cw + 1e-5) >= self.white_min_aspect:
                cv2.drawContours(result, [cnt], -1, 255, -1)
        return result

    def _ema(self, value):
        if value is None:
            self.x_yellow_f = None
            return None
        if self.x_yellow_f is None:
            self.x_yellow_f = value
            return value
        self.x_yellow_f = (1 - self.ema_alpha) * self.x_yellow_f + self.ema_alpha * value
        return self.x_yellow_f

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
        mask_yellow = cv2.inRange(hsv, self.yellow_lo, self.yellow_hi)

        lab = cv2.cvtColor(warp, cv2.COLOR_BGR2LAB)
        mask_white_raw = cv2.inRange(lab, self.white_lo_lab, self.white_hi_lab)

        kernel = np.ones((3, 3), np.uint8)
        mask_yellow    = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, kernel)
        mask_yellow    = cv2.morphologyEx(mask_yellow, cv2.MORPH_CLOSE, kernel)
        mask_white_raw = cv2.morphologyEx(mask_white_raw, cv2.MORPH_OPEN, kernel)
        mask_white_raw = cv2.morphologyEx(mask_white_raw, cv2.MORPH_CLOSE, kernel)
        mask_white_raw = cv2.bitwise_and(mask_white_raw, cv2.bitwise_not(mask_yellow))
        mask_white = self._filter_line_shape(mask_white_raw)

        # 3 bandas — igual que lane_detector.py
        band_rows   = [h // 6, h // 2, (5 * h) // 6]
        band_slices = [slice(0, h // 3), slice(h // 3, (2 * h) // 3), slice((2 * h) // 3, h)]
        lane_width_px = self.lane_width_m * self.px_per_meter

        def band_center(sl):
            xy = self._centroid_x(mask_yellow[sl, :])
            if xy is None:
                return None, None
            return xy, xy + lane_width_px / 2.0

        band_points    = [band_center(sl) for sl in band_slices]
        trajectory_pts = [(c, r) for (_, c), r in zip(band_points, band_rows) if c is not None]

        if len(trajectory_pts) >= 2:
            (x_top, _), (x_bot, _) = trajectory_pts[0], trajectory_pts[-1]
            slope_m = (x_top - x_bot) / self.px_per_meter
        else:
            slope_m = float('nan')

        yellow_vals  = [xy for xy, _ in band_points if xy is not None]
        x_yellow_raw = sum(yellow_vals) / len(yellow_vals) if yellow_vals else None
        x_yellow     = self._ema(x_yellow_raw)
        x_white      = self._centroid_x(mask_white)   # solo informativo

        center_px = x_yellow + lane_width_px / 2.0 if x_yellow is not None else None
        error_m   = (center_px - w / 2.0) / self.px_per_meter if center_px is not None else float('nan')

        out = Float32()
        out.data = float(error_m)
        self.pub_err.publish(out)

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
                f'[CALIB] Amarillo={x_yellow:.0f}px  separación={separacion_cm:.1f}cm  '
                f'error={error_sep_cm:+.1f}cm → {estado}  |  pendiente={slope_m*100:+.1f}cm ({recto})  '
                f'|  yaw={yaw_deg:.1f}°  {pos_txt}  '
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
