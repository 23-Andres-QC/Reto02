#!/usr/bin/env python3
"""CapyTown lap_logger - RC-2.

Reemplaza a lap_plotter.py: en vez de graficar en vivo, este nodo SOLO
REGISTRA datos en archivos CSV (uno por serie). No depende de matplotlib,
así que es más liviano y robusto — y separa el registro (en vivo, mientras
el carrito corre) del graficado (después, con plot_lap_logs.py, cuantas
veces quieras, sin volver a correr el robot).

Standalone e independiente: NO modifica ni depende de la lógica interna de
lane_detector.py / lane_controller.py, y NUNCA publica /cmd_vel (no mueve el
robot, no afecta el control). Solo escucha y escribe a disco.

Cada fila se escribe y se hace flush() inmediatamente — si se corta con
Ctrl+C o se cae el proceso, todo lo ya recibido queda guardado en disco,
no se pierde por no haber llegado al final.

Archivos generados en `output_dir` (parámetro ROS, '/root' por defecto):
  log_trayectoria.csv        t,x,y                     (de /odom_raw)
  log_velocidades.csv        t,linear,angular           (de /cmd_vel)
  log_error.csv              t,error_cm                 (de /lane_error)
  log_deteccion_amarillo.csv t,pos_cm                    (detección propia, banda inferior)
  log_deteccion_blanco.csv   t,pos_cm                    (detección propia, banda inferior)
  log_slope.csv              t,slope_cm,in_sharp_turn,anticipation_timer
                              — para diagnosticar a qué distancia/tiempo real
                              se dispara el giro, y por qué algunas esquinas
                              no llegan a contarse como giro cerrado
  log_giros.csv              giro_num,t,slope_cm,e_cm   (cada vez que se
                              completa un giro cerrado)

Igual que antes, para saber cuándo van las 3 vueltas (12 giros + 10cm)
replica — SOLO COMO LECTOR, sin tocar ni publicar nada — el mismo criterio
que usa lane_controller.py para entrar/salir de un giro cerrado de esquina
(mismos umbrales que config/pid_params.yaml).

Uso:
  python3 lap_logger.py
  python3 lap_logger.py --ros-args -p output_dir:=/root/logs

Después, para graficar:
  python3 plot_lap_logs.py
"""
import csv
import math
import os
import time

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge


class LapLogger(Node):

    def __init__(self):
        super().__init__('lap_logger')

        self.declare_parameters('', [
            ('image_topic', '/image_raw'),
            ('odom_topic',  '/odom_raw'),
            ('output_dir',  '/root'),
            ('target_turns', 12),     # 3 vueltas x 4 esquinas
            ('extra_distance_m', 0.10),
            # Mismos umbrales que config/pid_params.yaml (lane_controller.py) —
            # se replican aquí SOLO para contar giros, de solo lectura, sin
            # tocar ni publicar nada en el control real.
            ('slope_curve_threshold', 0.04),
            ('sharp_turn_slope_threshold', 0.13),
            ('calib_tolerance', 0.025),
            ('max_anticipation_time', 0.8),
            ('error_timeout', 0.5),
            # Detección HSV — mismos valores que hsv_params.yaml (lane_detector.py)
            ('lane_width_m', 0.22),
            ('px_per_meter', 600.0),
        ])
        gp = self.get_parameter
        image_topic = str(gp('image_topic').value)
        odom_topic  = str(gp('odom_topic').value)
        self.output_dir       = str(gp('output_dir').value)
        self.target_turns     = int(gp('target_turns').value)
        self.extra_distance_m = float(gp('extra_distance_m').value)

        self.slope_curve_threshold      = float(gp('slope_curve_threshold').value)
        self.sharp_turn_slope_threshold = float(gp('sharp_turn_slope_threshold').value)
        self.calib_tolerance            = float(gp('calib_tolerance').value)
        self.max_anticipation_time      = float(gp('max_anticipation_time').value)
        self.error_timeout              = float(gp('error_timeout').value)

        self.lane_width_m = float(gp('lane_width_m').value)
        self.px_per_meter = float(gp('px_per_meter').value)

        os.makedirs(self.output_dir, exist_ok=True)

        # HSV — mismos rangos que lane_detector.py / calib_hsv_lab.py
        self.white_lo = np.array([0, 0, 170], dtype=np.uint8)
        self.white_hi = np.array([180, 65, 255], dtype=np.uint8)
        self.yellow_lo = np.array([15, 45, 80], dtype=np.uint8)
        self.yellow_hi = np.array([45, 255, 255], dtype=np.uint8)
        self.white_min_area, self.white_max_area, self.white_min_elong   = 1000, 8000, 5.0
        self.yellow_min_area, self.yellow_max_area, self.yellow_min_elong = 500, 20000, 8.0

        self.bridge = CvBridge()
        self.M = None
        self.warp_size = None

        # ---- Estado de inicio/fin del registro ----
        self.started = False
        self.done    = False
        self.t0      = None

        self.x0 = None
        self.y0 = None
        self.last_x = None
        self.last_y = None

        self.error  = None   # último /lane_error válido (m)
        self.error_yellow = None   # último /lane_error_yellow válido (m) — usado al girar
        self.slope  = 0.0    # último /lane_slope válido (m)
        self.last_err_rx = None

        # ---- Replica de solo lectura de la lógica de giro de esquina ----
        self.in_sharp_turn      = False
        self.anticipation_timer = 0.0
        self.turns_done         = 0
        self._last_tick         = None

        self.finishing = False
        self.finish_trigger_x = None
        self.finish_trigger_y = None

        # ---- Archivos CSV (abiertos en modo texto, flush por fila) ----
        self._f_traj  = open(f'{self.output_dir}/log_trayectoria.csv', 'w', newline='')
        self._f_vel   = open(f'{self.output_dir}/log_velocidades.csv', 'w', newline='')
        self._f_err   = open(f'{self.output_dir}/log_error.csv', 'w', newline='')
        self._f_det_y = open(f'{self.output_dir}/log_deteccion_amarillo.csv', 'w', newline='')
        self._f_det_w = open(f'{self.output_dir}/log_deteccion_blanco.csv', 'w', newline='')
        self._f_slope = open(f'{self.output_dir}/log_slope.csv', 'w', newline='')
        self._f_giros = open(f'{self.output_dir}/log_giros.csv', 'w', newline='')

        self._w_traj  = csv.writer(self._f_traj);  self._w_traj.writerow(['t', 'x', 'y'])
        self._w_vel   = csv.writer(self._f_vel);   self._w_vel.writerow(['t', 'linear', 'angular'])
        self._w_err   = csv.writer(self._f_err);   self._w_err.writerow(['t', 'error_cm'])
        self._w_det_y = csv.writer(self._f_det_y); self._w_det_y.writerow(['t', 'pos_cm'])
        self._w_det_w = csv.writer(self._f_det_w); self._w_det_w.writerow(['t', 'pos_cm'])
        self._w_slope = csv.writer(self._f_slope); self._w_slope.writerow(['t', 'slope_cm', 'in_sharp_turn', 'anticipation_timer'])
        self._w_giros = csv.writer(self._f_giros); self._w_giros.writerow(['giro_num', 't', 'slope_cm', 'e_cm'])

        self.sub_img  = self.create_subscription(Image, image_topic, self.on_image, 10)
        self.sub_err  = self.create_subscription(Float32, '/lane_error', self.on_error, 10)
        self.sub_err_yellow = self.create_subscription(Float32, '/lane_error_yellow', self.on_error_yellow, 10)
        self.sub_slope = self.create_subscription(Float32, '/lane_slope', self.on_slope, 10)
        self.sub_cmd  = self.create_subscription(Twist, '/cmd_vel', self.on_cmd_vel, 10)
        self.sub_odom = self.create_subscription(Odometry, odom_topic, self.on_odom, 10)

        # Tick de lógica de giro (replica de lectura) a 30Hz — también
        # registra slope/estado en cada tick para diagnóstico fino.
        self.turn_timer = self.create_timer(1.0 / 30.0, self._turn_logic_tick)

        self.get_logger().info(
            f'lap_logger listo — esperando color para iniciar. '
            f'Objetivo: {self.target_turns} giros cerrados + {self.extra_distance_m*100:.0f}cm. '
            f'CSVs en {self.output_dir}/')

    # ------------------------------------------------------------------
    def _elapsed(self):
        return time.time() - self.t0 if self.t0 is not None else 0.0

    def _maybe_start(self):
        if self.started:
            return
        self.started = True
        self.t0 = time.time()
        self._last_tick = self.t0
        if self.last_x is not None:
            self.x0, self.y0 = self.last_x, self.last_y
        self.get_logger().info('Color detectado — inicia el registro.')

    # ------------------------------------------------------------------
    def on_error(self, msg):
        if math.isnan(msg.data):
            return
        self._maybe_start()
        if self.done:
            return
        self.error = msg.data
        self.last_err_rx = time.time()
        self._w_err.writerow([f'{self._elapsed():.3f}', f'{msg.data*100.0:.2f}'])
        self._f_err.flush()

    def on_error_yellow(self, msg):
        if not math.isnan(msg.data):
            self.error_yellow = msg.data

    def on_slope(self, msg):
        if not math.isnan(msg.data):
            self.slope = msg.data

    def on_cmd_vel(self, msg):
        if not self.started or self.done:
            return
        self._w_vel.writerow([f'{self._elapsed():.3f}', f'{msg.linear.x:.4f}', f'{msg.angular.z:.4f}'])
        self._f_vel.flush()

    def on_odom(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        self.last_x, self.last_y = x, y

        if self.done or not self.started:
            return
        if self.x0 is None:
            self.x0, self.y0 = x, y

        self._w_traj.writerow([f'{self._elapsed():.3f}', f'{x:.4f}', f'{y:.4f}'])
        self._f_traj.flush()

        if self.finishing:
            dist = math.hypot(x - self.finish_trigger_x, y - self.finish_trigger_y)
            if dist >= self.extra_distance_m:
                self._finish()

    # ------------------------------------------------------------------
    def _turn_logic_tick(self):
        """Replica de SOLO LECTURA del criterio de giro cerrado de
        lane_controller.py — no publica nada, no decide nada del control
        real, solo cuenta y registra para diagnosticar."""
        if not self.started or self.done:
            return
        now = time.time()
        dt = now - self._last_tick
        self._last_tick = now
        if dt <= 0.0 or self.error is None:
            return

        # Sin color reciente: igual que el controlador, no acumula anticipación
        if self.last_err_rx is not None and (now - self.last_err_rx) > self.error_timeout:
            self.anticipation_timer = 0.0
            self.in_sharp_turn = False
            return

        e = self.error
        if abs(e) < 0.01:
            e = 0.0

        anticipating_now = abs(self.slope) > self.slope_curve_threshold
        if anticipating_now:
            self.anticipation_timer += dt
        else:
            self.anticipation_timer = 0.0

        if (abs(self.slope) > self.sharp_turn_slope_threshold
                or self.anticipation_timer > self.max_anticipation_time):
            self.in_sharp_turn = True
            self.anticipation_timer = 0.0

        self._w_slope.writerow([
            f'{self._elapsed():.3f}', f'{self.slope*100.0:.2f}',
            int(self.in_sharp_turn), f'{self.anticipation_timer:.3f}'])
        self._f_slope.flush()

        if self.in_sharp_turn:
            # Igual que lane_controller.py: mientras gira, la condición de
            # salida usa el error SOLO-AMARILLO (ignora blanco), no el
            # combinado — un blanco de otro tramo de pista puede corromper
            # el error combinado durante el giro. Pero para CONFIRMAR la
            # salida se exige ADEMÁS que el error combinado (self.error)
            # también esté centrado — asegura que el blanco de la pista
            # nueva ya esté del lado correcto y a la separación esperada,
            # no solo que el amarillo se vea recto.
            e_turn = self.error_yellow if self.error_yellow is not None else e
            if abs(e_turn) < 0.01:
                e_turn = 0.0
            yellow_ok   = abs(self.slope) < self.slope_curve_threshold and abs(e_turn) < self.calib_tolerance
            combined_ok = abs(self.error) < self.calib_tolerance
            if yellow_ok and combined_ok:
                self.in_sharp_turn = False
                self.turns_done += 1
                self._w_giros.writerow([
                    self.turns_done, f'{self._elapsed():.3f}',
                    f'{self.slope*100.0:.2f}', f'{e_turn*100.0:.2f}'])
                self._f_giros.flush()
                self.get_logger().info(f'Giro cerrado {self.turns_done}/{self.target_turns} completado')
                if self.turns_done >= self.target_turns and not self.finishing:
                    self.finishing = True
                    self.finish_trigger_x = self.last_x
                    self.finish_trigger_y = self.last_y
                    self.get_logger().info(
                        f'Objetivo de giros alcanzado — avanzando '
                        f'{self.extra_distance_m*100:.0f}cm más antes de terminar...')

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

    @staticmethod
    def _centroid_x(mask):
        m = cv2.moments(mask, binaryImage=True)
        if m['m00'] < 1e-3:
            return None
        return m['m10'] / m['m00']

    def on_image(self, msg):
        if not self.started or self.done:
            return
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
        mask_white  = cv2.inRange(hsv, self.white_lo, self.white_hi)

        open_k  = np.ones((3, 3), np.uint8)
        close_k = np.ones((7, 7), np.uint8)
        mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, open_k)
        mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_CLOSE, close_k)
        mask_white  = cv2.morphologyEx(mask_white, cv2.MORPH_OPEN, open_k)
        mask_white  = cv2.morphologyEx(mask_white, cv2.MORPH_CLOSE, close_k)
        mask_white  = cv2.bitwise_and(mask_white, cv2.bitwise_not(mask_yellow))

        mask_yellow = self._component_filter(mask_yellow, self.yellow_min_area,
                                              self.yellow_max_area, self.yellow_min_elong)
        mask_white  = self._component_filter(mask_white, self.white_min_area,
                                              self.white_max_area, self.white_min_elong)

        # Banda inferior únicamente — la más cercana al robot.
        bottom = slice((2 * h) // 3, h)
        xy = self._centroid_x(mask_yellow[bottom, :])
        xw = self._centroid_x(mask_white[bottom, :])

        t = self._elapsed()
        if xy is not None:
            self._w_det_y.writerow([f'{t:.3f}', f'{(xy - w/2.0)/self.px_per_meter*100.0:.2f}'])
            self._f_det_y.flush()
        if xw is not None:
            self._w_det_w.writerow([f'{t:.3f}', f'{(xw - w/2.0)/self.px_per_meter*100.0:.2f}'])
            self._f_det_w.flush()

    # ------------------------------------------------------------------
    def _finish(self):
        self.done = True
        self.get_logger().info(
            f'¡Registro terminado! {self.turns_done} giros cerrados completados. '
            f'CSVs guardados en {self.output_dir}/ — corre plot_lap_logs.py para graficar.')
        self._close_files()

    def _close_files(self):
        for f in (self._f_traj, self._f_vel, self._f_err,
                  self._f_det_y, self._f_det_w, self._f_slope, self._f_giros):
            try:
                f.close()
            except Exception:
                pass


def main(args=None):
    rclpy.init(args=args)
    node = LapLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._close_files()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
