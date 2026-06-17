#!/usr/bin/env python3
"""CapyTown lap_plotter - RC-2.

Script standalone e independiente: NO modifica ni depende de la lógica interna
de lane_detector.py / lane_controller.py, y NUNCA publica /cmd_vel (no mueve
el robot, no afecta el control). Solo escucha y dibuja.

Hace su PROPIA detección de amarillo/blanco a partir de /image_raw (mismo
pipeline que calib_hsv_lab.py: IPM + HSV + morfología + filtro PCA + banda
inferior) únicamente para poder graficar esos puntos — no usa los topics
internos del detector para eso.

Además escucha:
  /lane_error  → error lateral (cm) en el tiempo, y señal de INICIO del
                 registro (primer valor válido, no NaN)
  /cmd_vel     → velocidad lineal y angular comandada en el tiempo
  /odom_raw    → trayectoria x,y del carrito

Para saber cuándo se completaron las 3 vueltas, replica — SOLO COMO LECTOR,
sin tocar ni publicar nada — el mismo criterio que usa lane_controller.py
para entrar/salir de un giro cerrado de esquina (mismos umbrales que
config/pid_params.yaml): magnitud de /lane_slope o tiempo anticipando, y
salida cuando vuelve a estar recto y centrado. La pista tiene 3 vueltas de
4 esquinas cada una, así que 12 giros cerrados completados = 3 vueltas.
Tras el giro 12, se sigue registrando 10cm más de avance (odometría) antes
de terminar, para no cortar el gráfico justo en la última esquina.

TIEMPO REAL: cada `plot_interval_s` segundos (1.5s por defecto) se
sobreescriben los 3 PNG con los datos acumulados hasta ese momento — si el
carrito se frena, se mata el proceso, o nunca llega a completar las 3
vueltas, en el disco ya van a estar las imágenes con el progreso hecho
hasta ese instante, no solo al final.

Genera 3 imágenes en `output_dir` (parámetro ROS, '/root' por defecto):
  velocidades_errores.png → velocidad lineal/angular y error lateral en el tiempo
  deteccion_colores.png   → posición de amarillo y blanco detectados en el tiempo
  recorrido.png           → trayectoria x,y del carrito

Uso:
  python3 lap_plotter.py
  python3 lap_plotter.py --ros-args -p output_dir:=/root/graficos

Requiere matplotlib (pip install matplotlib si falta).
"""
import math
import time

import cv2
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge


class LapPlotter(Node):

    def __init__(self):
        super().__init__('lap_plotter')

        self.declare_parameters('', [
            ('image_topic', '/image_raw'),
            ('odom_topic',  '/odom_raw'),
            ('output_dir',  '/root'),
            ('plot_interval_s', 1.5),
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
        self.plot_interval_s  = float(gp('plot_interval_s').value)
        self.target_turns     = int(gp('target_turns').value)
        self.extra_distance_m = float(gp('extra_distance_m').value)

        self.slope_curve_threshold      = float(gp('slope_curve_threshold').value)
        self.sharp_turn_slope_threshold = float(gp('sharp_turn_slope_threshold').value)
        self.calib_tolerance            = float(gp('calib_tolerance').value)
        self.max_anticipation_time      = float(gp('max_anticipation_time').value)
        self.error_timeout              = float(gp('error_timeout').value)

        self.lane_width_m = float(gp('lane_width_m').value)
        self.px_per_meter = float(gp('px_per_meter').value)

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

        # ---- Series registradas ----
        self.traj_x, self.traj_y = [], []
        self.vel_t, self.vel_lin, self.vel_ang = [], [], []
        self.err_t,  self.err_v  = [], []
        self.det_t_y, self.det_y = [], []
        self.det_t_w, self.det_w = [], []

        self.sub_img  = self.create_subscription(Image, image_topic, self.on_image, 10)
        self.sub_err  = self.create_subscription(Float32, '/lane_error', self.on_error, 10)
        self.sub_slope = self.create_subscription(Float32, '/lane_slope', self.on_slope, 10)
        self.sub_cmd  = self.create_subscription(Twist, '/cmd_vel', self.on_cmd_vel, 10)
        self.sub_odom = self.create_subscription(Odometry, odom_topic, self.on_odom, 10)

        # Tick de lógica de giro (replica de lectura) a 30Hz
        self.turn_timer = self.create_timer(1.0 / 30.0, self._turn_logic_tick)
        # Guardado de gráficos en tiempo real
        self.plot_timer = self.create_timer(self.plot_interval_s, self._save_all_plots)

        self.get_logger().info(
            f'lap_plotter listo — esperando color para iniciar. '
            f'Objetivo: {self.target_turns} giros cerrados + {self.extra_distance_m*100:.0f}cm. '
            f'Gráficos en {self.output_dir}/ cada {self.plot_interval_s}s.')

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
        self.err_t.append(self._elapsed())
        self.err_v.append(msg.data * 100.0)   # cm

    def on_slope(self, msg):
        if not math.isnan(msg.data):
            self.slope = msg.data

    def on_cmd_vel(self, msg):
        if not self.started or self.done:
            return
        self.vel_t.append(self._elapsed())
        self.vel_lin.append(msg.linear.x)
        self.vel_ang.append(msg.angular.z)

    def on_odom(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        self.last_x, self.last_y = x, y

        if self.done or not self.started:
            return
        if self.x0 is None:
            self.x0, self.y0 = x, y

        self.traj_x.append(x)
        self.traj_y.append(y)

        if self.finishing:
            dist = math.hypot(x - self.finish_trigger_x, y - self.finish_trigger_y)
            if dist >= self.extra_distance_m:
                self._finish(x, y)

    # ------------------------------------------------------------------
    def _turn_logic_tick(self):
        """Replica de SOLO LECTURA del criterio de giro cerrado de
        lane_controller.py — no publica nada, no decide nada del control
        real, solo cuenta para saber cuándo van 3 vueltas (12 giros)."""
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

        if self.in_sharp_turn:
            if abs(self.slope) < self.slope_curve_threshold and abs(e) < self.calib_tolerance:
                self.in_sharp_turn = False
                self.turns_done += 1
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
            self.det_t_y.append(t)
            self.det_y.append((xy - w / 2.0) / self.px_per_meter * 100.0)   # cm
        if xw is not None:
            self.det_t_w.append(t)
            self.det_w.append((xw - w / 2.0) / self.px_per_meter * 100.0)   # cm

    # ------------------------------------------------------------------
    def _finish(self, xf, yf):
        self.done = True
        dist_final = math.hypot(xf - self.x0, yf - self.y0)
        self.get_logger().info(
            f'¡Registro terminado! {self.turns_done} giros cerrados completados. '
            f'Inicial=({self.x0:.3f}, {self.y0:.3f})  Final=({xf:.3f}, {yf:.3f})  '
            f'distancia inicio↔fin={dist_final * 100:.1f}cm')
        self._save_all_plots()
        self.get_logger().info(f'Gráficos finales guardados en {self.output_dir}/')

    def _save_all_plots(self):
        if not self.started:
            return
        self._plot_velocidades_errores()
        self._plot_deteccion()
        self._plot_recorrido()

    def _plot_velocidades_errores(self):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

        ax1.plot(self.vel_t, self.vel_lin, '-b', label='Velocidad lineal (m/s)')
        ax1.plot(self.vel_t, self.vel_ang, '-g', label='Velocidad angular (rad/s)')
        ax1.axhline(0, color='black', linewidth=0.8)
        ax1.set_ylabel('velocidad')
        ax1.set_title(f'Velocidades y error — {self.turns_done}/{self.target_turns} giros cerrados')
        ax1.legend()
        ax1.grid(True)

        ax2.plot(self.err_t, self.err_v, '-r', label='Error lateral (cm)')
        ax2.axhline(0, color='black', linewidth=0.8)
        ax2.set_xlabel('tiempo (s)')
        ax2.set_ylabel('error (cm)')
        ax2.legend()
        ax2.grid(True)

        fig.tight_layout()
        fig.savefig(f'{self.output_dir}/velocidades_errores.png')
        plt.close(fig)

    def _plot_deteccion(self):
        fig = plt.figure(figsize=(10, 4))
        plt.plot(self.det_t_y, self.det_y, '-', color='gold', label='Amarillo (cm desde el centro)')
        plt.plot(self.det_t_w, self.det_w, '-', color='gray', label='Blanco (cm desde el centro)')
        plt.axhline(0, color='black', linewidth=0.8)
        plt.xlabel('tiempo (s)')
        plt.ylabel('posición (cm)')
        plt.title('Detección de carril — amarillo y blanco')
        plt.legend()
        plt.grid(True)
        fig.tight_layout()
        fig.savefig(f'{self.output_dir}/deteccion_colores.png')
        plt.close(fig)

    def _plot_recorrido(self):
        fig = plt.figure(figsize=(6, 6))
        plt.plot(self.traj_x, self.traj_y, '-b', linewidth=1.5, label='Recorrido')
        if self.x0 is not None:
            plt.plot(self.x0, self.y0, 'go', markersize=10, label='Inicio')
        if self.traj_x:
            plt.plot(self.traj_x[-1], self.traj_y[-1], 'rx', markersize=10, markeredgewidth=2, label='Actual/Fin')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.title(f'Recorrido del carrito — {self.turns_done}/{self.target_turns} giros cerrados')
        plt.axis('equal')
        plt.legend()
        plt.grid(True)
        fig.tight_layout()
        fig.savefig(f'{self.output_dir}/recorrido.png')
        plt.close(fig)


def main(args=None):
    rclpy.init(args=args)
    node = LapPlotter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._save_all_plots()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
