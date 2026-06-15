#!/usr/bin/env python3
"""CapyTown lane_controller - RC-2.

Estados:
  ESPERA      : espera 5s tras primera detección
  CALIB_BUSCA : sin amarillo → giro suave izquierda hasta encontrarlo
  CALIB_CENTRA: amarillo visible pero error grande → corrección suave sin avanzar
  AVANCE      : centrado y amarillo estable → avanza + corrección suave

Convención:  error > 0 → robot desplazado a la derecha → ω < 0 (girar derecha)
             error < 0 → robot desplazado a la izquierda → ω > 0 (girar izquierda)
"""

import math
from collections import deque

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from geometry_msgs.msg import Twist


class LaneController(Node):

    def __init__(self):
        super().__init__('lane_controller')

        self.declare_parameters('', [
            # PID — solo activo en modo AVANCE
            ('kp',              4.0),
            ('ki',              0.3),
            ('kd',              0.4),
            ('kff',             1.0),
            # Velocidades
            ('linear_speed',    0.14),   # m/s en modo AVANCE
            ('max_angular',     2.0),
            # Calibración
            ('calib_w',         0.30),   # rad/s — corrección suave en calibración
            ('error_tolerance', 0.03),   # m — umbral para considerar "centrado"
            ('stable_frames',   6),      # frames consecutivos con amarillo antes de avanzar
            # Control
            ('integral_limit',  0.5),
            ('error_timeout',   0.8),    # s — tiempo sin amarillo → modo búsqueda
            ('control_rate',    30.0),
            ('turn_threshold',  0.3),
            ('history_size',    10),
            ('start_delay',     5.0),
        ])

        gp = self.get_parameter
        self.kp              = float(gp('kp').value)
        self.ki              = float(gp('ki').value)
        self.kd              = float(gp('kd').value)
        self.kff             = float(gp('kff').value)
        self.v               = float(gp('linear_speed').value)
        self.max_w           = float(gp('max_angular').value)
        self.calib_w         = float(gp('calib_w').value)
        self.error_tolerance = float(gp('error_tolerance').value)
        self.stable_frames   = int(gp('stable_frames').value)
        self.i_limit         = float(gp('integral_limit').value)
        self.timeout         = float(gp('error_timeout').value)
        self.turn_threshold  = float(gp('turn_threshold').value)
        self.start_delay     = float(gp('start_delay').value)
        hist                 = int(gp('history_size').value)
        rate                 = float(gp('control_rate').value)

        self.error         = None
        self.last_error    = 0.0
        self.last_w        = 0.0
        self.smooth_w      = 0.0
        self.integral      = 0.0
        self.initialized   = False
        self.start_time    = None
        self.last_stamp    = self.get_clock().now()
        self.last_rx       = self.get_clock().now()
        self.error_history = deque(maxlen=hist)
        self.yellow_streak = 0   # frames consecutivos con amarillo válido

        self.sub   = self.create_subscription(Float32, '/lane_error', self.on_error, 10)
        self.pub   = self.create_publisher(Twist, '/cmd_vel', 10)
        self.timer = self.create_timer(1.0 / rate, self.control_loop)

        self.get_logger().info(
            f'lane_controller listo — v={self.v} calib_w={self.calib_w} '
            f'tol={self.error_tolerance}m estable={self.stable_frames}f')

    # ------------------------------------------------------------------
    def on_error(self, msg):
        if not math.isnan(msg.data):
            self.error       = msg.data
            self.last_rx     = self.get_clock().now()
            self.yellow_streak += 1
            if not self.initialized:
                self.initialized = True
                self.start_time  = self.get_clock().now()
                self.get_logger().info(f'Amarillo detectado — esperando {self.start_delay:.0f}s...')
        else:
            self.yellow_streak = 0   # perdió amarillo, reinicia contador

    # ------------------------------------------------------------------
    def _trend(self):
        n = len(self.error_history)
        if n < 3:
            return 0.0
        lst = list(self.error_history)
        return (lst[-1] - lst[0]) / n

    # ------------------------------------------------------------------
    def _smooth(self, target, alpha=0.25):
        """Suavizado exponencial — alpha bajo = cambio muy lento."""
        self.smooth_w = (1.0 - alpha) * self.smooth_w + alpha * target
        return self.smooth_w

    # ------------------------------------------------------------------
    def control_loop(self):
        now = self.get_clock().now()
        dt  = (now - self.last_stamp).nanoseconds * 1e-9
        self.last_stamp = now

        if dt <= 0.0:
            return

        # ── ESPERA INICIAL ────────────────────────────────────────────
        if not self.initialized:
            self.pub.publish(Twist())
            return
        if (now - self.start_time).nanoseconds * 1e-9 < self.start_delay:
            self.pub.publish(Twist())
            return

        age = (now - self.last_rx).nanoseconds * 1e-9
        cmd = Twist()

        # ── CALIB_BUSCA: sin amarillo ─────────────────────────────────
        if age > self.timeout:
            self.integral = 0.0
            self.error_history.clear()
            # Giro suave izquierda — el amarillo está a la izquierda
            w_target = +self.calib_w
            cmd.linear.x  = 0.0
            cmd.angular.z = self._smooth(w_target, alpha=0.15)
            self.pub.publish(cmd)
            self.get_logger().debug('[BUSCA] Sin amarillo — giro suave izquierda')
            self.last_w = cmd.angular.z
            return

        e = self.error
        self.error_history.append(e)

        centrado = abs(e) < self.error_tolerance
        estable  = self.yellow_streak >= self.stable_frames

        # ── CALIB_CENTRA: amarillo visible pero fuera de tolerancia ───
        if not centrado:
            # Corrección proporcional suave, sin avanzar
            # Ganancia reducida (kp/4) para que sea lenta y progresiva
            w_target = -(self.kp / 4.0) * e
            w_target = max(-self.calib_w, min(self.calib_w, w_target))  # limitar a calib_w
            cmd.linear.x  = 0.0
            cmd.angular.z = self._smooth(w_target, alpha=0.20)
            self.pub.publish(cmd)
            self.last_w    = cmd.angular.z
            self.last_error = e
            self.get_logger().debug(f'[CENTRA] e={e:.3f}m w={cmd.angular.z:.2f}')
            return

        # ── AVANCE: centrado + amarillo estable ───────────────────────
        if not estable:
            # Centrado pero aún esperando frames consecutivos (esquinas)
            w_target = -(self.kp / 4.0) * e
            w_target = max(-self.calib_w, min(self.calib_w, w_target))
            cmd.linear.x  = 0.0
            cmd.angular.z = self._smooth(w_target, alpha=0.20)
            self.pub.publish(cmd)
            self.last_w     = cmd.angular.z
            self.last_error = e
            self.get_logger().debug(f'[ESPERA_ESTABLE] streak={self.yellow_streak}/{self.stable_frames}')
            return

        # PID completo solo en modo avance
        P = self.kp * e
        self.integral += e * dt
        self.integral  = max(-self.i_limit, min(self.i_limit, self.integral))
        I  = self.ki * self.integral
        D  = self.kd * (e - self.last_error) / dt
        FF = self.kff * self._trend() if abs(self.last_w) > self.turn_threshold else 0.0

        w_pid = -(P + I + D + FF)
        w_pid = max(-self.max_w, min(self.max_w, w_pid))

        # Velocidad 10% reducida para correcciones más suaves mientras avanza
        cmd.linear.x  = self.v * 0.9
        cmd.angular.z = self._smooth(w_pid, alpha=0.30)
        self.pub.publish(cmd)

        self.last_error = e
        self.last_w     = cmd.angular.z


def main(args=None):
    rclpy.init(args=args)
    node = LaneController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.pub.publish(Twist())
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
