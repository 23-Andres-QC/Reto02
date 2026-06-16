#!/usr/bin/env python3
"""CapyTown lane_controller - RC-2.

Estados:
  ESPERA      : espera 5s tras primera detección
  CALIB_BUSCA : sin amarillo → giro suave izquierda hasta encontrarlo
  CALIB_CENTRA: amarillo visible pero error grande → corrección suave sin avanzar
  AVANCE      : centrado y amarillo estable → avanza + corrección suave

Detección de esquina: si el ángulo girado acumulado en una misma dirección
supera 90° (π/2 rad), se considera que se completó una curva y se resetea
el contador de estabilidad para re-calibrar antes de volver a avanzar.

Convención:  error > 0 → robot desplazado a la derecha → ω < 0 (girar derecha)
             error < 0 → robot desplazado a la izquierda → ω > 0 (girar izquierda)
"""

import math
from collections import deque

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from geometry_msgs.msg import Twist

CORNER_THRESHOLD = math.pi / 2.0   # 90 grados en radianes


class LaneController(Node):

    def __init__(self):
        super().__init__('lane_controller')

        self.declare_parameters('', [
            ('kp',              4.0),
            ('ki',              0.3),
            ('kd',              0.4),
            ('kff',             1.0),
            ('linear_speed',    0.21),   # +50% sobre 0.14
            ('max_angular',     2.0),
            ('calib_w',         0.36),   # +10% sobre 0.33 rad/s
            ('error_tolerance', 0.03),
            ('stable_frames',   6),
            ('integral_limit',  0.5),
            ('error_timeout',   0.8),
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
        self.yellow_streak = 0

        # Acumulador de ángulo para detección de esquina
        self.cum_angle      = 0.0   # rad girados en la dirección actual
        self.last_turn_sign = 0     # +1 izq, -1 der, 0 recto

        self.sub   = self.create_subscription(Float32, '/lane_error', self.on_error, 10)
        self.pub   = self.create_publisher(Twist, '/cmd_vel', 10)
        self.timer = self.create_timer(1.0 / rate, self.control_loop)

        self.get_logger().info(
            f'lane_controller — v={self.v:.3f} calib_w={self.calib_w:.2f} '
            f'tol={self.error_tolerance}m stable={self.stable_frames}f')

    # ------------------------------------------------------------------
    def on_error(self, msg):
        if not math.isnan(msg.data):
            self.error         = msg.data
            self.last_rx       = self.get_clock().now()
            self.yellow_streak += 1
            if not self.initialized:
                self.initialized = True
                self.start_time  = self.get_clock().now()
                self.get_logger().info(f'Amarillo detectado — esperando {self.start_delay:.0f}s...')
        else:
            self.yellow_streak = 0

    # ------------------------------------------------------------------
    def _trend(self):
        n = len(self.error_history)
        if n < 3:
            return 0.0
        lst = list(self.error_history)
        return (lst[-1] - lst[0]) / n

    def _smooth(self, target, alpha=0.25):
        self.smooth_w = (1.0 - alpha) * self.smooth_w + alpha * target
        return self.smooth_w

    def _track_corner(self, w, dt):
        """Acumula ángulo girado en la misma dirección.
        Si cambia de dirección, resetea. Si supera 90°, detecta esquina."""
        if abs(w) < 0.05:
            # Casi recto: resetea acumulador
            self.cum_angle      = 0.0
            self.last_turn_sign = 0
            return False

        sign = 1 if w > 0 else -1
        if sign != self.last_turn_sign:
            # Cambió de dirección: resetea
            self.cum_angle      = 0.0
            self.last_turn_sign = sign

        self.cum_angle += abs(w) * dt

        if self.cum_angle >= CORNER_THRESHOLD:
            self.cum_angle      = 0.0
            self.last_turn_sign = 0
            return True   # esquina completada

        return False

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
            w_target = +self.calib_w   # gira izquierda suave — amarillo está a la izquierda
            cmd.linear.x  = 0.0
            cmd.angular.z = self._smooth(w_target, alpha=0.15)
            self.pub.publish(cmd)
            self._track_corner(cmd.angular.z, dt)
            self.last_w = cmd.angular.z
            return

        e = self.error
        self.error_history.append(e)

        # ── AVANCE: amarillo visible → siempre avanza, PID corrige ──────
        P = self.kp * e
        self.integral += e * dt
        self.integral  = max(-self.i_limit, min(self.i_limit, self.integral))
        I  = self.ki * self.integral
        D  = self.kd * (e - self.last_error) / dt
        FF = self.kff * self._trend() if abs(self.last_w) > self.turn_threshold else 0.0

        w_pid = -(P + I + D + FF)
        w_pid = max(-self.max_w, min(self.max_w, w_pid))

        cmd.linear.x  = self.v
        cmd.angular.z = self._smooth(w_pid, alpha=0.30)
        self.pub.publish(cmd)

        corner = self._track_corner(cmd.angular.z, dt)
        if corner:
            self.yellow_streak = 0
            self.get_logger().info('[ESQUINA] >90° detectados')

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
