#!/usr/bin/env python3
"""CapyTown lane_controller - Semana 11 (RC-2).

Controlador PID + feed-forward predictivo sobre /lane_error → publica /cmd_vel.

Convención de signo:
  error > 0  →  línea a la DERECHA  →  ω < 0 (girar derecha)
  error < 0  →  línea a la IZQUIERDA →  ω > 0 (girar izquierda)

Predicción de giro:
  Mantiene un historial de errores. Si la tendencia (pendiente) es sostenida
  en un sentido, añade un término feed-forward para pre-girar antes de perder
  la línea.
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
            ('kp',             2.5),
            ('ki',             0.0),
            ('kd',             0.3),
            ('kff',            1.0),   # ganancia feed-forward de tendencia
            ('linear_speed',   0.20),
            ('max_angular',    2.0),
            ('integral_limit', 0.5),
            ('error_timeout',  0.5),
            ('control_rate',   30.0),
            ('recovery_w',     0.6),
            ('recovery_v',     0.0),
            ('history_size',   10),    # muestras para calcular tendencia (~0.33 s a 30 Hz)
            ('turn_threshold', 0.3),   # |ω| mínimo para considerar que hay giro real
        ])

        gp               = self.get_parameter
        self.kp          = float(gp('kp').value)
        self.ki          = float(gp('ki').value)
        self.kd          = float(gp('kd').value)
        self.kff         = float(gp('kff').value)
        self.v           = float(gp('linear_speed').value)
        self.max_w       = float(gp('max_angular').value)
        self.i_limit     = float(gp('integral_limit').value)
        self.timeout     = float(gp('error_timeout').value)
        self.recovery_w     = float(gp('recovery_w').value)
        self.recovery_v     = float(gp('recovery_v').value)
        self.turn_threshold = float(gp('turn_threshold').value)
        hist                = int(gp('history_size').value)
        rate                = float(gp('control_rate').value)

        self.error        = None
        self.last_error   = 0.0
        self.last_w       = 0.0
        self.integral     = 0.0
        self.initialized  = False
        self.last_stamp   = self.get_clock().now()
        self.last_rx      = self.get_clock().now()
        self.error_history = deque(maxlen=hist)

        self.sub   = self.create_subscription(
            Float32, '/lane_error', self.on_error, 10)
        self.pub   = self.create_publisher(Twist, '/cmd_vel', 10)
        self.timer = self.create_timer(1.0 / rate, self.control_loop)

        self.get_logger().info('lane_controller listo.')
        self.get_logger().info(
            f'PID kp={self.kp} ki={self.ki} kd={self.kd} kff={self.kff}  '
            f'v={self.v} m/s  max_w={self.max_w} rad/s  '
            f'recovery_w={self.recovery_w} rad/s')

    def on_error(self, msg):
        if not math.isnan(msg.data):
            self.error       = msg.data
            self.last_rx     = self.get_clock().now()
            self.initialized = True

    def _trend(self):
        """Pendiente media del historial de errores (m por muestra)."""
        n = len(self.error_history)
        if n < 3:
            return 0.0
        lst = list(self.error_history)
        return (lst[-1] - lst[0]) / n

    def control_loop(self):
        now = self.get_clock().now()
        dt  = (now - self.last_stamp).nanoseconds * 1e-9
        self.last_stamp = now

        if dt <= 0.0:
            return

        age = (now - self.last_rx).nanoseconds * 1e-9

        # Esperar primera detección
        if not self.initialized:
            self.pub.publish(Twist())
            return

        # Línea perdida → recovery: continuar girando en el mismo sentido
        if age > self.timeout:
            self.integral = 0.0
            self.error_history.clear()
            cmd = Twist()
            cmd.linear.x  = self.recovery_v
            cmd.angular.z = math.copysign(self.recovery_w, self.last_w) if self.last_w != 0.0 else -self.recovery_w
            self.pub.publish(cmd)
            return

        e = self.error
        self.error_history.append(e)

        # P
        P = self.kp * e

        # I con anti-windup
        self.integral += e * dt
        self.integral  = max(-self.i_limit, min(self.i_limit, self.integral))
        I = self.ki * self.integral

        # D
        derivative = (e - self.last_error) / dt
        D = self.kd * derivative

        # FF: solo anticipa si el robot YA está girando (giro real, no offset recto)
        # Si va recto con la línea a un lado, el PID corrige sin FF
        trend = self._trend()
        if abs(self.last_w) > self.turn_threshold:
            FF = self.kff * trend
        else:
            FF = 0.0

        # ω total: negado porque error>0 → girar derecha → ω<0
        w = -(P + I + D + FF)
        w = max(-self.max_w, min(self.max_w, w))

        cmd           = Twist()
        cmd.linear.x  = self.v
        cmd.angular.z = w
        self.pub.publish(cmd)

        self.last_error = e
        self.last_w     = w


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
