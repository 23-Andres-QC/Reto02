#!/usr/bin/env python3
"""CapyTown lane_controller - RC-2.

Modos:
  ESPERA    : espera 5s
  AVANCE    : amarillo visible → avanza + PID
  DERIVA    : tendencia creciente de error → micro-correcciones angulares SIN frenar
  BUSCA     : amarillo perdido + curva → para + gira izquierda suave

IMU: registra yaw inicial cuando arranca.  Durante DERIVA usa el yaw actual
     como referencia adicional para no alejarse del ángulo de partida.

Convención: error > 0 → desplazado derecha → ω < 0
            error < 0 → desplazado izquierda → ω > 0
"""

import math
from collections import deque

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu

CORNER_THRESHOLD = math.pi / 2.0   # 90° en rad


def quat_to_yaw(q):
    """Quaternion → yaw (rad)."""
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


class LaneController(Node):

    def __init__(self):
        super().__init__('lane_controller')

        self.declare_parameters('', [
            ('kp',              4.0),
            ('ki',              0.3),
            ('kd',              0.4),
            ('kff',             1.0),
            ('linear_speed',    0.21),
            ('max_angular',     2.0),
            ('calib_w',         0.20),
            ('drift_w',         0.15),   # rad/s giro suave buscando amarillo
            ('drift_trend',     0.004),  # umbral de tendencia para activar DERIVA
            ('integral_limit',  0.5),
            ('error_timeout',   0.8),
            ('control_rate',    30.0),
            ('turn_threshold',  0.3),
            ('history_size',    15),
            ('start_delay',     5.0),
            ('imu_topic',       '/imu'),
        ])

        gp = self.get_parameter
        self.kp          = float(gp('kp').value)
        self.ki          = float(gp('ki').value)
        self.kd          = float(gp('kd').value)
        self.kff         = float(gp('kff').value)
        self.v           = float(gp('linear_speed').value)
        self.max_w       = float(gp('max_angular').value)
        self.calib_w     = float(gp('calib_w').value)
        self.drift_w     = float(gp('drift_w').value)
        self.drift_trend = float(gp('drift_trend').value)
        self.i_limit     = float(gp('integral_limit').value)
        self.timeout     = float(gp('error_timeout').value)
        self.turn_thr    = float(gp('turn_threshold').value)
        self.start_delay = float(gp('start_delay').value)
        hist             = int(gp('history_size').value)
        rate             = float(gp('control_rate').value)
        imu_topic        = str(gp('imu_topic').value)

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

        # IMU — yaw
        self.yaw         = None
        self.initial_yaw = None   # yaw cuando el robot arranca

        # Acumulador esquina
        self.cum_angle      = 0.0
        self.last_turn_sign = 0
        self.in_corner      = False

        self.sub_err = self.create_subscription(Float32, '/lane_error', self.on_error, 10)
        self.sub_imu = self.create_subscription(Imu, imu_topic, self.on_imu, 10)
        self.pub     = self.create_publisher(Twist, '/cmd_vel', 10)
        self.timer   = self.create_timer(1.0 / rate, self.control_loop)

        self.get_logger().info(
            f'lane_controller — v={self.v:.3f} calib_w={self.calib_w:.2f} '
            f'drift_w={self.drift_w:.2f} drift_trend={self.drift_trend}')

    # ------------------------------------------------------------------
    def on_imu(self, msg):
        self.yaw = quat_to_yaw(msg.orientation)
        if self.initial_yaw is None and self.initialized:
            elapsed = (self.get_clock().now() - self.start_time).nanoseconds * 1e-9
            if elapsed >= self.start_delay:
                self.initial_yaw = self.yaw
                self.get_logger().info(f'Yaw inicial registrado: {math.degrees(self.yaw):.1f}°')

    def on_error(self, msg):
        if not math.isnan(msg.data):
            self.error   = msg.data
            self.last_rx = self.get_clock().now()
            if not self.initialized:
                self.initialized = True
                self.start_time  = self.get_clock().now()
                self.get_logger().info(f'Amarillo — esperando {self.start_delay:.0f}s...')
        else:
            self.error = None

    # ------------------------------------------------------------------
    def _trend(self):
        lst = [x for x in self.error_history if x is not None]
        n = len(lst)
        if n < 4:
            return 0.0
        return (lst[-1] - lst[0]) / n

    def _smooth(self, target, alpha=0.25):
        self.smooth_w = (1.0 - alpha) * self.smooth_w + alpha * target
        return self.smooth_w

    def _track_corner(self, w, dt):
        if abs(w) < 0.05:
            self.cum_angle      = 0.0
            self.last_turn_sign = 0
            self.in_corner      = False
            return False
        sign = 1 if w > 0 else -1
        if sign != self.last_turn_sign:
            self.cum_angle      = 0.0
            self.last_turn_sign = sign
        self.cum_angle += abs(w) * dt
        if self.cum_angle >= CORNER_THRESHOLD:
            self.cum_angle      = 0.0
            self.last_turn_sign = 0
            self.in_corner      = True
            return True
        return False

    def _yaw_correction(self):
        """Corrección angular suave basada en desviación del yaw inicial."""
        if self.yaw is None or self.initial_yaw is None:
            return 0.0
        delta = self.yaw - self.initial_yaw
        # Normalizar a [-π, π]
        delta = (delta + math.pi) % (2 * math.pi) - math.pi
        # Ganancia baja: corrección de 0.05 rad/s por cada grado de desviación
        return -delta * 0.8

    # ------------------------------------------------------------------
    def control_loop(self):
        now = self.get_clock().now()
        dt  = (now - self.last_stamp).nanoseconds * 1e-9
        self.last_stamp = now
        if dt <= 0.0:
            return

        # ── ESPERA ───────────────────────────────────────────────────
        if not self.initialized:
            self.pub.publish(Twist())
            return
        if (now - self.start_time).nanoseconds * 1e-9 < self.start_delay:
            self.pub.publish(Twist())
            return

        age = (now - self.last_rx).nanoseconds * 1e-9
        cmd = Twist()

        # ── SIN AMARILLO: busca según última dirección conocida ──────
        if age > self.timeout:
            self.integral = 0.0
            self.error_history.clear()
            # Si el último error era negativo (robot iba hacia amarillo / demasiado izquierda)
            # → busca a la DERECHA para alejarse del amarillo
            # Si el último error era positivo (robot se alejó del amarillo)
            # → busca a la IZQUIERDA para volver a encontrarlo
            if self.last_error < -0.01:
                w_search = -self.drift_w   # gira derecha — estaba sobre el amarillo
            else:
                w_search = +self.drift_w   # gira izquierda — se alejó del amarillo
            yaw_corr      = self._yaw_correction()
            w_search      = max(-self.calib_w, min(self.calib_w, w_search + yaw_corr * 0.3))
            cmd.angular.z = self._smooth(w_search, alpha=0.12)
            cmd.linear.x  = 0.0 if self.in_corner else self.v * 0.4
            self.pub.publish(cmd)
            self._track_corner(cmd.angular.z, dt)
            self.last_w = cmd.angular.z
            return

        e = self.error
        self.error_history.append(e)
        trend = self._trend()

        # ── DETECTA DERIVA INCIPIENTE: tendencia creciente → micro-corrección anticipada ──
        drifting = abs(trend) > self.drift_trend

        # ── AVANCE con PID ───────────────────────────────────────────
        P = self.kp * e
        self.integral += e * dt
        self.integral  = max(-self.i_limit, min(self.i_limit, self.integral))
        I = self.ki * self.integral
        D = self.kd * (e - self.last_error) / dt

        if drifting:
            # Tendencia de perder amarillo: amplifica corrección, no usa FF para no sobregirar
            w_pid = -(P + I + D)
            # Limita a drift_w para que sea suave
            w_pid = max(-self.drift_w * 3, min(self.drift_w * 3, w_pid))
        else:
            FF    = self.kff * trend if abs(self.last_w) > self.turn_thr else 0.0
            w_pid = -(P + I + D + FF)
            w_pid = max(-self.max_w, min(self.max_w, w_pid))

        self.in_corner = False   # si sigue viendo amarillo, ya salió de la curva
        cmd.linear.x   = self.v
        cmd.angular.z  = self._smooth(w_pid, alpha=0.25 if drifting else 0.30)
        self.pub.publish(cmd)

        self._track_corner(cmd.angular.z, dt)

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
