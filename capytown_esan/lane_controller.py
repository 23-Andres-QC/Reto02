#!/usr/bin/env python3
"""CapyTown lane_controller - RC-2.

Secuencia de arranque:
  1. Detecta amarillo por primera vez → entra en CALIBRACIÓN ACTIVA
  2. CALIBRACIÓN ACTIVA: ajustes angulares suaves, SIN avanzar, hasta que
     el error (y su tendencia) estén cerca de cero varios frames seguidos
     → posición x,y correcta, ángulo recto respecto al amarillo/línea planeada
  3. Calibrado → arranca cronómetro de start_delay (5s), sigue sin avanzar
  4. Termina la espera → recién entonces avanza con PID

Una sola ley de control PID continua. Estados según el tiempo sin
detección válida de amarillo (`age`):

  age <= error_timeout (0.8s)                    → PID normal
  error_timeout < age <= search_timeout (2.2s)    → pérdida breve: PID con
                                                      el último error congelado
                                                      (sin tocar integral/derivada)
  age > search_timeout                            → búsqueda real: gira
                                                      izquierda con rampa lenta,
                                                      corrigiendo hacia el yaw
                                                      inicial (IMU) para no
                                                      desviarse del heading
                                                      original mientras busca

IMU: registra yaw inicial cuando termina la espera de arranque (start_delay).

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
from nav_msgs.msg import Odometry

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
            ('kp',              2.2),
            ('ki',              0.12),
            ('kd',              0.25),
            ('kff',             0.6),
            ('linear_speed',    0.30),   # requisito de competencia ≥ 0.2 m/s, con margen
            ('max_angular',     2.0),
            ('calib_w',         0.20),
            ('drift_w',         0.15),   # rad/s giro suave buscando amarillo
            ('integral_limit',  0.5),
            ('error_timeout',   0.8),    # pérdida breve: congela última corrección, sigue recto
            ('search_timeout',  2.2),    # pérdida sostenida: recién aquí entra en búsqueda activa
            ('control_rate',    30.0),
            ('history_size',    15),
            ('start_delay',     5.0),
            ('imu_topic',       '/imu'),
            ('odom_topic',      '/odom_raw'),
            ('yaw_weight',      0.3),    # peso del rumbo planeado (yaw inicial) en avance normal
            ('calib_tolerance', 0.025),  # m — error máximo para considerar "calibrado"
            ('calib_stable_frames', 8),  # frames consecutivos centrado+quieto para confirmar calibración
            ('calib_kp',        2.0),    # ganancia angular durante calibración inicial (sin avanzar)
            ('slope_tolerance', 0.03),   # m — pendiente máx. de la línea guía para considerar "recto"
            ('calib_kp_slope',  2.0),    # ganancia angular sobre la pendiente durante calibración
            ('calib_min_w',     0.12),   # rad/s — piso mínimo para superar zona muerta del motor
            ('curve_speed_factor', 0.6), # reduce velocidad lineal 40% siempre
            ('slope_curve_threshold', 0.04),  # m — pendiente mínima para anticipar curva (predictivo)
            ('sharp_turn_slope_threshold', 0.09),  # m — pendiente que indica esquina ~90° real
            ('sharp_turn_w',          0.40),   # rad/s — giro lento dedicado en la esquina
            ('sharp_turn_speed_factor', 0.3),  # avance muy reducido mientras gira en la esquina
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
        self.i_limit     = float(gp('integral_limit').value)
        self.timeout     = float(gp('error_timeout').value)
        self.search_timeout = float(gp('search_timeout').value)
        self.start_delay = float(gp('start_delay').value)
        hist             = int(gp('history_size').value)
        rate             = float(gp('control_rate').value)
        imu_topic        = str(gp('imu_topic').value)
        odom_topic       = str(gp('odom_topic').value)
        self.yaw_weight  = float(gp('yaw_weight').value)
        self.calib_tolerance     = float(gp('calib_tolerance').value)
        self.calib_stable_frames = int(gp('calib_stable_frames').value)
        self.calib_kp             = float(gp('calib_kp').value)
        self.slope_tolerance      = float(gp('slope_tolerance').value)
        self.calib_kp_slope       = float(gp('calib_kp_slope').value)
        self.calib_min_w          = float(gp('calib_min_w').value)
        self.curve_speed_factor   = float(gp('curve_speed_factor').value)
        self.slope_curve_threshold = float(gp('slope_curve_threshold').value)
        self.sharp_turn_slope_threshold = float(gp('sharp_turn_slope_threshold').value)
        self.sharp_turn_w               = float(gp('sharp_turn_w').value)
        self.sharp_turn_speed_factor    = float(gp('sharp_turn_speed_factor').value)

        self.error         = None
        self.slope          = 0.0   # pendiente de la línea guía (0 = recta, sin dato aún)
        self.last_error    = 0.0
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

        # Odometría — posición real x,y
        self.pos_x  = None
        self.pos_y  = None
        self.pos_x0 = None   # origen registrado al iniciar
        self.pos_y0 = None

        # Acumulador esquina (informativo, no cambia la ley de control)
        self.cum_angle      = 0.0
        self.last_turn_sign = 0
        self.in_corner      = False

        # Esquina ~90° real: giro lento dedicado hasta reencontrar línea recta
        self.in_sharp_turn = False

        # Calibración inicial: bias de error capturado al terminar start_delay.
        self.calib_bias = None

        # Fase de calibración ACTIVA (antes de start_delay): el robot ajusta su
        # ángulo (sin avanzar) hasta que el error de cámara y su tendencia estén
        # cerca de cero — recién ahí se considera "alineado y centrado" y arranca
        # el cronómetro de start_delay.
        self.pre_calibrated    = False
        self.calib_stable_count = 0
        self.calib_smooth_w     = 0.0

        self.sub_err   = self.create_subscription(Float32, '/lane_error', self.on_error, 10)
        self.sub_slope = self.create_subscription(Float32, '/lane_slope', self.on_slope, 10)
        self.sub_imu  = self.create_subscription(Imu, imu_topic, self.on_imu, 10)
        self.sub_odom = self.create_subscription(Odometry, odom_topic, self.on_odom, 10)
        self.pub      = self.create_publisher(Twist, '/cmd_vel', 10)
        self.timer    = self.create_timer(1.0 / rate, self.control_loop)
        self.log_timer = self.create_timer(0.5, self._log_position)

        self.get_logger().info(
            f'lane_controller — v={self.v:.3f} calib_w={self.calib_w:.2f} '
            f'drift_w={self.drift_w:.2f} kp={self.kp}')

    # ------------------------------------------------------------------
    def on_imu(self, msg):
        self.yaw = quat_to_yaw(msg.orientation)
        # Solo registrar el yaw inicial DESPUÉS de la calibración activa + start_delay
        if self.initial_yaw is None and self.pre_calibrated:
            elapsed = (self.get_clock().now() - self.start_time).nanoseconds * 1e-9
            if elapsed >= self.start_delay:
                self.initial_yaw = self.yaw
                self.get_logger().info(f'Yaw inicial registrado: {math.degrees(self.yaw):.1f}°')

    def on_odom(self, msg):
        self.pos_x = msg.pose.pose.position.x
        self.pos_y = msg.pose.pose.position.y
        if self.pos_x0 is None and self.pre_calibrated:
            elapsed = (self.get_clock().now() - self.start_time).nanoseconds * 1e-9
            if elapsed >= self.start_delay:
                self.pos_x0 = self.pos_x
                self.pos_y0 = self.pos_y
                self.get_logger().info(f'Posición inicial registrada: ({self.pos_x0:.3f}, {self.pos_y0:.3f})')

    def _log_position(self):
        if self.pos_x is None:
            self.get_logger().info('Posición robot: sin odometría aún')
            return
        rel_x = self.pos_x - self.pos_x0 if self.pos_x0 is not None else self.pos_x
        rel_y = self.pos_y - self.pos_y0 if self.pos_y0 is not None else self.pos_y
        yaw_deg = math.degrees(self.yaw) if self.yaw is not None else float('nan')
        self.get_logger().info(
            f'Posición robot: x={rel_x:+.3f}m y={rel_y:+.3f}m yaw={yaw_deg:.1f}°')

    def on_error(self, msg):
        # OJO: solo actualizamos self.error con lecturas válidas.
        # Si llega NaN, NO lo pisamos — así "age" (tiempo desde la última
        # lectura válida) es la única señal que decide modo búsqueda,
        # y self.error nunca queda en None mientras age <= timeout.
        if not math.isnan(msg.data):
            self.error   = msg.data
            self.last_rx = self.get_clock().now()
            if not self.initialized:
                self.initialized = True
                self.get_logger().info('Amarillo detectado — calibrando posición inicial...')

    def on_slope(self, msg):
        # Pendiente de la línea guía (top vs bottom). Si llega NaN (línea
        # insuficiente para trazarla) se mantiene el último valor conocido.
        if not math.isnan(msg.data):
            self.slope = msg.data

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

        # ── SIN DETECCIÓN AÚN ────────────────────────────────────────
        if not self.initialized:
            self.pub.publish(Twist())
            return

        # ── CALIBRACIÓN ACTIVA: busca su punto de partida ────────────
        # El robot llega mirando al amarillo pero no necesariamente bien
        # alineado/centrado. Aquí NO avanza — solo hace ajustes angulares
        # suaves hasta que el error (y su tendencia, para detectar que no
        # sigue derivando) estén cerca de cero durante varios frames
        # consecutivos. Recién ahí se considera "calibrado": posición x,y
        # correcta, ángulo recto respecto al amarillo y a la línea de
        # recorrido, distancia exacta amarillo↔centro.
        if not self.pre_calibrated:
            e     = self.error if self.error is not None else 0.0
            slope = self.slope   # pendiente de la línea guía (0 = recta)
            self.error_history.append(e)
            trend = self._trend()

            centrado = abs(e) < self.calib_tolerance and abs(trend) < self.calib_tolerance
            recto    = abs(slope) < self.slope_tolerance

            if centrado and recto:
                self.calib_stable_count += 1
            else:
                self.calib_stable_count = 0

            if self.calib_stable_count >= self.calib_stable_frames:
                self.pre_calibrated = True
                self.calib_bias     = e            # bias residual al momento de calibrar
                self.start_time     = now           # arranca el cronómetro de start_delay AHORA
                self.error_history.clear()
                self.calib_smooth_w = 0.0
                self.get_logger().info(
                    f'Calibración inicial lograda (error={e*100:+.2f}cm, '
                    f'pendiente={slope*100:+.2f}cm) — esperando {self.start_delay:.0f}s antes de avanzar...')
                self.pub.publish(Twist())
                return

            # Corrige tanto el centrado (e) como la rectitud (slope) de la línea guía
            w_calib = -(self.calib_kp * e + self.calib_kp_slope * slope)
            # Piso mínimo: si hace falta corregir pero el valor proporcional es muy
            # chico, el motor real puede no moverse (zona muerta/fricción estática).
            # Se garantiza una magnitud mínima efectiva, conservando el signo.
            if abs(w_calib) > 1e-4 and abs(w_calib) < self.calib_min_w:
                w_calib = math.copysign(self.calib_min_w, w_calib)
            w_calib = max(-self.calib_w, min(self.calib_w, w_calib))
            self.calib_smooth_w = 0.85 * self.calib_smooth_w + 0.15 * w_calib
            cmd = Twist()
            cmd.angular.z = self.calib_smooth_w
            cmd.linear.x  = 0.0   # nunca avanza durante la calibración inicial
            self.pub.publish(cmd)
            return

        # ── ESPERA (tras calibrar, antes de avanzar) ─────────────────
        if (now - self.start_time).nanoseconds * 1e-9 < self.start_delay:
            self.pub.publish(Twist())
            return

        age = (now - self.last_rx).nanoseconds * 1e-9
        cmd = Twist()

        # ── PÉRDIDA SOSTENIDA (> search_timeout): recién aquí búsqueda activa ──
        # Esto es la curva genuina / fuera de pista real. Rampa MUY lenta
        # (alpha bajo) para no generar un giro brusco que cambie la posición.
        if age > self.search_timeout:
            self.integral = 0.0
            self.error_history.clear()
            # Corrección hacia el yaw inicial mientras busca, para no
            # desviarse del heading original durante la búsqueda
            yaw_corr = self._yaw_correction()
            w_target = self.drift_w + yaw_corr
            w_target = max(-self.calib_w, min(self.calib_w, w_target))
            cmd.angular.z = self._smooth(w_target, alpha=0.06)
            cmd.linear.x  = self.v * 0.4
            self.pub.publish(cmd)
            self._track_corner(cmd.angular.z, dt)
            return

        # ── PÉRDIDA BREVE (timeout < age <= search_timeout): NO cambia de modo ──
        # Se sigue usando la última lectura válida conocida (self.error, congelada)
        # en la misma ley PID. Esto evita el salto que generaba el zigzag:
        # antes, perder el amarillo un instante disparaba un giro fijo que
        # cambiaba la posición real del robot y generaba el error opuesto.
        stale = age > self.timeout

        e = self.error - self.calib_bias   # corrige contra el bias capturado al inicio
        if abs(e) < 0.01:
            e = 0.0

        # ── ESQUINA ~90° (track con curvas de 90°, no continuas) ────────
        # Si la pendiente crece mucho (la línea se va casi de canto), no es
        # una curva suave a corregir con FF — es una esquina real. Entra en
        # un giro lento y dedicado en la dirección de la pendiente, y se
        # mantiene girando hasta volver a ver la línea recta y centrada
        # (la "siguiente" línea amarilla/blanca tras la esquina) — ahí frena
        # el giro y vuelve al PID normal.
        if abs(self.slope) > self.sharp_turn_slope_threshold:
            self.in_sharp_turn = True

        if self.in_sharp_turn:
            turn_dir = math.copysign(1.0, self.slope) if self.slope != 0.0 else 1.0
            w_target = -turn_dir * self.sharp_turn_w   # mismo signo que la corrección normal
            cmd.angular.z = self._smooth(w_target, alpha=0.10)
            cmd.linear.x  = self.v * self.sharp_turn_speed_factor
            self.pub.publish(cmd)
            self._track_corner(cmd.angular.z, dt)
            # Salir del giro: línea ya recta (slope bajo) y centrada (e bajo)
            if abs(self.slope) < self.slope_curve_threshold and abs(e) < self.calib_tolerance:
                self.in_sharp_turn = False
            return

        # ── AVANCE con PID — una sola ley de control, sin saltos de modo ──
        P = self.kp * e
        if not stale:
            self.integral += e * dt
            self.integral  = max(-self.i_limit, min(self.i_limit, self.integral))
        I  = self.ki * self.integral
        D  = self.kd * (e - self.last_error) / dt if not stale else 0.0

        # Anticipación de curva: pendiente entre el punto CENTRAL e INFERIOR de
        # la línea guía (no superior-inferior — el punto lejano anticipaba el
        # giro demasiado pronto). Es una lectura del frame actual, sin retraso.
        # Usar `trend` aquí sería tardío: se calcula acumulando varias muestras
        # de error en el tiempo (~0.5s de ventana), así que la corrección
        # llegaba tarde aunque la detección de curva ya fuera inmediata. Por
        # eso el FF usa slope directamente.
        anticipa_curva = abs(self.slope) > self.slope_curve_threshold
        FF = self.kff * self.slope if anticipa_curva else 0.0

        # Corrección complementaria de bajo peso hacia la línea recta planeada
        # en la calibración inicial (initial_yaw). NO es una línea rígida: solo
        # actúa en tramos rectos para evitar deriva lenta; se desactiva igual
        # que el feed-forward cuando se anticipa una curva, para no resistir
        # el giro que la cámara ya está viendo venir.
        yaw_term = (self._yaw_correction() * self.yaw_weight) if not anticipa_curva else 0.0

        w_pid = -(P + I + D + FF) + yaw_term
        w_pid = max(-self.max_w, min(self.max_w, w_pid))

        self.in_corner = False   # si sigue viendo amarillo, ya salió de la curva
        # Velocidad reducida 40% siempre (no solo en curvas) — margen de
        # reacción y corrección más cómodo en todo el recorrido.
        cmd.linear.x   = self.v * self.curve_speed_factor
        cmd.angular.z  = self._smooth(w_pid, alpha=0.12)   # transición lenta — calibración gradual
        self.pub.publish(cmd)

        self._track_corner(cmd.angular.z, dt)

        if not stale:
            self.last_error = e


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
