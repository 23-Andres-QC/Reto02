#!/usr/bin/env python3
"""CapyTown lane_controller - RC-2.

Secuencia de arranque:
  1. Detecta amarillo y/o blanco por primera vez → arranca cronómetro de
     start_delay (5s), sin calibración activa: no gira en el sitio, solo
     espera quieto a que pase el tiempo.
  2. Termina la espera → avanza con PID directo.

La posición lateral (error) la marca la banda INFERIOR de detección (la más
cercana al robot); las bandas superior y central solo alimentan la pendiente
(/lane_slope) usada para anticipar curvas — ver lane_detector.py.

Regla simple de avance: el carrito SOLO avanza si detecta amarillo O blanco.
Si no detecta NINGUNO de los dos colores por más de `error_timeout`, FRENA
por completo (no avanza, no gira buscando) y se queda quieto hasta volver a
detectar cualquiera de los dos colores — ahí retoma el PID normal de inmediato.

IMU: solo se usa para el log de diagnóstico de posición, no en la ley de
control — un rumbo objetivo fijo no sigue al carril después de una esquina
real y termina compitiendo con la corrección visual.

Convención: error > 0 → desplazado derecha → ω < 0
            error < 0 → desplazado izquierda → ω > 0
"""

import math

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry


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
            ('integral_limit',  0.5),
            ('error_timeout',   0.5),    # sin amarillo NI blanco por más de esto → frena
            ('control_rate',    30.0),
            ('start_delay',     5.0),
            ('imu_topic',       '/imu'),
            ('odom_topic',      '/odom_raw'),
            ('calib_tolerance', 0.025),  # m — error/pendiente máximos para considerar "centrado" (uso: salida de esquina)
            ('curve_speed_factor', 0.6), # reduce velocidad lineal 40% siempre
            ('slope_curve_threshold', 0.04),  # m — pendiente mínima para anticipar curva (predictivo)
            ('sharp_turn_slope_threshold', 0.13),  # m — pendiente que indica esquina ~90° real (antes 0.09, muy temprano)
            ('sharp_turn_kp_slope',   3.0),    # ganancia del giro sobre la pendiente del amarillo (antes tasa fija sharp_turn_w)
            ('sharp_turn_kp_e',       2.5),    # ganancia sobre el error lateral — cierra el giro
            ('sharp_turn_max_w',      0.80),   # rad/s — tope del giro de esquina (base + corrección)
            ('sharp_turn_speed_factor', 0.3),  # avance muy reducido mientras gira en la esquina
            ('max_anticipation_time', 0.8),    # s — tope de tiempo anticipando antes de forzar el giro cerrado
        ])

        gp = self.get_parameter
        self.kp          = float(gp('kp').value)
        self.ki          = float(gp('ki').value)
        self.kd          = float(gp('kd').value)
        self.kff         = float(gp('kff').value)
        self.v           = float(gp('linear_speed').value)
        self.max_w       = float(gp('max_angular').value)
        self.i_limit     = float(gp('integral_limit').value)
        self.timeout     = float(gp('error_timeout').value)
        self.start_delay = float(gp('start_delay').value)
        rate             = float(gp('control_rate').value)
        imu_topic        = str(gp('imu_topic').value)
        odom_topic       = str(gp('odom_topic').value)
        self.calib_tolerance     = float(gp('calib_tolerance').value)
        self.curve_speed_factor   = float(gp('curve_speed_factor').value)
        self.slope_curve_threshold = float(gp('slope_curve_threshold').value)
        self.sharp_turn_slope_threshold = float(gp('sharp_turn_slope_threshold').value)
        self.sharp_turn_kp_slope         = float(gp('sharp_turn_kp_slope').value)
        self.sharp_turn_kp_e            = float(gp('sharp_turn_kp_e').value)
        self.sharp_turn_max_w           = float(gp('sharp_turn_max_w').value)
        self.sharp_turn_speed_factor    = float(gp('sharp_turn_speed_factor').value)
        self.max_anticipation_time      = float(gp('max_anticipation_time').value)

        self.error         = None
        self.slope          = 0.0   # pendiente de la línea guía (0 = recta, sin dato aún)
        self.last_error    = 0.0
        self.smooth_w      = 0.0
        self.integral      = 0.0
        self.initialized   = False
        self.start_time    = None
        self.last_stamp    = self.get_clock().now()
        self.last_rx       = self.get_clock().now()
        self.anticipation_timer = 0.0   # tiempo acumulado anticipando una curva sin resolver

        # IMU — yaw (solo diagnóstico/log, no se usa en la ley de control:
        # un rumbo fijo capturado una vez no sigue al carril después de un
        # giro real, y termina compitiendo con la corrección visual del PID)
        self.yaw = None

        # Odometría — posición real x,y
        self.pos_x  = None
        self.pos_y  = None
        self.pos_x0 = None   # origen registrado al iniciar
        self.pos_y0 = None

        # Esquina real: giro lento dedicado hasta reencontrar línea recta
        self.in_sharp_turn = False

        self.sub_err   = self.create_subscription(Float32, '/lane_error', self.on_error, 10)
        self.sub_slope = self.create_subscription(Float32, '/lane_slope', self.on_slope, 10)
        self.sub_imu  = self.create_subscription(Imu, imu_topic, self.on_imu, 10)
        self.sub_odom = self.create_subscription(Odometry, odom_topic, self.on_odom, 10)
        self.pub      = self.create_publisher(Twist, '/cmd_vel', 10)
        self.timer    = self.create_timer(1.0 / rate, self.control_loop)
        self.log_timer = self.create_timer(0.5, self._log_position)

        self.get_logger().info(
            f'lane_controller — v={self.v:.3f} kp={self.kp} '
            f'error_timeout={self.timeout:.2f}s (sin color → frena)')

    # ------------------------------------------------------------------
    def on_imu(self, msg):
        self.yaw = quat_to_yaw(msg.orientation)

    def on_odom(self, msg):
        self.pos_x = msg.pose.pose.position.x
        self.pos_y = msg.pose.pose.position.y
        if self.pos_x0 is None and self.initialized:
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
        # OJO: solo actualizamos self.error con lecturas válidas (amarillo
        # y/o blanco detectado — el detector ya hace el fallback solo-amarillo
        # o solo-blanco). Si llega NaN (sin ningún color), NO lo pisamos —
        # así "age" (tiempo desde la última lectura válida) es la única
        # señal que decide si frenar, y self.error nunca queda en None
        # mientras age <= timeout.
        if not math.isnan(msg.data):
            self.error   = msg.data
            self.last_rx = self.get_clock().now()
            if not self.initialized:
                self.initialized = True
                self.start_time  = self.last_rx
                self.get_logger().info(
                    f'Color detectado — esperando {self.start_delay:.0f}s antes de avanzar...')

    def on_slope(self, msg):
        # Pendiente del amarillo (banda central vs inferior). Si llega NaN
        # (línea insuficiente para trazarla) se mantiene el último valor conocido.
        if not math.isnan(msg.data):
            self.slope = msg.data

    # ------------------------------------------------------------------
    def _smooth(self, target, alpha=0.25):
        self.smooth_w = (1.0 - alpha) * self.smooth_w + alpha * target
        return self.smooth_w

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

        # ── ESPERA (tras la primera detección, antes de avanzar) ─────
        # Sin calibración activa: no gira en el sitio buscando centrarse,
        # solo espera quieto a que pase start_delay y arranca con PID
        # directo usando la posición real en la que está.
        if (now - self.start_time).nanoseconds * 1e-9 < self.start_delay:
            self.pub.publish(Twist())
            return

        age = (now - self.last_rx).nanoseconds * 1e-9
        cmd = Twist()

        # ── SIN COLOR (ni amarillo ni blanco): FRENA ─────────────────────
        # El carrito solo avanza si detecta amarillo O blanco. Si no detecta
        # ninguno de los dos colores, frena por completo (no avanza, no gira
        # buscando) y se queda quieto hasta que vuelva a detectar algo.
        if age > self.timeout:
            self.integral = 0.0
            self.smooth_w = 0.0
            self.in_sharp_turn = False
            self.anticipation_timer = 0.0
            self.pub.publish(Twist())   # frena: linear=0, angular=0
            return

        e = self.error
        if abs(e) < 0.01:
            e = 0.0

        # ── ESQUINA real (la pista tiene esquinas marcadas, no curvas suaves
        # continuas) ──────────────────────────────────────────────────
        # Si la pendiente crece mucho (la línea se va casi de canto), no es
        # una curva suave a corregir con FF — es una esquina real. Entra en
        # un giro lento y dedicado en la dirección de la pendiente, y se
        # mantiene girando hasta volver a ver la línea recta y centrada
        # (la "siguiente" línea amarilla/blanca tras la esquina) — ahí frena
        # el giro y vuelve al PID normal. NO gira un ángulo fijo (ni 90° ni
        # ningún otro): gira lo que haga falta, frame a frame, hasta volver
        # a encontrar el centro al otro lado — la duración del giro la decide
        # únicamente la condición de salida (slope+e bajos), nunca un ángulo
        # acumulado.
        #
        # Además del umbral por MAGNITUD (sharp_turn_slope_threshold), hay un
        # umbral por TIEMPO: si lleva "anticipando" (|slope| > slope_curve_
        # threshold, corrección FF suave) más de max_anticipation_time
        # seguido sin resolver, se fuerza el giro cerrado de todas formas —
        # antes se quedaba anticipando con el FF suave hasta 2s antes de
        # comprometerse al giro real, lo cual se sentía como un giro
        # adelantado/abierto. Esto limita esa ventana de anticipación.
        anticipating_now = abs(self.slope) > self.slope_curve_threshold
        if anticipating_now:
            self.anticipation_timer += dt
        else:
            self.anticipation_timer = 0.0

        if (abs(self.slope) > self.sharp_turn_slope_threshold
                or self.anticipation_timer > self.max_anticipation_time):
            self.in_sharp_turn = True
            self.anticipation_timer = 0.0   # ya se comprometió al giro, no sigue acumulando

        if self.in_sharp_turn:
            # El giro NO es a un ritmo fijo/preprogramado (eso generaba un
            # giro de radio constante — "abierto" — que no necesariamente
            # converge al centro real, perdiendo el amarillo en vez de
            # seguirlo, y no reflejaba lo que la cámara realmente ve). El
            # giro se deriva directamente de la pendiente ACTUAL del
            # amarillo (sharp_turn_kp_slope * slope): si la línea está muy
            # de canto, gira fuerte; si ya casi se enderezó, gira poco —
            # proporcional a lo que el amarillo muestra en cada frame, no
            # a una tasa fija. Se suma la corrección sobre el error lateral
            # ACTUAL para converger al centro si llegó desviado a la esquina.
            w_target = -(self.sharp_turn_kp_slope * self.slope + self.sharp_turn_kp_e * e)
            w_target = max(-self.sharp_turn_max_w, min(self.sharp_turn_max_w, w_target))
            cmd.angular.z = self._smooth(w_target, alpha=0.10)
            cmd.linear.x  = self.v * self.sharp_turn_speed_factor
            self.pub.publish(cmd)
            # Salir del giro: línea ya recta (slope bajo) y centrada (e bajo)
            # — es decir, ya llegó al centro de la proyección de las líneas
            # nuevas, no solo "se ve recta" por casualidad de ángulo.
            if abs(self.slope) < self.slope_curve_threshold and abs(e) < self.calib_tolerance:
                self.in_sharp_turn = False
            return

        # ── AVANCE con PID — una sola ley de control, sin saltos de modo ──
        P = self.kp * e
        self.integral += e * dt
        self.integral  = max(-self.i_limit, min(self.i_limit, self.integral))
        I  = self.ki * self.integral
        D  = self.kd * (e - self.last_error) / dt

        # Anticipación de curva: pendiente entre el punto CENTRAL e INFERIOR de
        # la línea guía (no superior-inferior — el punto lejano anticipaba el
        # giro demasiado pronto). Es una lectura del frame actual, sin retraso.
        # Usar `trend` aquí sería tardío: se calcula acumulando varias muestras
        # de error en el tiempo (~0.5s de ventana), así que la corrección
        # llegaba tarde aunque la detección de curva ya fuera inmediata. Por
        # eso el FF usa slope directamente.
        anticipa_curva = abs(self.slope) > self.slope_curve_threshold
        FF = self.kff * self.slope if anticipa_curva else 0.0

        # SIN término de rumbo fijo (yaw_term, eliminado): un rumbo objetivo
        # capturado una sola vez al arrancar no sigue al carril después de
        # una esquina real (el carril ya giró ~90°, pero ese rumbo "objetivo"
        # seguía siendo el de antes de la esquina) — competía con esta misma
        # ley de control y producía zigzag sostenido. El control depende
        # 100% de lo que la cámara ve en cada frame (e, slope), nunca de un
        # plan fijo.
        w_pid = -(P + I + D + FF)
        w_pid = max(-self.max_w, min(self.max_w, w_pid))

        # Velocidad reducida 40% siempre (no solo en curvas) — margen de
        # reacción y corrección más cómodo en todo el recorrido.
        cmd.linear.x   = self.v * self.curve_speed_factor
        cmd.angular.z  = self._smooth(w_pid, alpha=0.12)   # transición lenta — calibración gradual
        self.pub.publish(cmd)

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
